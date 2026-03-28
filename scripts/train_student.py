#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distill a Shorkie student model from ensemble predictions.

Loads:
    Distillation/synthetic_{chrom}_mean_preds.npz

Required flag:
    --chrom CHROMOSOME_NAME
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------- Reproducibility ----------
import random, numpy as np
SEED = 42

os.environ["PYTHONHASHSEED"]         = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_bfloat16")
# ------------------------------------

import json
import argparse
from baskerville.seqnn import SeqNN
from src.data_loader import *

# ---------- CONSTANTS / PATHS ----------
PARAMS_JSON          = "Models/shorkie/params.json"
TRUNK_H5_TEMPLATE    = "Models/shorkie/f0/model_best.h5"        
NPZ_TEMPLATE         = "Results/Distillation/{chrom}/synthetic_{chrom}_mean_preds.npz"
OUT_WEIGHTS_TEMPLATE = "Results/Distillation/{chrom}/student_{chrom}_distilled.h5"

# --- Default hyperparameters (used in the paper) ---
WINDOW_BP = 16384  
BATCH_SIZE = 64
LR        = 2e-5
# ------------------------------------


def make_ds(seqs, labels, batch_size=BATCH_SIZE):
    """
    Dataset that yields (x, y):
        x: (WINDOW_BP, 170) bf16
        y: (896,) bf16
    """
    def gen():
        for s, y in zip(seqs, labels):
            x = build_shorkie_features(s)
            yield x, y.astype(np.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(WINDOW_BP, 170), dtype=tf.uint8),
            tf.TensorSpec(shape=(labels.shape[1],), dtype=tf.float32),
        ),
    )
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.bfloat16), tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
# ------------------------------------


# ---------- METRICS ----------
def pearson_r(y_true, y_pred):
    """
    Pearson correlation across bins per sequence, then averaged over batch.
    Shapes: (batch, 896)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    xm = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
    ym = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    num = tf.reduce_sum(xm * ym, axis=-1)
    den = tf.sqrt(
        tf.reduce_sum(xm * xm, axis=-1) * tf.reduce_sum(ym * ym, axis=-1)
    ) + 1e-8

    r = num / den
    return tf.reduce_mean(r)


def spearman_r(y_true, y_pred):
    """
    Spearman correlation via Pearson on ranks.
    Ranks are computed per sequence (axis=-1), then Pearson is applied batch-wise.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_rank = tf.argsort(tf.argsort(y_true, axis=-1), axis=-1)
    pred_rank = tf.argsort(tf.argsort(y_pred, axis=-1), axis=-1)

    true_rank = tf.cast(true_rank, tf.float32)
    pred_rank = tf.cast(pred_rank, tf.float32)

    return pearson_r(true_rank, pred_rank)
# ------------------------------------


# ---------- ARG PARSER ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Distill a Shorkie student on ensemble outputs (synthetic train set only)."
    )
    p.add_argument(
        "--chrom",
        required=True,
        type=str,
        help="Chromosome name used in synthetic_{chrom}_mean_preds.npz",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train for.",
    )
    return p.parse_args()
# ------------------------------------

def build_student(params, trunk_h5):
    m = SeqNN(params["model"])
    before = [w.numpy().copy() for w in m.model_trunk.weights]

    m.model_trunk.load_weights(trunk_h5, by_name=True)

    after = [w.numpy() for w in m.model_trunk.weights]
    changed = any(np.any(b != a) for b, a in zip(before, after))
    if not changed:
        raise RuntimeError(f"No trunk weights loaded from {trunk_h5} (name mismatch/path?)")

    trunk_out = m.model_trunk.output
    y = tf.keras.layers.Dense(1, name="distill_head")(trunk_out)
    y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
    return tf.keras.Model(inputs=m.model.input, outputs=y)

def main():
    args = parse_args()
    chrom = args.chrom

    EPOCHS = args.epochs

    # ---- Load NPZ ----
    npz_path = NPZ_TEMPLATE.format(chrom=chrom)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing NPZ file: {npz_path}")

    print(f"Loading ensemble NPZ from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Expect keys produced by the generator: train_* and val_*
    train_sequences  = data["train_sequences"].tolist()
    train_mean_preds = data["train_mean_preds"].astype(np.float32)  # (N_train, 896)


    print("Train sequences shape: ", np.array(train_sequences).shape)
    print("Train preds shape:    ", train_mean_preds.shape)

    # ---- Build datasets ----
    train_ds = make_ds(train_sequences, train_mean_preds, batch_size=BATCH_SIZE)

    # ---- Load model params ----
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    # ---- Build Shorkie trunk and load pretrained weights ----
    print("\nBuilding Shorkie trunk via SeqNN...")
    student = build_student(params, TRUNK_H5_TEMPLATE)

    student.summary(line_length=120)

    # ---- Compile ----
    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    student.compile(
        optimizer=opt,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            pearson_r,
            spearman_r,
        ],
    )

    # ---- Training
    out_path = OUT_WEIGHTS_TEMPLATE.format(chrom=chrom)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    history = student.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    student.save_weights(out_path)
    print(f"\nTraining finished. Final student weights saved to: {out_path}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()