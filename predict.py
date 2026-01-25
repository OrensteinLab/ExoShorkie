#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from baskerville.seqnn import SeqNN

from Bio.Seq import Seq
from Bio import SeqIO

# Use your existing feature builder
from src.data_loader import build_shorkie_features

SEED = 42
WINDOW_BP   = 16384
CROP_BP     = 1024
PRED_BP     = WINDOW_BP - 2 * CROP_BP   # 14336
STRIDE_BP   = PRED_BP                   # 14336
BIN_SIZE_BP = 16

PARAMS_JSON = "Shorki_params.json"
MODEL_TEMPLATE = "Models/{chrom}/cv{cv}/f{f}/model_finetune.h5"


def setup_env():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    mixed_precision.set_global_policy("mixed_bfloat16")


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict per-position coverage for a FASTA using Shorkie features and stride=14336."
    )
    p.add_argument("--chrom", required=True, type=str)
    p.add_argument("--cv", required=True, type=int)
    p.add_argument("--fold", required=True, type=int)

    p.add_argument("--fasta", required=True, type=str)

    p.add_argument("--batch", default=64, type=int)
    p.add_argument("--rc", action="store_true",
                   help="Predict on reverse complement sequence")
    p.add_argument("--out", required=True, type=str)
    

    return p.parse_args()


def read_fasta_one_record(fasta_path: str) -> str:
    # If there are multiple records, concatenate (or you can change to take first).
    seqs = []
    with open(fasta_path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            seqs.append(str(rec.seq).upper())
    if not seqs:
        raise ValueError(f"No records found in FASTA: {fasta_path}")
    return "".join(seqs)


def build_inference_model(params_json: str):
    with open(params_json) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    base = SeqNN(params["model"])
    y = tf.keras.layers.Dense(1, name="per_bin")(base.model_trunk.output)
    y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
    model = tf.keras.Model(base.model.input, y)
    return model


def get_model_path(chrom: str, cv: int, fold: int):
    return MODEL_TEMPLATE.format(chrom=chrom, cv=cv, f=fold)


def pad_to_windows(seq: str):
    # pad so first/last window still yields a valid central predicted region
    pad = "N" * CROP_BP
    return pad + seq + pad


def predict_bins_from_features(model, X_full_uint8: np.ndarray, L_original: int, batch: int):
    """
    X_full_uint8: features for PADDED sequence, shape (L_original+2*CROP_BP, 170), dtype uint8
    Returns pred_bins (ceil(L/16),) filled once using stride=14336.
    """
    total_bins = int(np.ceil(L_original / BIN_SIZE_BP))
    pred_bins = np.full((total_bins,), np.nan, dtype=np.float32)

    starts = list(range(0, L_original, STRIDE_BP))  # original coordinates

    # Build batches of window slices
    for i in range(0, len(starts), batch):
        batch_starts = starts[i:i+batch]
        X_batch = []

        for s in batch_starts:
            a = s
            b = s + WINDOW_BP
            w = X_full_uint8[a:b]
            if w.shape[0] < WINDOW_BP:
                # pad with N features if needed (should mostly happen only at tail)
                pad_len = WINDOW_BP - w.shape[0]
                # "N" features = build_shorkie_features("N"*pad_len) (returns uint8)
                w = np.concatenate([w, build_shorkie_features("N" * pad_len)], axis=0)
            X_batch.append(w)

        X_batch = np.stack(X_batch, axis=0).astype(np.uint8)

        # In training you cast to bfloat16; here we do the same for inference.
        Y = model.predict(tf.cast(X_batch, tf.bfloat16), verbose=1)  # (B, 896)

        for b_idx, start_bp in enumerate(batch_starts):
            y_bins = np.asarray(Y[b_idx], dtype=np.float32)  # length 896
            start_bin = start_bp // BIN_SIZE_BP
            end_bin = start_bin + y_bins.shape[0]

            end_bin_clipped = min(end_bin, total_bins)
            keep = end_bin_clipped - start_bin
            if keep > 0:
                pred_bins[start_bin:end_bin_clipped] = y_bins[:keep]

    return pred_bins


def main():
    setup_env()
    args = parse_args()

    seq = read_fasta_one_record(args.fasta)
    L = len(seq)

    # Build padded seq features ONCE (fast)
    seq_padded = pad_to_windows(seq)
    X_fwd = build_shorkie_features(seq_padded)  # (L+2048,170) uint8

    if args.rc:
        seq_rc = str(Seq(seq).reverse_complement())
        X_rc = build_shorkie_features(pad_to_windows(seq_rc))
    else:
        X_rc = None

    # Load  model
    path = get_model_path(args.chrom, args.cv, args.fold)
    if not os.path.exists(path):
        raise FileNotFoundError("No model weights found. Checked:\n" + path)

    model = build_inference_model(PARAMS_JSON)
    model.load_weights(path)

    if args.rc:
        bins_rc = predict_bins_from_features(model, X_rc, L, args.batch)
        pred_bins = bins_rc[::-1]   # reverse bins back to forward
    else:
        pred_bins = predict_bins_from_features(model, X_fwd, L, args.batch)

    pred_bp = np.repeat(pred_bins, BIN_SIZE_BP)[:L].astype(np.float32)

    # Output
    Path("Results").mkdir(exist_ok=True)
    out = f"Results/{args.out}.npz"

    np.savez_compressed(
        out,
        chrom=args.chrom,
        cv=args.cv,
        fold=args.fold,
        rc=args.rc,
        fasta=args.fasta,
        seq_len=L,
        window_bp=WINDOW_BP,
        crop_bp=CROP_BP,
        pred_bp_region=PRED_BP,
        stride_bp=STRIDE_BP,
        bin_size_bp=BIN_SIZE_BP,
        pred_bins=pred_bins,  # per-16bp bin
        pred_bp=pred_bp,      # per-bp expanded
    )

    print("Saved:", out)
    print("pred_bins:", pred_bins.shape, "pred_bp:", pred_bp.shape)


if __name__ == "__main__":
    main()