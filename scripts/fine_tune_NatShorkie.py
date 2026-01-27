#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import copy
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from baskerville.seqnn import SeqNN


# --- IMPORT EVERYTHING FROM DATALOADER ---
from src.data_loader import *

# --- CONFIG parameters of Paper ---
SEED = 42
WINDOW_BP     = 16384
CROP_BP       = 1024
BIN_SIZE_BP   = 16
LR            = 2e-5
EPOCHS        = 5
BATCH_SIZE    = 32
TARGET_WINS = 10000

PARAMS_JSON = "Shorkie_params.json"
TRUNK_H5_TEMPLATE = "Models/shorkie/f{fold}/model_best.h5"

def setup_env():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    mixed_precision.set_global_policy("mixed_bfloat16")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-fwd", type=str, required=True)
    parser.add_argument("--npz-rev", type=str, required=True)
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--ensemble", type=int, default=8)
    return parser.parse_args()

def main():
    setup_env()
    args = parse_args()

    GENOME_FASTA = args.fasta
    GENOME_FWD = args.npz_fwd
    GENOME_REV = args.npz_rev
    # 1. Path Setup 
    FINETUNE_H5_TEMPLATE = f"Models/NatShorkie/f{{fold}}/model_finetune.h5"

    # 2. Data Loading (Using src/data_loader)
    YEAST_CHROMS = [f"chr{x}" for x in ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI"]]
    sources = [
        Source(name=c, npz_fwd=GENOME_FWD, npz_rev=GENOME_REV, chrom=c, fa_path=GENOME_FASTA)
        for c in YEAST_CHROMS
    ]
    
    print("Loading coverage and sequences...")
    load_coverages_and_seqs(sources)
    print("Precomputing features...")
    precompute_full_features(sources)

    train_pairs, stride = build_windows_full_genome(sources, WINDOW_BP, target_windows=TARGET_WINS)
    
    print(f"Stride used: {stride} bp")
    print("Total training windows:", len(train_pairs))
    
    mu, sigma = compute_logz_stats_multi(sources, train_pairs, CROP_BP, BIN_SIZE_BP)

    y_fwd, y_rc = precompute_window_labels(
    sources,
    train_pairs,
    apply_norm=True,
    mu=mu,
    sigma=sigma,
    crop_bp=CROP_BP,
    bin_size_bp=BIN_SIZE_BP
    )
    print(f"Computed log-z score stats: mu={mu:.4f}, sigma={sigma:.4f}")

    # 4. Model Setup
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    FOLDS = args.ensemble

    # 5. Training Loop
    for fold in range(FOLDS):
        tf.keras.backend.clear_session()
        fold_seed = SEED + fold
        print(f"\n=== Fold {fold} (Seed {fold_seed}) ===")
        
        # Reset RNG
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        # Create Dataset
        train_ds = make_ds_from_pairs(
            sources, train_pairs, y_fwd, y_rc,
            WINDOW_BP, BATCH_SIZE, shuffle=True, seed=fold_seed
        )

        # Build Model (using deepcopy to be safe)
        m = SeqNN(copy.deepcopy(params["model"]))

        # Add Head
        y = tf.keras.layers.Dense(1, name=f"per_bin_f{fold}")(m.model_trunk.output)
        y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
        ft = tf.keras.Model(m.model.input, y)

        # Weight Loading Logic
        trunk_path = TRUNK_H5_TEMPLATE.format(fold=fold)
        print(f"Loading weights: {trunk_path}")
        # ----- collect BEFORE weights -----
        trunk_before = [w.numpy().copy() for w in m.model_trunk.weights]

        # ----- load weights -----
        # (Logic preserved exactly as requested)
        ft.load_weights(trunk_path, by_name=True)

        # ----- collect AFTER weights -----
        trunk_after = [w.numpy() for w in m.model_trunk.weights]

        # ----- check trunk -----
        trunk_changed = any(np.any(b != a) for b, a in zip(trunk_before, trunk_after))

        print(f"Trunk weights changed? {trunk_changed}")

        if not trunk_changed:
            raise RuntimeError("Trunk weights did not change – incorrect model file or mismatch.")

        # Train & Save
        ft.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
        ft.fit(train_ds, epochs=EPOCHS, verbose=1)

        out_path = FINETUNE_H5_TEMPLATE.format(fold=fold)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ft.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()