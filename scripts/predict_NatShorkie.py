#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from tensorflow.keras import mixed_precision
from baskerville.seqnn import SeqNN
from pathlib import Path
from src.data_loader import *
import copy

# --- Default hyperparameters (used in the paper) ---
SEED = 42
WINDOW_BP     = 16384
CROP_BP       = 1024
BIN_SIZE_BP   = 16
STRIDE_BP     = 1024
PRED_BATCH_SIZE = 256 
PARAMS_JSON   = "Shorkie_params.json"


def _layer_weights_snapshot(layer):
    return [w.numpy().copy() for w in layer.weights]


def _weights_changed(before, after):
    return any(np.any(b != a) for b, a in zip(before, after))


def load_natshorkie_ensemble_member(model_params: dict, fold: int, model_path: str) -> tf.keras.Model:
    """Build SeqNN + per-fold head, load weights, verify first/last trunk and head layers updated."""
    m = SeqNN(copy.deepcopy(model_params))
    head_name = f"per_bin_f{fold}"
    y = tf.keras.layers.Dense(1, name=head_name)(m.model_trunk.output)
    y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
    ft = tf.keras.Model(m.model.input, y)

    head_layer = ft.get_layer(head_name)
    trunk_layers = [l for l in m.model_trunk.layers if l.weights]
    if not trunk_layers:
        raise RuntimeError("Trunk has no layers with weights.")
    first_trunk, last_trunk = trunk_layers[0], trunk_layers[-1]

    w_first_before = _layer_weights_snapshot(first_trunk)
    w_last_before = _layer_weights_snapshot(last_trunk)
    w_head_before = _layer_weights_snapshot(head_layer)

    ft.load_weights(model_path, by_name=True)

    first_changed = _weights_changed(w_first_before, _layer_weights_snapshot(first_trunk))
    last_changed = _weights_changed(w_last_before, _layer_weights_snapshot(last_trunk))
    head_changed = _weights_changed(w_head_before, _layer_weights_snapshot(head_layer))

    print(f"    First trunk layer ({first_trunk.name}) changed? {first_changed}")
    print(f"    Last trunk layer ({last_trunk.name}) changed? {last_changed}")
    print(f"    Head ({head_name}) changed? {head_changed}")
    if not (first_changed and last_changed and head_changed):
        raise ValueError(
            f"Expected all checked layers to update after load_weights({model_path!r}). "
            f"got first={first_changed}, last={last_changed}, head={head_changed}"
        )

    return ft


def setup_env():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    mixed_precision.set_global_policy("mixed_bfloat16")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 5-fold CV Ensemble")

    parser.add_argument("--ensemble", type=int, default=8)

    # --- Sources ---
    parser.add_argument("--src1-name", type=str, default="src1")
    parser.add_argument("--src1-chrom", type=str, required=True)
    parser.add_argument("--src1-npz-fwd", type=str, required=True)
    parser.add_argument("--src1-npz-rev", type=str, required=True)
    parser.add_argument("--src1-fasta", type=str, required=True)
    
    parser.add_argument("--src2-name", type=str, default="src2")
    parser.add_argument("--src2-chrom", type=str)
    parser.add_argument("--src2-npz-fwd", type=str)
    parser.add_argument("--src2-npz-rev", type=str)
    parser.add_argument("--src2-fasta", type=str)
    
    return parser.parse_args()

def main():
    setup_env()
    args = parse_args()

    N_ENSEMBLE = args.ensemble

    MODEL_TEMPLATE = "Models/NatShorkie/f{fold}/model_finetune.h5"

    # 1. Load Sources
    sources = [
        Source(name=args.src1_name, npz_fwd=args.src1_npz_fwd, npz_rev=args.src1_npz_rev,
               chrom=args.src1_chrom, fa_path=args.src1_fasta)
    ]
    if args.src2_chrom and args.src2_npz_fwd:
        sources.append(
            Source(name=args.src2_name, npz_fwd=args.src2_npz_fwd, npz_rev=args.src2_npz_rev,
                   chrom=args.src2_chrom, fa_path=args.src2_fasta)
        )

    print("Loading data...")
    load_coverages_and_seqs(sources)
    print("Precomputing features...")
    precompute_full_features(sources)

    # 2. Windows & Splits
    print(f"Generating windows (Stride={STRIDE_BP})...")

    test_windows = []
    for si, s in enumerate(sources):
        L = len(s.seq)
            
        wins = make_windows(0, L, WINDOW_BP, STRIDE_BP)
        
        # Enumerate the windows to get 'wi' and append to the flat list
        for wi, (a, b) in enumerate(wins):
            test_windows.append((si, wi, a, b))

    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    chrom_name = sources[0].chrom

    # A. Prepare Labels    
    y_fwd, y_rc = precompute_window_labels(sources, test_windows, apply_norm=False,
                                               crop_bp=CROP_BP, bin_size_bp=BIN_SIZE_BP)
    
    # B. Make Dataset
    test_ds = make_ds_from_pairs(sources, test_windows, y_fwd, y_rc, 
                                    WINDOW_BP, PRED_BATCH_SIZE, shuffle=False)
    
    Y_test = np.concatenate([y for _, y in test_ds], axis=0)
    
    # D. Ensemble Prediction
    all_fold_preds = []
    for fold in range(N_ENSEMBLE):
        model_path = MODEL_TEMPLATE.format(fold=fold)
        if not os.path.exists(model_path):
            print(f"  [Warning] Missing: {model_path}")
            continue
            
        print(f"  Predicting: {model_path}")

        ft = load_natshorkie_ensemble_member(params["model"], fold, model_path)
        preds = ft.predict(test_ds, verbose=1)
        all_fold_preds.append(preds)

        tf.keras.backend.clear_session()

    # E. Stats
    preds_ens = np.mean(all_fold_preds, axis=0)

    starts = []
    for (si, wi, a, b) in test_windows:
        starts.append(a)
    starts = np.array(starts, dtype=int)

    correlations = []
    for p, y in zip(preds_ens, Y_test):
        correlations.append(spearmanr(p, y)[0])
    
    median_r = np.nanmedian(correlations)
    std_r = np.nanstd(correlations)

    print("Chromosome Results:", chrom_name)
    print(f"Median Spearman: {median_r:.4f}")
    print(f"STD Spearman: {std_r:.4f}")

    Path("Results/Correlations").mkdir(parents=True, exist_ok=True)

    out_npz = Path("Results/Correlations") / f"correlations_NatShorkie_{chrom_name}.npz"

    np.savez(
    out_npz,
    starts=starts,
    correlations=np.array(correlations, dtype=np.float32),
    median_spearman=np.float32(median_r),
    std_spearman=np.float32(std_r),
    )

    print(f"Saved correlations to {out_npz}")




if __name__ == "__main__":
    main()