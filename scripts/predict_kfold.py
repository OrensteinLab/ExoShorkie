#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from tensorflow.keras import mixed_precision
from baskerville.seqnn import SeqNN
from pathlib import Path
from src.data_loader import *

# --- Default hyperparameters (used in the paper) ---
SEED = 42
WINDOW_BP     = 16384
CROP_BP       = 1024
BIN_SIZE_BP   = 16
STRIDE_BP     = 1024
PRED_BATCH_SIZE = 256 
PARAMS_JSON   = "Shorkie_params.json"

N_FOLDS = 5


def build_kfold_finetuned_model(model_params: dict, fold: int) -> tf.keras.Model:
    """SeqNN trunk + per-fold dense head + squeeze (matches train.py)."""
    m = SeqNN(copy.deepcopy(model_params))
    y = tf.keras.layers.Dense(1, name=f"per_bin_f{fold}")(m.model_trunk.output)
    y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
    return tf.keras.Model(m.model.input, y)


def load_kfold_finetuned_model(model_params: dict, fold: int, model_path: str) -> tf.keras.Model:
    """Build graph and load `model_finetune.h5` weights."""
    ft = build_kfold_finetuned_model(model_params, fold)
    ft.load_weights(model_path)
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
    
    # --- Mode Control ---
    group = parser.add_mutually_exclusive_group()

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
    parser.add_argument("--ensemble", type=int, default=8)
    
    return parser.parse_args()

def main():
    setup_env()
    args = parse_args()

    # --- DYNAMIC PATH CONFIGURATION ---
    MODEL_TEMPLATE = f"Models/{{chrom}}/cv{{cv}}/f{{fold}}/model_finetune.h5"

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
    windows_per_source, _ = build_test_windows_per_source(sources, WINDOW_BP, stride=STRIDE_BP) 
    cv_test_pairs_list = make_5fold_splits(sources, windows_per_source, n_folds=N_FOLDS)

    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    chrom_name = sources[0].chrom
    cv_results = []

    # 3. CV Loop
    for cv_idx, (_, test_pairs) in enumerate(cv_test_pairs_list):
        print(f"\n=== CV Fold {cv_idx} ===")
        print(f"Test Windows: {len(test_pairs)}")

        if len(test_pairs) == 0: continue

        # A. Prepare Labels
        y_fwd, y_rc = precompute_window_labels(sources, test_pairs, apply_norm=False,
                                               crop_bp=CROP_BP, bin_size_bp=BIN_SIZE_BP)
        
        # B. Make Dataset
        test_ds = make_ds_from_pairs(sources, test_pairs, y_fwd, y_rc, 
                                     WINDOW_BP, PRED_BATCH_SIZE, shuffle=False)
        Y_test = np.concatenate([y for _, y in test_ds], axis=0)
        
        # D. Ensemble Prediction
        all_fold_preds = []
        for fold in range(args.ensemble):
           
            model_path = MODEL_TEMPLATE.format(chrom=chrom_name, cv=cv_idx, fold=fold)
            
            if not os.path.exists(model_path):
                print(f"  [Warning] Missing: {model_path}")
                continue
                
            print(f"  Predicting: {model_path}")
            
            tf.keras.backend.clear_session()
            ft = load_kfold_finetuned_model(params["model"], fold, model_path)
            preds = ft.predict(test_ds, verbose=1)
            all_fold_preds.append(preds)

        if not all_fold_preds:
            continue

        # E. Stats
        preds_ens = np.mean(all_fold_preds, axis=0)
        correlations = []
        for p, y in zip(preds_ens, Y_test):
            correlations.append(spearmanr(p, y)[0])
        
        median_r = np.nanmedian(correlations)
        mean_r = np.nanmean(correlations)
        
        print(f"  Median Spearman: {median_r:.4f}")
        
        cv_results.append({
            "cv_fold": cv_idx,
            "median_spearman": median_r,
            "mean_spearman": mean_r
        })

    # 4. Calculate Overall Statistics
    # Extract the metrics from the fold results
    fold_medians = [r["median_spearman"] for r in cv_results]
    fold_means   = [r["mean_spearman"] for r in cv_results]

    # Calculate Mean and Std across folds
    overall_mean_median = np.mean(fold_medians)
    overall_std_median  = np.std(fold_medians)
    
    overall_mean_mean   = np.mean(fold_means)
    overall_std_mean    = np.std(fold_means)

    print("\n=== Overall Summary ===")
    print(f"Median Spearman: {overall_mean_median:.4f} ± {overall_std_median:.4f}")
    print(f"Mean Spearman:   {overall_mean_mean:.4f} ± {overall_std_mean:.4f}")

    # Append Summary Rows to the results
    cv_results.append({
        "cv_fold": "Overall_Mean",
        "median_spearman": overall_mean_median,
        "mean_spearman": overall_mean_mean
    })
    
    cv_results.append({
        "cv_fold": "Overall_Std",
        "median_spearman": overall_std_median,
        "mean_spearman": overall_std_mean
    })

    # 4. Save

    df = pd.DataFrame(cv_results)

    results_dir = Path("Results/Cross_validation")

    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"results_{chrom_name}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()