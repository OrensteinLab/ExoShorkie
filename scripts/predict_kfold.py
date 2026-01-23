#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from tensorflow.keras import mixed_precision
from baskerville.seqnn import SeqNN
from pathlib import Path

# --- IMPORT FROM YOUR LIBRARY ---
from src.data_loader import *
# --- CONFIG ---
SEED = 42
WINDOW_BP     = 16384
CROP_BP       = 1024
BIN_SIZE_BP   = 16
STRIDE_BP     = 1024
PRED_BATCH_SIZE = 256 
PARAMS_JSON   = "Models/shorkie/params.json"

N_FOLDS = 5
N_ENSEMBLE = 8

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
    group.add_argument("--ablation", action="store_true", 
                        help="Evaluate 'Finetuned_models_ablation' (Control).")
    group.add_argument("--genomic", action="store_true", 
                        help="Evaluate 'Yeast_genome' base model (Zero-Shot).")

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

    # --- DYNAMIC PATH CONFIGURATION ---
    if args.genomic:
        print("--- Mode: GENOMIC (Zero-Shot Yeast Model) ---")
        # The Genomic model has NO CV splits. It's just f0..f7.
        # We ignore the {cv} and {chrom} placeholders in the logic below.
        MODEL_TEMPLATE = "Models/finetuned/Yeast_genome/f{fold}/model_finetune.h5"
        result_suffix = "genomic_zeroshot"
        
    elif args.ablation:
        print("--- Mode: ABLATION (Finetuned_models_ablation) ---")
        model_root = "Models/finetuned_ablation"
        MODEL_TEMPLATE = f"{model_root}/{{chrom}}/cv{{cv}}/f{{fold}}/model_finetune.h5"
        result_suffix = "ablation"
        
    else:
        print("--- Mode: STANDARD (Finetuned_models) ---")
        model_root = "Models/finetuned"
        MODEL_TEMPLATE = f"{model_root}/{{chrom}}/cv{{cv}}/f{{fold}}/model_finetune.h5"
        result_suffix = "cross_validation"


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
        for fold in range(N_ENSEMBLE):
            # --- PATH LOGIC ---
            if args.genomic:
                # Genomic mode ignores 'chrom' and 'cv' structure
                model_path = MODEL_TEMPLATE.format(fold=fold)
            else:
                # Standard/Ablation mode uses full path
                model_path = MODEL_TEMPLATE.format(chrom=chrom_name, cv=cv_idx, fold=fold)
            
            if not os.path.exists(model_path):
                print(f"  [Warning] Missing: {model_path}")
                continue
                
            print(f"  Predicting: {model_path}")
            
            # tf.keras.backend.clear_session()
            m = SeqNN(params["model"])
            y = tf.keras.layers.Dense(1, name=f"per_bin_f{fold}")(m.model_trunk.output)
            y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
            ft = tf.keras.Model(m.model.input, y)
            
            ft.load_weights(model_path)
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

    if args.ablation:
        results_dir = Path("Results/ISMB_results/Ablation")
    elif args.genomic:
        results_dir = Path("Results/ISMB_results/Genomic")
    else:
        results_dir = Path("Results/ISMB_results/Cross_validation")

    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"results_{chrom_name}_{result_suffix}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()