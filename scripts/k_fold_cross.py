#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import random
import copy
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from baskerville.seqnn import SeqNN

# Import your custom library
from src.data_loader import *

# --- Configuration ---
SEED = 42
WINDOW_BP = 16384
CROP_BP = 1024
BIN_SIZE_BP = 16
LR = 1e-5
EPOCHS = 3
BATCH_SIZE = 32
TARGET_WINS = 10_000

# Params
PARAMS_JSON = "Models/shorkie/params.json" 

def setup_env():
    """Sets seeds and environment variables for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    mixed_precision.set_global_policy("mixed_bfloat16") 


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Finetune SeqNN with K-Fold CV")
    
    # --- Mode Control ---
    parser.add_argument("--ablation", action="store_true", 
                        help="If set, loads base weights from 'Finetuned_models_ablation'. Else 'Finetuned_models'.")
    
    # --- Source 1 (Required) ---
    parser.add_argument("--src1-name", type=str, default="src1")
    parser.add_argument("--src1-chrom", type=str, required=True)
    parser.add_argument("--src1-npz-fwd", type=str, required=True)
    parser.add_argument("--src1-npz-rev", type=str, required=True)
    parser.add_argument("--src1-fasta", type=str, required=True)

    # --- Source 2 (Optional) ---
    parser.add_argument("--src2-name", type=str, default="src2")
    parser.add_argument("--src2-chrom", type=str, default=None)
    parser.add_argument("--src2-npz-fwd", type=str, default=None)
    parser.add_argument("--src2-npz-rev", type=str, default=None)
    parser.add_argument("--src2-fasta", type=str, default=None)

    return parser.parse_args()


def main():
    setup_env()
    args = parse_args()
    
    # --- DYNAMIC PATH CONFIGURATION ---
    if args.ablation:
        print("--- Mode: ABLATION (Loading from Finetuned_models_ablation) ---")
        model_dir_root = "finetuned_ablation"
    else:
        print("--- Mode: STANDARD (Loading from Finetuned_models) ---")
        model_dir_root = "finetuned"

    # Template for loading the Yeast Genome model (Base)
    BASE_H5_TEMPLATE = f"Models/{model_dir_root}/Yeast_genome/f{{fold}}/model_finetune.h5"

    # Template for saving the new Fine-tuned model
    FINETUNE_H5_TEMPLATE = f"Models/{model_dir_root}/{{chrom}}/cv{{cv}}/f{{fold}}/model_finetune.h5"


    # 1. Build Data Sources
    sources = [
        Source(
            name=args.src1_name,
            chrom=args.src1_chrom,
            npz_fwd=args.src1_npz_fwd,
            npz_rev=args.src1_npz_rev,
            fa_path=args.src1_fasta
        )
    ]

    if args.src2_chrom and args.src2_npz_fwd:
        print(f"Adding second source: {args.src2_name} ({args.src2_chrom})")
        sources.append(
            Source(
                name=args.src2_name,
                chrom=args.src2_chrom,
                npz_fwd=args.src2_npz_fwd,
                npz_rev=args.src2_npz_rev,
                fa_path=args.src2_fasta
            )
        )

    # 2. Preprocessing
    print("Loading coverage and sequences...")
    load_coverages_for_sources(sources)
    attach_sequences_for_sources(sources)
    
    print("Precomputing features...")
    precompute_full_features(sources)

    print("Generating windows...")
    wins_per_source, _ = build_windows_per_source(sources, WINDOW_BP, TARGET_WINS)
    
    # Global Stats
    print("Computing global dataset statistics...")
    global_mu, global_sigma = compute_global_stats(sources, wins_per_source, CROP_BP, BIN_SIZE_BP)

    # 3. Create CV Splits
    cv_splits = make_5fold_splits(sources, wins_per_source, n_folds=5)

    # 4. Load Params
    with open(PARAMS_JSON) as f:
        base_params = json.load(f)
    base_params["model"]["num_features"] = 170 


    # 5. Main Training Loop
    for cv, (train_pairs, test_pairs) in enumerate(cv_splits):
        print(f"\n========== CV Fold {cv} ==========")
        print(f"Train Windows: {len(train_pairs)} | Test Windows: {len(test_pairs)}")
        
        mu, sigma = compute_logz_stats_multi(sources, train_pairs, CROP_BP, BIN_SIZE_BP)

        # Training labels (with normalization)
        y_fwd_dict, y_rc_dict = precompute_window_labels(
            sources,
            train_pairs,
            apply_norm=True,
            mu=mu,
            sigma=sigma,
            crop_bp=CROP_BP,
            bin_size_bp=BIN_SIZE_BP
        )

        for fold in range(8):
            tf.keras.backend.clear_session()
            
            run_seed = SEED + (cv * 100) + fold
            random.seed(run_seed)
            np.random.seed(run_seed)
            tf.random.set_seed(run_seed)
            
            train_ds = make_ds_from_pairs(
                sources, train_pairs, y_fwd_dict, y_rc_dict,
                WINDOW_BP, BATCH_SIZE, 
                shuffle=True, seed=run_seed 
            )

            # Initialize Model (FIX: Added copy.deepcopy)
            m = SeqNN(copy.deepcopy(base_params["model"]))

            # Path for this specific fold
            model_path = BASE_H5_TEMPLATE.format(fold=fold)
            print(f"Loading weights from: {model_path}")
            
            trunk_out = m.model_trunk.output

            # Keep the Dense name for safety (weight loading)
            y = tf.keras.layers.Dense(1, name=f"per_bin_f{fold}")(trunk_out)

            y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
            
            ft = tf.keras.Model(m.model.input, y)

            # ----- collect BEFORE weights -----
            trunk_before = [w.numpy().copy() for w in m.model_trunk.weights]

            head_layer = ft.get_layer(f"per_bin_f{fold}")
            head_before = [w.numpy().copy() for w in head_layer.weights]

            # ----- load weights -----
            # (Logic preserved exactly as requested)
            ft.load_weights(model_path, by_name=True)

            # ----- collect AFTER weights -----
            trunk_after = [w.numpy() for w in m.model_trunk.weights]

            head_after = [w.numpy() for w in head_layer.weights]

            # ----- check trunk -----
            trunk_changed = any(np.any(b != a) for b, a in zip(trunk_before, trunk_after))

            # ----- check head -----
            head_changed = any(np.any(b != a) for b, a in zip(head_before, head_after))

            print(f"Trunk weights changed? {trunk_changed}")
            print(f"Head weights changed?  {head_changed}")

            if not trunk_changed:
                raise RuntimeError("Trunk weights did not change – incorrect model file or mismatch.")

            if not head_changed:
                print("WARNING: head weights did NOT change. Your checkpoint probably doesn't contain the Dense(1) head.")

            opt = tf.keras.optimizers.Adam(learning_rate=LR)

            ft.compile(optimizer=opt, loss='mse')
            # ft.summary()

            # FIX: Variable name was 'full_model', changed to 'ft'
            ft.fit(train_ds, epochs=EPOCHS, verbose=1)

            # Save
            out_path = FINETUNE_H5_TEMPLATE.format(
                chrom=args.src1_chrom,
                cv=cv,
                fold=fold
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # FIX: Variable name was 'full_model', changed to 'ft'
            ft.save(out_path)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()