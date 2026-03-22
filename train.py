#!/usr/bin/env python3
import os
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

# --- Default hyperparameters (used in the paper) ---
SEED = 42
WINDOW_BP = 16384
CROP_BP = 1024
BIN_SIZE_BP = 16
LR = 1e-5
EPOCHS = 3
BATCH_SIZE = 32


# Params
PARAMS_JSON = "Shorkie_params.json"


def _layer_weights_snapshot(layer):
    return [w.numpy().copy() for w in layer.weights]


def _weights_changed(before, after):
    return any(np.any(b != a) for b, a in zip(before, after))


def load_pretrained_natshorkie_for_finetune(model_params: dict, fold: int, weights_path: str) -> tf.keras.Model:
    """Build SeqNN + per-fold head, load NatShorkie weights, verify first/last trunk and head updated."""
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

    ft.load_weights(weights_path)

    first_changed = _weights_changed(w_first_before, _layer_weights_snapshot(first_trunk))
    last_changed = _weights_changed(w_last_before, _layer_weights_snapshot(last_trunk))
    head_changed = _weights_changed(w_head_before, _layer_weights_snapshot(head_layer))

    print(f"    First trunk layer ({first_trunk.name}) changed? {first_changed}")
    print(f"    Last trunk layer ({last_trunk.name}) changed? {last_changed}")
    print(f"    Head ({head_name}) changed? {head_changed}")
    if not (first_changed and last_changed and head_changed):
        raise ValueError(
            f"Expected all checked layers to update after load_weights({weights_path!r}). "
            f"got first={first_changed}, last={last_changed}, head={head_changed}"
        )

    return ft


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
    
    # --- Source ---
    parser.add_argument("--name", type=str, default="src1")
    parser.add_argument("--chrom", type=str, required=True)
    parser.add_argument("--npz-fwd", type=str, required=True)
    parser.add_argument("--npz-rev", type=str, required=True)
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--target-wins", type=int, default=10_000)

    return parser.parse_args()


def main():
    setup_env()
    args = parse_args()

    TARGET_WINS = args.target_wins
    
    # Template for loading the Yeast Genome model (Base)
    BASE_H5_TEMPLATE = f"Models/NatShorkie/f{{fold}}/model_finetune.h5"

    # Template for saving the new Fine-tuned model
    FINETUNE_H5_TEMPLATE = f"Models/{{chrom}}/cv{{cv}}/f{{fold}}/model_finetune.h5"

    # 1. Build Data Sources
    sources = [
        Source(
            name=args.name,
            chrom=args.chrom,
            npz_fwd=args.npz_fwd,
            npz_rev=args.npz_rev,
            fa_path=args.fasta
        )
    ]

    # 2. Preprocessing
    print("Loading coverage and sequences...")
    load_coverages_for_sources(sources)
    attach_sequences_for_sources(sources)
    
    print("Precomputing features...")
    precompute_full_features(sources)

    print("Generating windows...")
    wins_per_source, _ = build_windows_per_source(sources, WINDOW_BP, TARGET_WINS)

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

        for fold in range(args.ensemble):
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

            model_path = BASE_H5_TEMPLATE.format(fold=fold)
            print(f"Loading weights from: {model_path}")
            ft = load_pretrained_natshorkie_for_finetune(base_params["model"], fold, model_path)

            opt = tf.keras.optimizers.Adam(learning_rate=LR)

            ft.compile(optimizer=opt, loss='mse')

            ft.fit(train_ds, epochs=EPOCHS, verbose=2)

            # Save
            out_path = FINETUNE_H5_TEMPLATE.format(
                chrom=args.chrom,
                cv=cv,
                fold=fold
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # FIX: Variable name was 'full_model', changed to 'ft'
            ft.save(out_path)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()