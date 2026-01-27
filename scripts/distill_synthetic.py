#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------- Reproducibility (must be before importing TF) ----------
import random, numpy as np
import argparse
import json
import gzip # Added missing import
from typing import List, Tuple
from Bio.Seq import Seq
from dataclasses import dataclass # Needed for Source definition if not imported

# Set SEED globally before any other imports
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
mixed_precision.set_global_policy("float32")
from baskerville.seqnn import SeqNN

# --- IMPORT NECESSARY COMPONENTS FROM data_loader.py ---
# Note: Augmentation functions are deliberately kept local.
from src.data_loader import *

@dataclass
class Source:
    name: str
    chrom: str
    fa_path: str
    cov_fwd: np.ndarray = None
    cov_rev: np.ndarray = None
    seq: str = None
    start0: int = 0
    end0: int = 0
    X_fwd: np.ndarray = None
    X_rc: np.ndarray = None

# --- CONFIGURATION ---
WINDOW_BP       = 16384
PRED_BATCH_SIZE = 512
PARAMS_JSON     = "Models/shorkie/params.json"
BASES = ["A", "C", "G", "T"]

N_CV_FOLDS = 5          # Standard CV folds for Finetuned mode

# --- MODEL PATH TEMPLATES ---
# Finetuned template includes CV folder
FINETUNE_H5_TEMPLATE = "Models/{chrom}/cv{cv}/f{fold}/model_finetune.h5"
# Genomic template is simpler (no CV)
GENOMIC_H5_TEMPLATE = "Models/NatShorkie/f{fold}/model_finetune.h5"

# --- GLOBAL GENOMIC DATA (Used for Genomic Mode) ---
GENOMIC_CHROMS = [
    "chrI", "chrII", "chrIII", "chrIV", "chrV",
    "chrVI", "chrVII", "chrVIII", "chrIX", "chrX",
    "chrXI", "chrXII", "chrXIII", "chrXIV", "chrXV"
]


# ---------- ALPHA GENOME AUGMENTATION (LOCAL DEFINITION) ----------
# (Augmentation functions defined locally as requested)

def random_base(exclude: str = None) -> str:
    """Return a random base, optionally excluding one."""
    if exclude is None:
        return random.choice(BASES)
    b = random.choice(BASES)
    while b == exclude:
        b = random.choice(BASES)
    return b


def mutate_bases(seq: str, frac: float = 0.04) -> str:
    """Mutate ~4% of positions by substituting with a *different* random base."""
    L = len(seq)
    if L == 0: return seq
    n_mut = max(1, int(round(frac * L)))
    positions = np.random.choice(L, size=n_mut, replace=False)
    seq_list = list(seq)
    for pos in positions:
        seq_list[pos] = random_base(exclude=seq_list[pos])
    return "".join(seq_list)


def apply_structural_variants(seq: str, lam: float = 1.0, max_len: int = 20) -> str:
    """Apply a Poisson(lam) number of structural variants (ins/del/inv)."""
    k = np.random.poisson(lam)
    for _ in range(k):
        if len(seq) <= 1: break
        vtype = random.choice(["ins", "del", "inv"])
        L = random.randint(1, max_len)

        if vtype == "ins":
            pos = random.randint(0, len(seq))
            insert = "".join(random_base() for _ in range(L))
            seq = seq[:pos] + insert + seq[pos:]
        elif vtype == "del":
            if len(seq) <= L: continue
            pos = random.randint(0, len(seq) - L)
            seq = seq[:pos] + seq[pos + L:]
        elif vtype == "inv":
            if len(seq) <= L: continue
            pos = random.randint(0, len(seq) - L)
            subseq = seq[pos:pos + L]
            inv = str(Seq(subseq).reverse_complement())
            seq = seq[:pos] + inv + seq[pos + L:]
    return seq


def normalize_length(seq: str, target_len: int) -> str:
    """Normalize sequence length to `target_len` by cropping or padding with Ns."""
    L = len(seq)
    if L == target_len: return seq
    if L > target_len:
        start = random.randint(0, L - target_len)
        return seq[start:start + target_len]
    pad = target_len - L
    left = pad // 2
    right = pad - left
    return "N" * left + seq + "N" * right


def alphagenome_augment_window(seq_win: str) -> str:
    """Apply AlphaGenome-style augmentations to a single window."""
    if random.random() < 0.5:
        seq_win = str(Seq(seq_win).reverse_complement())

    seq_win = mutate_bases(seq_win, frac=0.04)
    seq_win = apply_structural_variants(seq_win, lam=1.0, max_len=20)
    seq_win = normalize_length(seq_win, WINDOW_BP)

    return seq_win




# ---------- Generation Logic (Uses LOCAL Augmentation) ----------

def generate_synthetic_sequences(sources: List[Source],
                                 window_pairs: List[Tuple[int, int, int, int]],
                                 n_synth: int) -> Tuple[List[str], np.ndarray]:
    """Generates n_synth synthetic sequences by sampling from window_pairs."""
    n_base = len(window_pairs)
    if n_base == 0:
        raise RuntimeError("No base windows found to generate synthetic sequences from.")

    print(f"Total base windows available for sampling: {n_base}")

    synth_seqs = []
    base_indices = []

    for i in range(n_synth):
        si, wi, a, b = window_pairs[np.random.randint(n_base)]
        s = sources[si]
        seq_win = s.seq[a:b]
        
        syn = alphagenome_augment_window(seq_win) 
        
        synth_seqs.append(syn)
        base_indices.append((si, wi, a, b))

        if (i + 1) % 5000 == 0:
            print(f"Generated {i+1} / {n_synth} synthetic sequences")

    return synth_seqs, np.array(base_indices, dtype=np.int64)



# ---------- Ensemble Prediction Helper (MODIFIED FOR CV) ----------
def ensemble_predict_on_sequences(
    synth_sequences: List[str],
    model_template: str,
    chrom_for_path: str,
    n_ensemble: int,
    is_genomic_mode: bool
) -> np.ndarray:
    """
    Run the ensemble prediction, accounting for N_CV_FOLDS if not in genomic mode.
    """
    
    # Helper functions for dataset creation
    def cast_for_model_x(x_uint8):
        return tf.cast(x_uint8, tf.float32)
        
    def make_synth_ds(sequences: List[str], batch_size: int = PRED_BATCH_SIZE):
        def gen():
            for seq in sequences:
                x = build_shorkie_features(seq) 
                yield x
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=tf.TensorSpec(shape=(WINDOW_BP, 170), dtype=tf.uint8),
        )
        ds = ds.map(cast_for_model_x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    sum_preds = None
    model_counter = 0

    ds = make_synth_ds(synth_sequences)
    
    # Define iteration loops based on mode
    cv_folds = range(1) if is_genomic_mode else range(N_CV_FOLDS)
    
    n_total_models = n_ensemble * len(cv_folds)

    for cv_idx in cv_folds:
        for fold in range(n_ensemble):
            
            # 1. Determine model path based on mode
            if is_genomic_mode:
                 # Genomic mode uses GENOMIC_H5_TEMPLATE (no {cv})
                model_path = model_template.format(fold=fold)
            else:
                # Finetuned mode uses FINETUNE_H5_TEMPLATE (with {cv})
                model_path = model_template.format(
                    chrom=chrom_for_path, cv=cv_idx, fold=fold
                )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model for CV{cv_idx}, f{fold}: {model_path}")

            print(f"\n=== Predicting with CV{cv_idx}, f{fold} from {model_path} ===")

            m = SeqNN(params["model"])
            trunk_out = m.model_trunk.output

            # Name the Dense and Lambda layers explicitly
            dense_layer_name = f"per_bin_f{fold}"
            
            y = tf.keras.layers.Dense(1, name=dense_layer_name)(trunk_out)
            y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
            ft = tf.keras.Model(m.model.input, y)

            # ----- weight checking logic -----
            trunk_before = [w.numpy().copy() for w in m.model_trunk.weights]
            head_layer = ft.get_layer(dense_layer_name)
            head_before = [w.numpy().copy() for w in head_layer.weights]

            ft.load_weights(model_path, by_name=True)

            trunk_after = [w.numpy() for w in m.model_trunk.weights]
            head_after = [w.numpy() for w in head_layer.weights]

            trunk_changed = any(np.any(b != a) for b, a in zip(trunk_before, trunk_after))
            head_changed = any(np.any(b != a) for b, a in zip(head_before, head_after))

            if not trunk_changed:
                raise RuntimeError("Trunk weights did not change – incorrect model file or mismatch.")
            if not head_changed:
                raise RuntimeError("Head weights did not change – incorrect model file or mismatch.")
            # ------------------------------------------------------------------

            preds = ft.predict(ds, verbose=1)
            preds = preds.astype(np.float32)

            if sum_preds is None:
                sum_preds = preds
            else:
                sum_preds += preds

            model_counter += 1
            print(f"Accumulated predictions from {model_counter} / {n_total_models} models")
            tf.keras.backend.clear_session()

    mean_preds = sum_preds / float(n_total_models) # CRITICAL: Divide by total models run
    print(f"\nFinal mean_preds shape: {mean_preds.shape}")
    return mean_preds


# ---------- CLI (Unified) ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AlphaGenome-style synthetic sequences and "
                    "predict with a Shorkie ensemble (90/10 split)."
    )
    
    parser.add_argument("--genomic", action="store_true",
                             help="Use the Genomic Zero-Shot model template. Sources are overwritten to be all yeast chromosomes.")

    # Arguments are now optional for Genomic Mode
    parser.add_argument("--src1-chrom", type=str, help="Chromosome/source name for output file naming.")
    parser.add_argument("--src1-fasta", type=str, help="Path to FASTA file for Source 1.")

    parser.add_argument("--src2-chrom", type=str, help="Chromosome/source name for Source 2 (Optional).")
    parser.add_argument("--src2-fasta", type=str, help="Path to FASTA file for Source 2 (Optional).")

    parser.add_argument("--n-synth", type=int, default=50_000,
                        help="Total number of synthetic sequences (train). Default 50000.")
    
    parser.add_argument("--target-windows", type=int, default=10_000,
                        help="Target total genomic windows to sample from. Default 10000.")
    
    parser.add_argument("--ensemble", type=int, default=8)

    return parser.parse_args()


# ---------- Main (Unified and Final) ----------

def main():
    tf.get_logger().setLevel("ERROR")
    args = parse_args()

    # --- 1. Set Model Parameters and Define Sources ---
    is_genomic = args.genomic

    if is_genomic:
        MODEL_TEMPLATE = GENOMIC_H5_TEMPLATE
        print("--- Mode: GENOMIC (Zero-Shot) ---")
        
        #  Ensure FASTA is provided for Genomic Mode
        if not args.src1_fasta:
             raise ValueError("Genomic Mode requires --src1-fasta path to the yeast genome file.")
        
        chrom_for_path = "genomic"
        print(f"Loading ALL {len(GENOMIC_CHROMS)} Yeast chromosomes as sources.")
        
        FASTA_PATH = args.src1_fasta

        # Override sources to be all yeast chromosomes, using the user-provided FASTA path
        sources = [
            Source(
                name=chrom,
                chrom=chrom,
                fa_path=FASTA_PATH, 
            ) for chrom in GENOMIC_CHROMS
        ]
        
    else: # Finetuned Mode 
        MODEL_TEMPLATE = FINETUNE_H5_TEMPLATE
        print("--- Mode: FINETUNED (Specific Chromosome) ---")
        
        if not all([args.src1_chrom, args.src1_fasta]):
            raise ValueError("Finetuned Mode requires both --src1-chrom and --src1-fasta.")

        # Standard Source Loading (Uses user inputs)
        sources = [
            Source(
                name=args.src1_chrom,
                chrom=args.src1_chrom,
                fa_path=args.src1_fasta,
            )
        ]
        if args.src2_chrom and args.src2_fasta:
            sources.append(
                Source(
                    name=args.src2_chrom,
                    chrom=args.src2_chrom,
                    fa_path=args.src2_fasta,
                )
            )
        chrom_for_path = sources[0].chrom


    # --- 2. Load Sequences, Generate Windows, and Split ---
    print("Loading sequences from FASTA…")
    attach_sequences_for_sources(sources)

    print(f"\nWindow spec: win={WINDOW_BP}")
    windows_per_source, _ = build_windows_per_source(
        sources, WINDOW_BP, target_wins=args.target_windows, train_ratio=1
    )

    all_windows = []

    for si, s in enumerate(windows_per_source):
        wins = windows_per_source[si]
        for wi, (start_bp, end_bp) in enumerate(wins):  
           all_windows.append((si, wi, start_bp, end_bp))
            
    print("Number of windows for distillation: ", len(all_windows))
    
    # --- 3. Generate Synthetic Sequences ---
    total_synth = args.n_synth

    print("\nGenerating TRAIN synthetic sequences (from training genomic windows)…")
    train_synth_sequences, train_base_indices = generate_synthetic_sequences(
        sources, all_windows, n_synth=total_synth 
    )

    # --- 4. Ensemble Prediction ---
    print("\nRunning ensemble predictions on TRAIN synthetic sequences…")
    train_mean_preds = ensemble_predict_on_sequences(
        train_synth_sequences, MODEL_TEMPLATE, chrom_for_path, args.ensemble, is_genomic
    )

    # --- 5. Save Results ---
    os.makedirs(f"Results/Distillation/{chrom_for_path}", exist_ok=True)
    out_path = f"Results/Distillation/{chrom_for_path}/synthetic_{chrom_for_path}_mean_preds.npz"

    np.savez_compressed(
        out_path,
        train_sequences    = np.array(train_synth_sequences, dtype=object),
        train_mean_preds   = train_mean_preds.astype(np.float32),
        train_base_indices = train_base_indices,
    )

    print(f"\nSaved student train & synthetic sequences set to {out_path}")


if __name__ == "__main__":
    main()