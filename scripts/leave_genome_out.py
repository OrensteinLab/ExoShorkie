#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random, numpy as np
import tensorflow as tf
import gzip, json
from Bio import SeqIO
from Bio.Seq import Seq
from dataclasses import dataclass
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from tensorflow.keras import mixed_precision
from baskerville.seqnn import SeqNN
from src.data_loader import *
from pathlib import Path
# --------------------------------------
# Determinism
# --------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
mixed_precision.set_global_policy("mixed_bfloat16")

# --------------------------------------
# Constants
# --------------------------------------
WINDOW_BP   = 16384
CROP_BP     = 1024
BIN_SIZE_BP = 16
BATCH_SIZE  = 512
STRIDE_BP   = 1024
PARAMS_JSON = "Shorkie_params.json"
# Note: We still load models trained on CV folds to get the ensemble effect, 
# or you can restrict this to specific folds if desired. 
# Current logic: Ensembles all available CV models for the training genomes.
FINETUNE_H5_TEMPLATE = "Models/{chrom}/cv{cv}/f{fold}/model_finetune.h5"


# ================================================================
#   SOURCE STRUCTURE
# ================================================================
@dataclass
class Source:
    name: str
    npz_fwd: str
    npz_rev: str
    chrom: str
    fa_path: str
    cov_fwd: np.ndarray = None
    cov_rev: np.ndarray = None
    seq: str = None
    X_fwd: np.ndarray = None
    X_rc: np.ndarray = None


# ================================================================
#   LOAD SOURCES
# ================================================================
def load_and_attach(sources):
    for s in sources:
        s.cov_fwd = fetch_coverage(s.npz_fwd, s.chrom)
        s.cov_rev = fetch_coverage(s.npz_rev, s.chrom)
        seq = fetch_chr_seq(s.fa_path, s.chrom)
        if len(seq) != len(s.cov_fwd):
            raise ValueError(f"Seq vs cov mismatch in {s.name} length sequence: {len(seq)} vs length coverage: {len(s.cov_fwd)}")
        s.seq = seq
        s.X_fwd = build_shorkie_features(seq)
        rc_seq = str(Seq(seq).reverse_complement())
        s.X_rc  = build_shorkie_features(rc_seq)
    return sources


# ================================================================
#   WINDOW BUILDING
# ================================================================
def build_windows_per_source(sources, win_bp, stride=1024):
    windows_per_source = []
    for s in sources:
        L = len(s.seq)
        # make_windows typically returns list of (source_idx, win_idx, start, end)
        wins = make_windows(0, L, win_bp, stride)
        windows_per_source.append(wins)
    return windows_per_source


# ================================================================
#   RUN ALL MODELS OVER ALL TEST SETS
# ================================================================
def run_models_over_test_sets(test_sets, chrom_list, genome_names):
    """
    For each (chrom, cv, fold) model:
      - load model once
      - predict on all test_sets where chrom is in train_chroms
      - accumulate sum_pred and count
    """
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    # We iterate over all potential training models.
    # Even though we aren't using folds for testing, we use the models 
    # trained during the CV process (ensemble).
    for chrom, genome in zip(chrom_list,genome_names):
        for cv in range(5):
            for fold in range(8):
                model_path = FINETUNE_H5_TEMPLATE.format(chrom=genome, cv=cv, fold=fold)

                if not os.path.exists(model_path):
                    # It is possible not all folds exist or were run; skip if missing
                    continue

                print(f"\n[MODEL] chrom={chrom} cv={cv} fold={fold}")
                
                # Rebuild model
                m = SeqNN(params["model"])
                trunk_out = m.model_trunk.output
                y = tf.keras.layers.Dense(1, name=f"per_bin_f{fold}")(trunk_out)
                y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
                ft = tf.keras.Model(m.model.input, y)

                ft.load_weights(model_path)

                # Predict on all relevant test sets
                for ts in test_sets:
                    if ts["X"] is None:
                        continue
                    # Only predict if this model's chrom was NOT the test genome
                    if chrom not in ts["train_chroms"]:
                        continue

                    X = ts["X"]
                    p = ft.predict(X, batch_size=BATCH_SIZE, verbose=1) 

                    if ts["sum_pred"] is None:
                        ts["sum_pred"] = p.astype(np.float32)
                        ts["count"] = 1
                    else:
                        ts["sum_pred"] += p
                        ts["count"] += 1

                del ft, m
                tf.keras.backend.clear_session()


# ================================================================
#   BUILD ALL TEST SETS (ALL WINDOWS PER GENOME)
# ================================================================
def build_all_test_sets(all_sources):
    """
    Precompute inputs X, labels Y, and start indices for every genome 
    (using ALL windows, no splits).
    """
    chrom_list = []
    genome_names = []
    for genome_list in all_sources:
        chrom = genome_list[0].chrom
        if chrom not in chrom_list:
            chrom_list.append(chrom)
            genome_names.append(genome_list[0].name)

    test_sets = []

    for test_idx, test_genome in enumerate(all_sources):
        test_name = test_genome[0].name
        test_chrom = test_genome[0].chrom
        print(f"\n=== Preparing test set for genome = {test_name} (chrom={test_chrom}) ===")

        # Identify training chroms (everyone except current test genome)
        train_genomes = [g for i, g in enumerate(all_sources) if i != test_idx]
        train_chroms = [g[0].chrom for g in train_genomes]

        # 1. Get all windows for this genome
        wins_per_source = build_windows_per_source(test_genome, WINDOW_BP, stride=STRIDE_BP)
  
        X_list = []
        Y_list = []
        starts_list = []

        # 2. Flatten windows and build arrays
        # wins_per_source is a list of lists (if multi-chrom genome) or list of list (single)
        for si, source_wins in enumerate(wins_per_source):
            s = test_genome[si]
            L = len(s.seq)
            
            for win in source_wins:
                a, b = win
                # Forward
                X_list.append(s.X_fwd[a:b])
                Y_list.append(crop_and_bin_cov(s.cov_fwd[a:b], CROP_BP, BIN_SIZE_BP))
                starts_list.append(a)

                # Reverse Complement
                X_list.append(s.X_rc[L-b:L-a])
                Y_list.append(crop_and_bin_cov(s.cov_rev[a:b][::-1], CROP_BP, BIN_SIZE_BP))
                starts_list.append(a) # Same genomic start index for RC

        if len(X_list) == 0:
            print(f"Warning: No windows found for {test_name}")
            continue

        X = np.stack(X_list, axis=0) 
        Y = np.stack(Y_list, axis=0) 
        starts = np.array(starts_list, dtype=int)

        print(f"  Total windows: {len(starts_list)} (Fwd+Rev)")

        test_sets.append({
            "genome_name": test_name,
            "chrom": test_chrom,
            "train_chroms": train_chroms,
            "X": X,
            "Y": Y,
            "starts": starts, # Save indices
            "sum_pred": None,
            "count": 0,
        })

    return test_sets, chrom_list, genome_names


# ================================================================
#   LEAVE ONE GENOME OUT EVALUATION
# ================================================================
def evaluate_loocv(all_sources):

    all_results = []

    # preload everything
    for genome_list in all_sources:
        load_and_attach(genome_list)

    # 1) Build test sets (All windows)
    test_sets, chrom_list, genome_names = build_all_test_sets(all_sources)

    # 2) Run all models
    run_models_over_test_sets(test_sets, chrom_list, genome_names)

    # 3) Compute stats and save NPZ
    for ts in test_sets:
        name = ts["genome_name"]
        X = ts["X"]
        Y = ts["Y"]
        sum_pred = ts["sum_pred"]
        count = ts["count"]
        starts = ts["starts"]

        if count == 0:
            print(f"Skipping {name}, no predictions made.")
            continue

        Y_pred = sum_pred / float(count)

        print("Y pred shape:", Y_pred.shape)

        n_windows = Y.shape[0]

        # Calculate Spearman per window
        per_window_corrs = []
        for i in range(n_windows):
            # Check for constant arrays to avoid NaNs if necessary, 
            # though spearmanr usually handles it by returning nan
            c = spearmanr(Y_pred[i], Y[i]).correlation
            per_window_corrs.append(c)

        per_window_corrs = np.array(per_window_corrs, dtype=float)
        
        # Save NPZ: Starts and Correlations
        Path("Results/Correlations").mkdir(parents=True, exist_ok=True)
        npz_filename = Path("Results/Correlations") / f"correlations_leave_genome_out_{name}.npz"
        np.savez(npz_filename, starts=starts, correlations=per_window_corrs)

        # Summary Metric
        median_corr = np.nanmedian(per_window_corrs)
        print(f"[RESULT] genome={name} n_windows={n_windows} spearman_median={median_corr}")

        all_results.append({
            "genome": name,
            "n_windows": n_windows,
            "models_used": count,
            "spearman_median": median_corr,
        })

    # Global CSV
    df_all = pd.DataFrame(all_results)

    results_dir = Path("Results")

    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"leave_genome_out_results.csv"
    df_all.to_csv(out_csv, index=False)

    print(f"\nSaved global results to {out_csv}")


# ================================================================
#   MAIN
# ================================================================
def main():
    # Define sources as before
    human_sources = [
        Source("Human_chr_7",
               "Data/Human chr 7/Human_part1_fwd_norm.npz",
               "Data/Human chr 7/Human_part1_rev_norm.npz",
               "chr7", "Data/Human chr 7/chr7_part1.fa"),
        Source("Human_chr_7",
               "Data/Human chr 7/Human_part2_fwd_norm.npz",
               "Data/Human chr 7/Human_part2_rev_norm.npz",
               "chr7", "Data/Human chr 7/chr7_part2.fa")
    ]

    mpneumo = [Source("M_pneumoniae",
                      "Data/M. pneumoniae/Mpneumo_fwd_norm.npz",
                      "Data/M. pneumoniae/Mpneumo_rev_norm.npz",
                      "Mpneumo", "Data/M. pneumoniae/Mpneumo.fa")]

    mmyco = [Source("M_mycoides",
                    "Data/M. mycoides/Mmmyco_fwd_norm.npz",
                    "Data/M. mycoides/Mmmyco_rev_norm.npz",
                    "Mmmyco", "Data/M. mycoides/Mmmyco.fa")]

    hprt1 = [Source("HPRT1",
                    "Data/HPRT1/HPRT1_fwd_norm.npz",
                    "Data/HPRT1/HPRT1_rev_norm.npz",
                    "HPRT1", "Data/HPRT1/HPRT1.fa")]

    hprt1r = [Source("HPRT1R",
                     "Data/HPRT1R/HPRT1R_fwd_norm.npz",
                     "Data/HPRT1R/HPRT1R_rev_norm.npz",
                     "HPRT1R", "Data/HPRT1R/HPRT1R.fa")]

    dchr = [Source("Data_storage_chr",
                   "Data/Data-storage chr/dChr_fwd_norm.npz",
                   "Data/Data-storage chr/dChr_rev_norm.npz",
                   "dChr", "Data/Data-storage chr/dChr.fa")]

    all_sources = [human_sources, mpneumo, mmyco, hprt1, hprt1r, dchr]

    evaluate_loocv(all_sources)


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()