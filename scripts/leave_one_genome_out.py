#!/usr/bin/env python3

import argparse
import copy
import os
import random
import numpy as np
import tensorflow as tf
import gzip
import json
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

# Ablation: which Models/<name>/ checkpoints participate in the ensemble (must match
# Source.name / Hugging Face layout — not NPZ/FASTA keys in Source.chrom).
ABLATION_HUMAN_MODELS = ["Human_chr_7", "HPRT1"]
ABLATION_BACTERIA_MODELS = ["M_pneumoniae", "M_mycoides"]
ABLATION_SYN_MODELS = ["Data_storage_chr", "HPRT1R"]

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
# Max samples per GPU tensor to avoid OOM on large test sets (e.g. Mmmyco 2356 windows)
INFERENCE_CHUNK_SIZE = 512
STRIDE_BP   = 1024  # default for backward compat
# Strides for correlation reporting (bp)
REPORT_STRIDES = [1024, 14336]
PARAMS_JSON = "Models/shorkie/params.json"
# Note: We still load models trained on CV folds to get the ensemble effect, 
# or you can restrict this to specific folds if desired. 
# Current logic: Ensembles all available CV models for the training genomes.
FINETUNE_H5_TEMPLATE = "Models/{model_name}/cv{cv}/f{fold}/model_finetune.h5"


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
#   MODEL LOADING
# ================================================================
def _layer_weights_snapshot(layer):
    return [w.numpy().copy() for w in layer.weights]


def _weights_changed(before, after):
    return any(np.any(b != a) for b, a in zip(before, after))


def load_model_params():
    """Load and return model params (with num_features=170)."""
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170
    return params


def load_finetuned_model(model_path, params, fold):
    """
    Load a single finetuned model from disk.
    Verifies first/last trunk layers and dense head updated after load_weights.
    Returns a Keras Model (ft). Caller must clear_session when done.
    """
    m = SeqNN(copy.deepcopy(params["model"]))
    head_name = f"per_bin_f{fold}"
    trunk_out = m.model_trunk.output
    y = tf.keras.layers.Dense(1, name=head_name)(trunk_out)
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

    ft.load_weights(model_path)

    first_changed = _weights_changed(w_first_before, _layer_weights_snapshot(first_trunk))
    last_changed = _weights_changed(w_last_before, _layer_weights_snapshot(last_trunk))
    head_changed = _weights_changed(w_head_before, _layer_weights_snapshot(head_layer))

    print(
        f"    [{model_path}] first trunk ({first_trunk.name}) changed? {first_changed} | "
        f"last trunk ({last_trunk.name}) changed? {last_changed} | head ({head_name}) changed? {head_changed}"
    )
    if not (first_changed and last_changed and head_changed):
        raise ValueError(
            f"Expected trunk (first/last) and head to update after load_weights({model_path!r}). "
            f"got first={first_changed}, last={last_changed}, head={head_changed}"
        )

    return ft


# ================================================================
#   RUN ALL MODELS OVER ALL TEST SETS
# ================================================================
def run_models_over_test_sets(test_sets_by_stride, model_name_list, params):
    """
    test_sets_by_stride: dict[stride_bp -> list of test set dicts].
    For each (model_name, cv, fold) checkpoint under Models/<model_name>/, predict on all
    test sets (for each stride) where model_name is in train_chroms; accumulate sum_pred
    and count per test set. train_chroms uses the same strings as Source.name (HF layout).
    Only loads models that appear in at least one test set's train_chroms.
    """
    strides = sorted(test_sets_by_stride.keys())
    chroms_to_load = set()
    for ts in test_sets_by_stride[strides[0]]:
        chroms_to_load.update(ts["train_chroms"])
    for model_name in model_name_list:
        if model_name not in chroms_to_load:
            continue
        for cv in range(5):
            for fold in range(8):
                model_path = FINETUNE_H5_TEMPLATE.format(
                    model_name=model_name, cv=cv, fold=fold
                )
                if not os.path.exists(model_path):
                    continue
                print(f"\n[MODEL] model_name={model_name} cv={cv} fold={fold}")
                ft = load_finetuned_model(model_path, params, fold)
                for stride_bp in strides:
                    test_sets = test_sets_by_stride[stride_bp]
                    for ts in test_sets:
                        if ts["X"] is None:
                            continue
                        if model_name not in ts["train_chroms"]:
                            continue
                        X = ts["X"]
                        n_samples = X.shape[0]
                        # Predict in chunks to avoid OOM when X is large (e.g. Mmmyco 2356 windows)
                        pred_chunks = []
                        for start in range(0, n_samples, INFERENCE_CHUNK_SIZE):
                            end = min(start + INFERENCE_CHUNK_SIZE, n_samples)
                            chunk = X[start:end]
                            X_t = tf.cast(tf.convert_to_tensor(chunk), tf.bfloat16)
                            p_chunk = ft.predict(X_t, batch_size=len(chunk), verbose=1 if start == 0 else 0)
                            pred_chunks.append(p_chunk.astype(np.float32))
                        p = np.concatenate(pred_chunks, axis=0)
                        if ts["sum_pred"] is None:
                            ts["sum_pred"] = p
                            ts["count"] = 1
                        else:
                            ts["sum_pred"] += p
                            ts["count"] += 1
                del ft
                tf.keras.backend.clear_session()


# ================================================================
#   BUILD ALL TEST SETS (ALL WINDOWS PER GENOME)
# ================================================================
def _build_one_test_set(test_genome, train_chroms, stride_bp):
    """Build X, Y, starts for one genome at one stride. Returns (X, Y, starts) or (None, None, None) if no windows."""
    wins_per_source = build_windows_per_source(test_genome, WINDOW_BP, stride=stride_bp)
    X_list = []
    Y_list = []
    starts_list = []
    for si, source_wins in enumerate(wins_per_source):
        s = test_genome[si]
        L = len(s.seq)
        for win in source_wins:
            a, b = win
            X_list.append(s.X_fwd[a:b])
            Y_list.append(crop_and_bin_cov(s.cov_fwd[a:b], CROP_BP, BIN_SIZE_BP))
            starts_list.append(a)
            X_list.append(s.X_rc[L - b : L - a])
            Y_list.append(crop_and_bin_cov(s.cov_rev[a:b][::-1], CROP_BP, BIN_SIZE_BP))
            starts_list.append(a)
    if len(X_list) == 0:
        return None, None, None
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    starts = np.array(starts_list, dtype=int)
    return X, Y, starts


def build_all_test_sets(all_sources, ablation_mode="full", strides=None):
    """
    Precompute inputs X, labels Y, and start indices for every genome at each stride.
    Returns test_sets_by_stride: dict[stride_bp -> list of test set dicts], and model_name_list.
    Same genome order in each list; test_sets_by_stride[s][i] is genome i at stride s.

    ablation_mode: "full" | "human" | "bacteria" | "syn"
      - full: train_chroms = Source.name for all genomes except test (leave-one-out).
      - human: train_chroms = ABLATION_HUMAN_MODELS minus the test genome's Source.name.
      - bacteria: ABLATION_BACTERIA_MODELS minus test Source.name.
      - syn: ABLATION_SYN_MODELS minus test Source.name.

    train_chroms entries match Models/<name>/ on disk (Source.name), not NPZ/FASTA keys (Source.chrom).
    """
    if strides is None:
        strides = REPORT_STRIDES
    model_name_list = []
    for genome_list in all_sources:
        model_name = genome_list[0].name
        if model_name not in model_name_list:
            model_name_list.append(model_name)

    test_sets_by_stride = {s: [] for s in strides}

    for test_idx, test_genome in enumerate(all_sources):
        test_name = test_genome[0].name
        test_chrom = test_genome[0].chrom
        print(
            f"\n=== Preparing test set for genome = {test_name} "
            f"(data chrom={test_chrom}, model_dir={test_name}) ==="
        )

        if ablation_mode == "full":
            train_genomes = [g for i, g in enumerate(all_sources) if i != test_idx]
            train_chroms = [g[0].name for g in train_genomes]
        elif ablation_mode == "human":
            train_chroms = [c for c in ABLATION_HUMAN_MODELS if c != test_name]
        elif ablation_mode == "bacteria":
            train_chroms = [c for c in ABLATION_BACTERIA_MODELS if c != test_name]
        elif ablation_mode == "syn":
            train_chroms = [c for c in ABLATION_SYN_MODELS if c != test_name]
        else:
            raise ValueError(f"ablation_mode must be 'full', 'human', 'bacteria', or 'syn'; got {ablation_mode!r}")

        # Shared checkpoint: two parts of Human_chr_7 both use Models/Human_chr_7/; full LOOCV
        # leaves the other part in train, so test_name can still appear in train_chroms.
        print(f"  [DEBUG] data chrom = {test_chrom!r}  |  train_chroms (model dirs) = {train_chroms}")
        if test_name in train_chroms:
            print(
                f"  [NOTE] test model {test_name!r} is in train_chroms "
                f"(expected if another genome part shares this checkpoint)."
            )

        for stride_bp in strides:
            X, Y, starts = _build_one_test_set(test_genome, train_chroms, stride_bp)
            n_windows = len(starts) if starts is not None else 0
            print(f"  stride={stride_bp} bp: {n_windows} windows (Fwd+Rev)")
            test_sets_by_stride[stride_bp].append({
                "genome_name": test_name,
                "chrom": test_chrom,
                "train_chroms": train_chroms,
                "X": X,
                "Y": Y,
                "starts": starts,
                "sum_pred": None,
                "count": 0,
            })

    return test_sets_by_stride, model_name_list


# ================================================================
#   LEAVE ONE GENOME OUT EVALUATION
# ================================================================
def evaluate_loocv(all_sources, ablation_mode="full"):
    """
    ablation_mode: "full" | "human" | "bacteria" | "syn"
      - full: leave-one-out over all genomes.
      - human: only models under Human_chr_7, HPRT1 (ABLATION_HUMAN_MODELS).
      - bacteria: only M_pneumoniae, M_mycoides.
      - syn: only Data_storage_chr, HPRT1R.
    Reports correlations for both strides in REPORT_STRIDES (1024 bp and 14336 bp).
    """
    all_results = []

    # preload everything
    for genome_list in all_sources:
        load_and_attach(genome_list)

    params = load_model_params()

    # 1) Build test sets for each stride
    test_sets_by_stride, model_name_list = build_all_test_sets(
        all_sources, ablation_mode=ablation_mode, strides=REPORT_STRIDES
    )

    # 2) Run all models (predict on both stride-specific test sets)
    run_models_over_test_sets(test_sets_by_stride, model_name_list, params)

    # 3) Compute stats per stride, save NPZ per stride, and one result row per genome
    strides = sorted(test_sets_by_stride.keys())
    n_genomes = len(test_sets_by_stride[strides[0]])
    Path("Correlations").mkdir(parents=True, exist_ok=True)

    for i in range(n_genomes):
        name = test_sets_by_stride[strides[0]][i]["genome_name"]
        row = {"genome": name, "ablation": ablation_mode}
        models_used = None

        for stride_bp in strides:
            ts = test_sets_by_stride[stride_bp][i]
            X, Y, sum_pred, count, starts = ts["X"], ts["Y"], ts["sum_pred"], ts["count"], ts["starts"]

            if count == 0 or Y is None:
                row[f"n_windows_{stride_bp}"] = 0
                row[f"spearman_median_{stride_bp}"] = np.nan
                continue
            models_used = count
            Y_pred = sum_pred / float(count)
            n_windows = Y.shape[0]
            per_window_corrs = np.array(
                [spearmanr(Y_pred[j], Y[j]).correlation for j in range(n_windows)], dtype=float
            )
            median_corr = np.nanmedian(per_window_corrs)
            row[f"n_windows_{stride_bp}"] = n_windows
            row[f"spearman_median_{stride_bp}"] = median_corr
            print(f"[RESULT] genome={name} stride={stride_bp} bp n_windows={n_windows} spearman_median={median_corr}")

            npz_path = Path("Results/Correlations") / f"correlations_{name}_{ablation_mode}_stride{stride_bp}.npz"
            np.savez(npz_path, starts=starts, correlations=per_window_corrs)

        row["models_used"] = models_used if models_used is not None else 0
        all_results.append(row)

    # Global CSV with both stride columns
    df_all = pd.DataFrame(all_results)
    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"leave_one_genome_out_results_{ablation_mode}.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved global results to {out_csv}")


# ================================================================
#   MAIN
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Leave-one-genome-out evaluation. Ablation: restrict which genomes' models are used for prediction."
    )
    p.add_argument(
        "--ablation",
        choices=["full", "human", "bacteria", "syn"],
        default="full",
        help="full = leave-one-out over all genomes. "
             "human = only checkpoints under Human_chr_7, HPRT1. "
             "bacteria = only M_pneumoniae, M_mycoides. "
             "syn = only Data_storage_chr, HPRT1R. "
             "Names match Models/<name>/ (HF layout), not NPZ/FASTA keys.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ablation_mode = args.ablation
    print(f"Ablation mode: {ablation_mode}")

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

    evaluate_loocv(all_sources, ablation_mode=ablation_mode)


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()