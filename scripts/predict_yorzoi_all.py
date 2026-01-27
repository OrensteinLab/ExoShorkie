#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Yorzoi Imports
from yorzoi.model.borzoi import Borzoi
from yorzoi.config import BorzoiConfig
from yorzoi.dataset import GenomicDataset
from yorzoi.utils import untransform_then_unbin
import json
import re

# ================== Configuration ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Parameters
INPUT_LEN = 4992 
PRED_WINDOW = 3000 
BIN_SIZE = 10 

# Evaluation Parameters (MUST MATCH SHORKIE EXACTLY)
EVAL_WINDOW = 16384
EVAL_CROP = 1024
EVAL_BIN = 16
EVAL_STRIDE = 1024
TRACKS_JSON = "yorzoi/track_annotation.json"

@dataclass
class Source:
    name: str; npz_fwd: str; npz_rev: str; chrom: str; fa_path: str
    cov_fwd: np.ndarray = None; cov_rev: np.ndarray = None; seq: str = None


def extract_human_chr_indices(json_file):
    with open(json_file) as f:
        tracks = json.load(f)

    plus = tracks["+"]
    M = len(plus)

    pattern = re.compile(r"NC_0000(0[1-9]|1[0-9]|2[0-4])\.\d+")

    plus_idx = [i for i, name in enumerate(plus) if pattern.search(name)]
    minus_idx = [i + M for i in plus_idx]

    assert len(plus_idx) == 10, f"Expected 10 plus indices, got {len(plus_idx)}"
    assert len(minus_idx) == 10, f"Expected 10 minus indices, got {len(minus_idx)}"

    return plus_idx, minus_idx

# ================== Helpers ==================
def load_data(sources):
    for s in sources:
        records = list(SeqIO.parse(s.fa_path, "fasta"))
        if len(records) == 1:
            s.seq = str(records[0].seq)
        else:
            rec = next((r for r in records if s.chrom in r.id), None)
            if rec is None:
                raise ValueError(f"Chrom {s.chrom} not found")
            s.seq = str(rec.seq)

        s.seq = s.seq.upper().replace("N", "A")

        with np.load(s.npz_fwd, allow_pickle=True) as z:
            key = s.chrom if s.chrom in z else "coverage"
            s.cov_fwd = np.abs(z[key]).reshape(-1)

        with np.load(s.npz_rev, allow_pickle=True) as z:
            key = s.chrom if s.chrom in z else "coverage"
            s.cov_rev = np.abs(z[key]).reshape(-1)

        # ADD THIS CHECK HERE
        if (
            len(s.cov_fwd) != len(s.seq)
            or len(s.cov_rev) != len(s.seq)
        ):
            raise ValueError(
                f"Length mismatch for {s.name}: "
                f"seq={len(s.seq)} "
                f"fwd={len(s.cov_fwd)} "
                f"rev={len(s.cov_rev)}"
            )

    return sources


def run_inference_map(model, seq_str):
    """
    Runs model on full sequence and returns a dense prediction map (Tracks, L).
    """
    L = len(seq_str)
    centers = np.arange(INPUT_LEN//2, L - INPUT_LEN//2, PRED_WINDOW)
    oh = GenomicDataset.one_hot_encode(seq_str)
    seq_t = torch.tensor(oh, dtype=torch.float32, device=DEVICE)
    
    pred_map = None 
    
    with torch.no_grad():
        for i in range(0, len(centers), 128):
            batch_centers = centers[i:i+128]
            batch_seqs = torch.stack([seq_t[c-INPUT_LEN//2 : c+INPUT_LEN//2] for c in batch_centers])
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch_seqs) 
            
            out_1bp = untransform_then_unbin(out, BIN_SIZE).cpu().numpy() 

            if i == 0:
                print("Output shape (batch, tracks, pred_window):", out_1bp.shape)
            
            if pred_map is None: 
                n_tracks = out_1bp.shape[1]
                pred_map = np.zeros((n_tracks, L), dtype=np.float16)

            for idx, c in enumerate(batch_centers):
                s, e = c - PRED_WINDOW//2, c + PRED_WINDOW//2
                if s >= 0 and e <= L: 
                    pred_map[:, s:e] = out_1bp[idx]
                    
    return pred_map

def evaluate_genome_shorkie_style(model, sources, genome_name, human_plus_idx, human_minus_idx):
    """
    Evaluates Fwd and Rev simultaneously using Shorkie's exact window definitions.
    """
    all_results = []
    
    for s in sources:
        L = len(s.seq)

        # Run yorzoi model on fwd and rev sequences
        map_fwd = run_inference_map(model, s.seq)

        rc_seq = str(Seq(s.seq).reverse_complement())
        map_rev = run_inference_map(model, rc_seq)

        # collapse tracks early (1 x L)
        map_fwd_avg = map_fwd[human_plus_idx].mean(axis=0).astype(np.float32, copy=False)
        map_rev_avg = map_rev[human_plus_idx].mean(axis=0).astype(np.float32, copy=False)

        zero_pct_fwd = np.mean(map_fwd_avg == 0) * 100
        print(f"Zero values fwd: {zero_pct_fwd:.2f}%")

        zero_pct_rev = np.mean(map_rev_avg == 0) * 100
        print(f"Zero values reverse: {zero_pct_rev:.2f}%")

        # 2. Iterate EXACTLY like Shorkie (Genomic Coordinates)
        # Shorkie: make_windows(0, L, win_bp, stride) -> yields (start, end)
        starts = np.arange(0, L - EVAL_WINDOW + 1, EVAL_STRIDE)
        
        for a in tqdm(starts, desc=f"Eval {s.name}", leave=False):
            b = a + EVAL_WINDOW
            
            # Truth
            y_fwd = s.cov_fwd[a:b]
            y_fwd_bin = y_fwd[EVAL_CROP:-EVAL_CROP].reshape(-1, EVAL_BIN).sum(axis=1)
            
            # Pred
            p_fwd = map_fwd_avg[a:b]
            p_fwd_bin = p_fwd[EVAL_CROP:-EVAL_CROP].reshape(-1, EVAL_BIN).sum(axis=1)
            corr_fwd = spearmanr(p_fwd_bin, y_fwd_bin).correlation
            all_results.append({'start': a, 'corr': corr_fwd})
                        
            
            # Truth: Note Shorkie reverses the genomic coverage slice
            y_rev = s.cov_rev[a:b][::-1].copy()
            y_rev_bin = y_rev[EVAL_CROP:-EVAL_CROP].reshape(-1, EVAL_BIN).sum(axis=1)
            
            # Pred: map_rev is aligned to the RC sequence.
            # Genomic [a:b] corresponds to RC [L-b : L-a]
            rc_start = L - b
            rc_end   = L - a
            
            p_rev = map_rev_avg[rc_start:rc_end]
            p_rev_bin = p_rev[EVAL_CROP:-EVAL_CROP].reshape(-1, EVAL_BIN).sum(axis=1)
            corr_rev = spearmanr(p_rev_bin, y_rev_bin).correlation
            all_results.append({'start': a, 'corr': corr_rev})

        
    return all_results

# ================== Main ==================
def main():
    human_plus_idx, human_minus_idx = extract_human_chr_indices(TRACKS_JSON)
    print("Human plus idx:", human_plus_idx)
    print("Human minus idx:", human_minus_idx)
    print(f"Using device: {DEVICE}")
    ckpt = "tom-ellis-lab/yorzoi"
    
    # Load Model
    cfg_path = hf_hub_download(repo_id=ckpt, filename="config.json")
    with open(cfg_path, "r") as f: cfg_dict = json.load(f)
    config = BorzoiConfig(**cfg_dict)
    model = Borzoi(config)
    state_dict = load_file(hf_hub_download(repo_id=ckpt, filename="model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    
    # Define Sources
    human = [
        Source("human", "Data/normalized_expression/Human_part1_fwd_norm.npz", "Data/normalized_expression/Human_part1_rev_norm.npz", "chr7", "Data/genome/chr7_part1.fa"),
        Source("human", "Data/normalized_expression/Human_part2_fwd_norm.npz", "Data/normalized_expression/Human_part2_rev_norm.npz", "chr7", "Data/genome/chr7_part2.fa")
    ]
    mpneumo = [Source("Mpneumo", "Data/normalized_expression/Mpneumo_fwd_norm.npz", "Data/normalized_expression/Mpneumo_rev_norm.npz", "Mpneumo", "Data/genome/S288c_Mpneumo.fa")]
    mmyco = [Source("Mmmyco", "Data/normalized_expression/Mmmyco_fwd_norm.npz", "Data/normalized_expression/Mmmyco_rev_norm.npz", "Mmmyco", "Data/genome/W303_Mmmyco.fa")]
    hprt1 = [Source("HPRT1", "Data/normalized_expression/HPRT1_fwd_norm.npz", "Data/normalized_expression/HPRT1_rev_norm.npz", "HPRT1", "Data/genome/HPRT1.fa")]
    hprt1r = [Source("HPRT1R", "Data/normalized_expression/HPRT1R_fwd_norm.npz", "Data/normalized_expression/HPRT1R_rev_norm.npz", "HPRT1R", "Data/genome/HPRT1R.fa")]
    dchr = [Source("dChr", "Data/normalized_expression/dChr_fwd_norm.npz", "Data/normalized_expression/dChr_rev_norm.npz", "dChr", "Data/genome/dChr.fa")]

    experiments = [("Mpneumo", mpneumo), ("Mmmyco", mmyco), ("human", human), ("dChr", dchr), ("HPRT1", hprt1), ("HPRT1R", hprt1r)]

    final_summary = []

    for name, srcs in experiments:
        print(f"\n=== Processing {name} ===")
        sources = load_data(srcs)
        
        # Run Evaluation
        results = evaluate_genome_shorkie_style(model, sources, name, human_plus_idx, human_minus_idx)
        
        if not results: continue
        
        # Process Results
        starts = np.array([r['start'] for r in results], dtype=int)
        corrs = np.array([r['corr'] for r in results], dtype=np.float32)

        best_median = np.nanmedian(corrs)

        print(f"  Human tracks averaged (n=20). Median Spearman: {best_median:.4f}")
        print(f"  Windows: {len(starts)}")

        np.savez(
            f"Correlations/correlations_yorzoi_{name}.npz",
            starts=starts,
            correlations=corrs,
            human_track_fwd_indices=human_plus_idx,
            human_track_rev_indices=human_minus_idx
        )

        final_summary.append({"genome": name, "n_windows": len(starts), "spearman_median": best_median})

    pd.DataFrame(final_summary).to_csv("leave_one_genome_out_results_yorzoi.csv", index=False)
    print("\n[DONE]")

if __name__ == "__main__":
    main()