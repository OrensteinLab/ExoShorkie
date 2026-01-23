#!/usr/bin/env python3

# ---------- Reproducibility (must be before importing TF) ----------
import os, random, numpy as np
SEED = 42
os.environ["CUDA_VISIBLE_DEVICES"]  = "1" # Adjust to your available GPU index
os.environ["PYTHONHASHSEED"]        = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"]  = "1"
os.environ["TF_CUDNN_DETERMINISTIC"]= "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # deterministic CPU math

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
import time
tf.random.set_seed(SEED)

from tensorflow.keras import mixed_precision

# A100/H100: mixed_bfloat16 is best. For older GPUs (V100/T4), use mixed_float16.
mixed_precision.set_global_policy("float32") 

# ------------------------------------------------------------------
import gzip, json
from dataclasses import dataclass
from typing import List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
import argparse
from baskerville.seqnn import SeqNN
from src.data_loader import *

# ---------- Constants ----------
PARAMS_JSON = "Models/shorkie/params.json"

WINDOW_BP     = 16384
CROP_BP       = 1024
INTERIOR_BP   = WINDOW_BP - 2*CROP_BP           # 14336
BIN_SIZE_BP   = 16                               
STRIDE_BP     = INTERIOR_BP                      
BLOCK_SIZE    = 16384

# Optimized Batch Size for GPU-side generation
# Since we don't transfer data, we can push this higher on A100s
MUTATION_BATCH_SIZE = 512 

# ---------- GPU Optimization Function ----------
@tf.function(jit_compile=True)
def predict_on_gpu(model_func, X_ref_tensor, indices, alt_bases):
    """
    Generates mutations and runs prediction entirely on the GPU.
    
    Args:
        model_func: The Keras model (or model.call)
        X_ref_tensor: (1, L, F) - The reference sequence on GPU
        indices: (B,) - The sequence positions (0..L-1) to mutate
        alt_bases: (B,) - The new base values (0,1,2,3)
    
    Returns:
        Predictions tensor of shape (B, Output_Len)
    """
    batch_size = tf.shape(indices)[0]

    L = tf.shape(X_ref_tensor)[1]
    
    # 1. Tile the reference to create the batch (Happens in VRAM)
    # Shape: (B, L, F)
    X_batch = tf.tile(X_ref_tensor, [batch_size, 1, 1])
    
    # 2. Prepare indices for masking
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    # (B, 2) tensor of [[batch_0, pos_0], [batch_1, pos_1], ...]
    mask_indices = tf.stack([batch_indices, tf.cast(indices, tf.int32)], axis=1) 
    
    # 3. Create One-Hot Updates (B, 4)
    # 0=A, 1=C, 2=G, 3=T
    one_hot_updates = tf.one_hot(alt_bases, depth=4, dtype=tf.float32)
    
    # 4. Create a mask for the mutation positions
    # Scatter '1's to the mutation positions: Shape (B, L)
    mask = tf.scatter_nd(mask_indices, tf.ones(batch_size, dtype=tf.float32), [batch_size, L]) 
    mask = tf.expand_dims(mask, -1) # (B, L, 1)
    
    # 5. Apply Mutation
    # Split into DNA channels (0-4) and features (4-end)
    X_dna = X_batch[:, :, 0:4]
    X_other = X_batch[:, :, 4:]
    
    # Zero out the DNA channels at mutation sites
    X_dna = X_dna * (1.0 - mask)
    
    # Add the new bases
    # Scatter (B, 4) updates into (B, L, 4) using the same indices
    updates_expanded = tf.scatter_nd(mask_indices, one_hot_updates, [batch_size, L, 4])
    X_dna = X_dna + updates_expanded
    
    # Recombine features
    X_final = tf.concat([X_dna, X_other], axis=-1)
    
    # 6. Predict
    return model_func(X_final, training=False)

def load_model(model):

    # 5. Load Model
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    params["model"]["num_features"] = 170

    model_path = f"Distillation/{model}/student_{model}_distilled.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")

    # load model
    print(f"\n[MODEL] path={model_path}")
    m = SeqNN(params["model"])
    trunk_out = m.model_trunk.output
    
    # Distillation Head
    y = tf.keras.layers.Dense(1, name="distill_head")(trunk_out)
    y = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, -1))(y)
    student = tf.keras.Model(inputs=m.model.input, outputs=y)

    # --- Weight Check ---
    head_layer = student.get_layer("distill_head")
    head_before = [w.numpy() for w in head_layer.weights]
    trunk_before = [w.numpy().copy() for w in m.model_trunk.weights]

    # load weights
    student.load_weights(model_path)

    # ensure weights changed
    head_after = [w.numpy() for w in head_layer.weights]
    trunk_after = [w.numpy() for w in m.model_trunk.weights]

    trunk_changed = any(np.any(b != a) for b, a in zip(trunk_before, trunk_after))
    head_changed  = any(np.any(b != a) for b, a in zip(head_before, head_after))

    print(f"    Trunk weights changed? {trunk_changed}")
    print(f"    Head weights changed?  {head_changed}")

    if not trunk_changed:
        raise RuntimeError(f"Trunk weights for {model_path} did not update!")
    if not head_changed:
        raise RuntimeError(f"Distill head weights for {model_path} did not update!")
    
    return student


# ---------- Helper Classes/Functions ----------
@dataclass
class Source:
    name: str
    chrom: str
    fa_path: str
    seq: str = None

def load_seq(src: Source, rev: bool) -> Source:
    seq = fetch_chr_seq(src.fa_path, src.chrom).upper()
    if rev:
        seq = str(Seq(seq).reverse_complement())
    src.seq = seq
    return src

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute ISM over a single genome using Shorkie fine-tuned model."
    )
    p.add_argument("--model", type=str, required=True, help="Model name")
    p.add_argument("--chrom", type=str, required=True, help="Chromosome name")
    p.add_argument("--fasta",   type=str, required=True, help="Genome FASTA path")
    p.add_argument("--mu", type=float, required=True, help="log z-score normalization mu")
    p.add_argument("--sigma", type=float, required=True, help="log z-score normalization sigma")
    p.add_argument("--rev", action="store_true", help="whether to run on reverse complement")
    return p.parse_args()

def main():
    tf.get_logger().setLevel("ERROR")
    args = parse_args()

    MODEL = args.model
    CHROM = args.chrom
    FA_PATH = args.fasta
    ORIENT = "rev" if args.rev else "fwd"
    
    mu = args.mu
    sigma = args.sigma

    # 1. Load Data
    src = Source(name=CHROM, chrom=CHROM, fa_path=FA_PATH)
    src = load_seq(src,rev=args.rev)
    L = len(src.seq)
    print(f"{src.name}: {src.chrom}, length = {L}")

    # 2. Build Windows
    windows = make_windows(0, L, WINDOW_BP, STRIDE_BP)
    n_windows = len(windows)
    print(f"Using WINDOW_BP={WINDOW_BP}, CROP_BP={CROP_BP}, INTERIOR_BP={INTERIOR_BP}")
    print(f"Total windows: {n_windows}")

    # load student model
    student = load_model(MODEL)

    # 3. Prepare Output Directory & Memmap
    output_dir = "ISM/Maps/{CHROM}".format(CHROM=CHROM)
    os.makedirs(output_dir, exist_ok=True)
    
    ism_path = f"{output_dir}/{MODEL}_model_chr_{CHROM}_{ORIENT}.dat"
    ism_scores = np.memmap(ism_path, dtype="float32", mode="w+", shape=(L, 4))
    ism_scores[:] = 0.0 # Initialize to zero
    ism_scores.flush()
    print(f"Initialized memmap at: {ism_path}")

    # 4. Lookup Tables
    base_lut = np.full(256, -1, dtype=np.int8)
    base_lut[ord("A")] = 0
    base_lut[ord("C")] = 1
    base_lut[ord("G")] = 2
    base_lut[ord("T")] = 3

    alt_table = np.array([
        [1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]
    ], dtype=np.int8)

    # Pre-encode sequence
    seq_bytes = np.frombuffer(src.seq.encode("ascii"), dtype=np.uint8)

    # 6. ISM Loop
    interior_positions = np.arange(CROP_BP, WINDOW_BP - CROP_BP, dtype=np.int32)

    for w_idx, (a, b) in enumerate(windows):
        print(f"  Window {w_idx+1}/{n_windows}  [a={a}, b={b}]")

        # Prepare Reference
        seq_win_bytes = seq_bytes[a:b]
        seq_win       = seq_win_bytes.tobytes().decode("ascii")
        X_ref = build_shorkie_features(seq_win).astype("float32")

        # Move Reference to GPU ONCE
        X_ref_tensor = tf.convert_to_tensor(X_ref[None, :, :], dtype=tf.float32)

        print("X ref tensor", X_ref_tensor.shape)

        # Reference Prediction
        ref_pred = student(X_ref_tensor, training=False)
        ref_pred = ref_pred.numpy()

        ref_denorm = np.expm1(ref_pred * sigma + mu)
        baseline_sum = float(ref_denorm.sum())

        pos_block = interior_positions
        
        # Identify valid mutations
        base_codes = seq_win_bytes[pos_block]
        ref_idx_block = base_lut[base_codes]
        valid_mask = ref_idx_block >= 0
        
        valid_pos_win = pos_block[valid_mask]
        valid_ref_idx = ref_idx_block[valid_mask]
        
        # Expand to 3 alts per valid pos
        alt_idx_flat = alt_table[valid_ref_idx].reshape(-1)
        j_win_all    = np.repeat(valid_pos_win, 3)
        global_pos_all = a + j_win_all
        
        mut_idx = j_win_all.shape[0]
        processed = 0
        
        # ---- OPTIMIZED GPU BATCH LOOP ----
        for m_start in range(0, mut_idx, MUTATION_BATCH_SIZE):
            m_end = min(mut_idx, m_start + MUTATION_BATCH_SIZE)
            
            # 1. Get Indices (CPU)
            batch_win_pos = j_win_all[m_start:m_end]
            batch_alts    = alt_idx_flat[m_start:m_end]
            batch_gpos    = global_pos_all[m_start:m_end]

            # 2. Move Indices to GPU (Tiny Transfer)
            idx_tensor = tf.convert_to_tensor(batch_win_pos, dtype=tf.int32)
            alt_tensor = tf.convert_to_tensor(batch_alts, dtype=tf.int32)

            # 3. Generate & Predict on GPU
            preds_chunk = predict_on_gpu(student, X_ref_tensor, idx_tensor, alt_tensor)
            
            # 4. Move Predictions to CPU
            preds_chunk = preds_chunk.numpy()
            
            # 5. Calculate Score & Write to Memmap
            preds_denorm = np.expm1(preds_chunk * sigma + mu)
            sums_chunk   = preds_denorm.sum(axis=1)

            log2fc_chunk = np.log2((sums_chunk + 1.0) / (baseline_sum + 1.0))

            for gpos, ch, delta_val in zip(batch_gpos, batch_alts, log2fc_chunk):
                ism_scores[gpos, ch] += float(delta_val)
            
            processed += (m_end - m_start)
            
            if processed % (MUTATION_BATCH_SIZE * 5) == 0:
                    print(f"      Processed {processed}/{mut_idx} mutations...", end="\r")

        ism_scores.flush()
        print(f"\n    Saved Window {w_idx+1}")

    # Final cleanup
    ism_scores.flush()
    del ism_scores
    print(f"\nDONE. Output saved to {ism_path}")
    print(f"Shape: {(L, 4)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTotal runtime: {duration/60:.2f} minutes ({duration:.1f} seconds)")