import numpy as np
import gzip
import tensorflow as tf
from Bio import SeqIO
from Bio.Seq import Seq
from dataclasses import dataclass

# --- Data Classes ---
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
    start0: int = 0
    end0: int = 0
    X_fwd: np.ndarray = None
    X_rc: np.ndarray = None

# --- Core Helpers ---
def fetch_coverage(npz_path: str, chrom: str) -> np.ndarray:
    data = np.load(npz_path)
    if chrom not in data:
        raise KeyError(f"Chromosome key '{chrom}' not found in {npz_path}")
    return np.abs(np.array(data[chrom], dtype=np.float64))

def fetch_chr_seq(path: str, chrom: str) -> str:
    with open(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            if rec.id == chrom:
                return str(rec.seq).upper()
    raise KeyError(f"Chromosome '{chrom}' not found in {path}")

def per_bin_dna_features(seq: str) -> np.ndarray:
    s = seq.upper()
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    onehot = np.zeros((len(seq), 5), dtype=np.uint8)
    for ch, col in ((b"A",0), (b"C",1), (b"G",2), (b"T",3)):
        onehot[arr == ch[0], col] = 1
    onehot[arr == ord("N"), 4] = 1
    return onehot

def build_shorkie_features(seq: str) -> np.ndarray:
    dna4 = per_bin_dna_features(seq)
    species = np.zeros((165,), dtype=np.uint8)
    species[114] = 1
    species_tiled = np.tile(species, (dna4.shape[0], 1))
    return np.concatenate([dna4, species_tiled], axis=1)

def precompute_full_features(sources):
    for s in sources:
        print(f"Precomputing features for {s.name}...")
        s.X_fwd = build_shorkie_features(s.seq)
        seq_rc_full = str(Seq(s.seq).reverse_complement())
        s.X_rc = build_shorkie_features(seq_rc_full)
    return sources

def load_coverages_for_sources(sources):
    for s in sources:
        s.cov_fwd = fetch_coverage(s.npz_fwd, s.chrom)
        s.cov_rev = fetch_coverage(s.npz_rev, s.chrom)
        s.start0 = 0
        s.end0 = s.cov_fwd.shape[0]
    return sources

def attach_sequences_for_sources(sources):
    for s in sources:
        s.seq = fetch_chr_seq(s.fa_path, s.chrom)
    return sources

# --- Windowing & Splitting ---
def make_windows(start, end, win, stride):
    stops = []
    pos = start
    last_start = end - win
    while pos <= last_start:
        stops.append((pos, pos + win))
        pos += stride
    return stops

def build_windows_per_source(sources, win_bp, target_wins = 10_000, train_ratio = 0.8):

    # Effective training length across sources (approx for fold)
    total_train_len = sum(int(len(s.seq) * train_ratio) for s in sources)

    # Expected windows if stride=1 is roughly sum((L_train - win_bp)/1)+1, so stride scales linearly.
    stride = max(1, int((total_train_len - win_bp) / target_wins))

    windows_per_source = []
    strides = []
    for si, s in enumerate(sources):
        L = len(s.seq)
        wins = make_windows(0, L, win_bp, stride)
        windows_per_source.append(wins)
        strides.append(stride)

    return windows_per_source, strides


def build_test_windows_per_source(sources, win_bp, stride=1024):

    windows_per_source = []
    strides = []
    for si, s in enumerate(sources):
        L = len(s.seq)
        wins = make_windows(0, L, win_bp, stride)
        windows_per_source.append(wins)
        strides.append(stride)

    return windows_per_source, strides

def build_windows_full_genome(sources, win_bp, target_windows=10000):
    """
    Calculates a stride to generate exactly 'target_windows' across ALL provided sources.
    Returns a FLAT list of tuples (si, wi, a, b) ready for training.
    """
    all_windows = []
    
    # 1. Calculate Total Effective Length of ALL sources
    total_effective_length = 0
    for s in sources:
        # Use max(0, ...) to safely handle chromosomes shorter than win_bp
        total_effective_length += max(0, len(s.seq) - win_bp)

    # 2. Calculate Stride
    if total_effective_length == 0:
        print("Error: Sequences are shorter than the window size.")
        return [], 1
        
    stride = int(total_effective_length / target_windows)
    stride = max(1, stride) # Ensure stride is at least 1 bp

    print(f"--- Configuration ---")
    print(f"Target Windows:      {target_windows}")
    print(f"Total Genome Length: {total_effective_length:,} bp")
    print(f"Calculated Stride:   {stride} bp")
    print(f"---------------------")

    # 3. Generate Windows and flatten them immediately
    for si, s in enumerate(sources):
        L = len(s.seq)
        if L <= win_bp:
            continue
            
        wins = make_windows(0, L, win_bp, stride)
        
        # Enumerate the windows to get 'wi' and append to the flat list
        for wi, (a, b) in enumerate(wins):
            all_windows.append((si, wi, a, b))

    print(f"Total Windows Generated: {len(all_windows)}")

    return all_windows, stride


def make_5fold_splits(sources, windows_per_source, n_folds=5):
    cv_splits = []
    for cv in range(n_folds):
        train_pairs, test_pairs = [], []
        for si, s in enumerate(sources):
            wins = windows_per_source[si]
            if not wins: continue
            L = len(s.seq)
            test_start = int(cv * L / n_folds)
            test_end   = int((cv + 1) * L / n_folds) if cv < n_folds - 1 else L
            for wi, (a, b) in enumerate(wins):
                if b <= test_start or a >= test_end:
                    train_pairs.append((si, wi, a, b))
                elif a > test_start and b < test_end:
                    test_pairs.append((si, wi, a, b))
        cv_splits.append((train_pairs, test_pairs))
    return cv_splits

# --- Normalization & Generator ---
def crop_and_bin_cov(cov, crop_bp, bin_bp):
    x = cov[crop_bp: cov.shape[0]-crop_bp]
    L = (x.shape[0] // bin_bp) * bin_bp
    x = x[:L]
    bins = x.reshape(-1, bin_bp).sum(axis=1)
    return bins.astype(np.float32)

def compute_logz_stats_multi(sources, train_pairs, crop_bp, bin_size_bp):
    ys = []
    for (si, wi, a, b) in train_pairs:
        s = sources[si]
        y_fwd = crop_and_bin_cov(s.cov_fwd[a:b], crop_bp, bin_size_bp)
        y_rc  = crop_and_bin_cov(s.cov_rev[a:b][::-1], crop_bp, bin_size_bp)
        ys.extend([y_fwd, y_rc])
    ys_log = np.log1p(np.concatenate(ys))
    return ys_log.mean(), ys_log.std() or 1e-8

def precompute_window_labels(
    sources,
    pairs,
    crop_bp,
    bin_size_bp,
    apply_norm=True,
    mu=0.0,
    sigma=1.0,
):

    y_fwd_dict, y_rc_dict = {}, {}
    seen = set()

    for (si, wi, a, b) in pairs:
        if (si, wi) in seen:
            continue
        seen.add((si, wi))

        s = sources[si]

        # Forward strand
        bf = crop_and_bin_cov(s.cov_fwd[a:b], crop_bp, bin_size_bp)

        # Reverse strand (reverse orientation)
        br = crop_and_bin_cov(s.cov_rev[a:b][::-1], crop_bp, bin_size_bp)

        if apply_norm:
            bf = (np.log1p(bf) - mu) / sigma
            br = (np.log1p(br) - mu) / sigma

        y_fwd_dict[(si, wi)] = bf.astype(np.float32)
        y_rc_dict[(si, wi)] = br.astype(np.float32)

    return y_fwd_dict, y_rc_dict

def cast_for_model(x_uint8, y_raw):
    return tf.cast(x_uint8, tf.bfloat16), y_raw

def make_ds_from_pairs(sources, pairs, y_fwd_dict, y_rc_dict,
                       window_bp, batch_size, shuffle=True, seed=42):
    N = len(pairs)
    # RNG initialized OUTSIDE generator
    rng = np.random.default_rng(seed)

    def gen():
        idx = np.arange(N, dtype=np.int32)
        if shuffle:
            rng.shuffle(idx)  # Uses persistent RNG state

        for i in idx:
            si, wi, a, b = pairs[i]
            s = sources[si]
            L = len(s.seq)
            x_fwd = s.X_fwd[a:b]
            yield x_fwd, y_fwd_dict[(si, wi)]
            
            x_rc = s.X_rc[L-b : L-a]
            yield x_rc, y_rc_dict[(si, wi)]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(window_bp, 170), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    )
    ds = ds.map(cast_for_model, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Add to src/data_loader.py

def compute_global_stats(sources, windows_per_source, crop_bp, bin_size_bp):
    """
    Computes mean/std across ALL windows in all sources for logging.
    """
    ys = []
    total_wins = 0
    for si, s in enumerate(sources):
        wins = windows_per_source[si]
        total_wins += len(wins)
        for (a, b) in wins:
            # Forward
            y_fwd = crop_and_bin_cov(s.cov_fwd[a:b], crop_bp, bin_size_bp)
            # Reverse
            y_rc  = crop_and_bin_cov(s.cov_rev[a:b][::-1], crop_bp, bin_size_bp)
            ys.extend([y_fwd, y_rc])

    ys_all = np.concatenate(ys).astype(np.float64)
    ys_log = np.log1p(ys_all)
    
    mu, sigma = ys_log.mean(), ys_log.std()
    print(f"\n[GLOBAL STATS] Windows: {total_wins} | Mean: {mu:.4f} | Std: {sigma:.4f}")
    return mu, sigma



def load_coverages_and_seqs(sources):
    """
    Wrapper to run both load_coverages and attach_sequences in one go.
    """
    load_coverages_for_sources(sources)
    attach_sequences_for_sources(sources)