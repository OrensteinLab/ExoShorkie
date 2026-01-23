from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from matplotlib.patches import Rectangle
import torch
from Bio import SeqIO
from scipy.stats import pearsonr,spearmanr
# Tangermeme imports
from tangermeme.io import read_meme
from tangermeme.seqlet import recursive_seqlets
from tangermeme.annotate import annotate_seqlets
from src.data_loader import *
import numpy as np
import torch
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator



WINDOW_BP   = 16384
CROP_BP     = 1024
BIN_SIZE_BP = 16
INTERIOR_BP = WINDOW_BP - 2*CROP_BP
N_BINS      = INTERIOR_BP // BIN_SIZE_BP  # 896

BASES           = ["A", "C", "G", "T"]
MAX_E_VALUE     = 0.05
MAX_Q_VALUE     = 0.05

def one_hot(seq: str) -> np.ndarray:
    s = seq.upper()
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    onehot = np.zeros((len(seq), 4), dtype=np.uint8)
    for ch, col in ((b"A",0), (b"C",1), (b"G",2), (b"T",3)):
        onehot[arr == ch[0], col] = 1
    return onehot

def compute_attr_from_ism(ism: np.ndarray, seq: str) -> tuple[np.ndarray, np.ndarray]:
    ism_centered = ism - ism.mean(axis=1, keepdims=True)
    onehot = one_hot(seq).astype(np.float32) 
    attr = (ism_centered * onehot).sum(axis=1).astype(np.float32)
    return ism_centered, attr

def find_seqlets(attr: np.ndarray, threshold: float = 0.001, min_len: int = 6) -> pd.DataFrame:
    # Positives
    X_pos = torch.from_numpy(attr)
    df_pos = recursive_seqlets(X_pos, threshold=threshold, min_seqlet_len=min_len)
    if not df_pos.empty: df_pos["sign"] = 1
    
    # Negatives
    X_neg = torch.from_numpy(-attr)
    df_neg = recursive_seqlets(X_neg, threshold=threshold, min_seqlet_len=min_len)
    if not df_neg.empty: df_neg["sign"] = -1
    
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    if df.empty: return df
    
    # enforce types
    df[["start", "end", "example_idx"]] = df[["start", "end", "example_idx"]].astype(int)

    # ADD STRAND COLUMN
    df["strand"] = df["example_idx"].map({0: "fwd", 1: "rev"})

    return df.sort_values(["strand", "start"]).reset_index(drop=True)

def annotate_seqlets_with_motifs(seqlet_df, onehot, meme_db_path, MAX_E_VALUE=0.05):

    seqlet_df = seqlet_df.reset_index(drop=True)

    motifs = read_meme(meme_db_path)
    names = np.array(list(motifs.keys()))
    n_db = len(names)

    X_seq = torch.from_numpy(np.transpose(onehot, (0, 2, 1))).float()

    print("X_seq shape: ", X_seq.shape)

    idx, pvals = annotate_seqlets(
        X_seq, seqlet_df, meme_db_path, n_nearest=5
    )

    rows = []

    for i, row in seqlet_df.iterrows():
        for j in range(idx.shape[1]):
            p = pvals[i, j].item()
            e = p * n_db
            if e <= MAX_E_VALUE:
                rows.append({
                    **row.to_dict(),
                    "motif_name": names[idx[i, j]],
                    "motif_p_value": p,
                    "motif_e_value": e
                })

    out_df = pd.DataFrame(rows)

    print("Significant motif hits kept:", len(out_df))
    print("Unique seqlets annotated:", out_df['Query_ID'].nunique() if 'Query_ID' in out_df else "NA")

    return out_df


def choose_disagreement_windows(
    seqlets_df,
    attr_gen,
    attr_rand,
    window_size=400,
    filter=0.0,
):
    """
    Keep windows where Pearson(attr_gen, attr_rand) < filter.
    Pearson is computed per strand independently.

    Returns: list of (strand_idx, ws, we, r)
    """
    attr_gen = np.asarray(attr_gen)
    attr_rand = np.asarray(attr_rand)

    # Normalize shapes to (S, L)
    if attr_gen.ndim == 1:
        attr_gen = attr_gen[None, :]
    if attr_rand.ndim == 1:
        attr_rand = attr_rand[None, :]

    if attr_gen.shape != attr_rand.shape:
        raise ValueError(f"shape mismatch: {attr_gen.shape} vs {attr_rand.shape}")

    S, L = attr_gen.shape

    selected = []
    seen = set()  # (strand_idx, ws, we)

    # Keep only significant motif hits
    sig_mask = (
        (seqlets_df["motif_e_value"] <= MAX_E_VALUE) &
        (seqlets_df["motif_name"] != "NA")
    )
    sig_df = seqlets_df[sig_mask].copy()


    for _, hit in sig_df.iterrows():
        start = int(hit.start)
        end = int(hit.end)

        mid = (start + end) // 2
        ws = max(0, mid - window_size // 2)
        we = ws + window_size

        if we > L:
            continue

        s = 0 if hit.strand == "fwd" else 1

        key = (s, ws, we)
        if key in seen:
            continue

        a = attr_gen[s, ws:we]
        b = attr_rand[s, ws:we]

        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() < 3:
            continue

        a = a[finite]
        b = b[finite]

        # Constant window => Pearson undefined
        if np.all(a == a[0]) or np.all(b == b[0]):
            continue

        r = pearsonr(a, b)[0]

        if np.isfinite(r) and r < filter:
            selected.append((s, ws, we, float(r)))
            seen.add(key)

    # Sort by strongest disagreement (most negative first)
    selected.sort(key=lambda x: x[3])
    return selected



def _plot_track(ax_logo, window_seq, attr, seqlets, title, custom_ylim=None):
    Lw = len(window_seq)
    
    # --- 1. Plot Logo ---
    logo_df = pd.DataFrame(index=range(Lw), columns=BASES, data=0.0)
    for i, base in enumerate(window_seq):
        logo_df.at[i, base] = attr[i]
        
    logo = logomaker.Logo(logo_df, color_scheme="classic", ax=ax_logo)
    logo.style_spines(visible=False)
    ax_logo.set_xticks([])
    
    # FIX 1: Add padding to title
    # ax_logo.set_title(title, fontsize=10, pad=25)
    
    # --- FIX: UNIFIED SCALING LOGIC ---
    if custom_ylim is not None:
        # If a global scale is provided, use it strictly
        ax_logo.set_ylim(custom_ylim)
    else:
        # Fallback to local auto-scaling with padding (Original behavior)
        ymin, ymax = ax_logo.get_ylim()
        y_range = max(abs(ymin), abs(ymax))
        ax_logo.set_ylim(-y_range * 1.2, y_range * 1.2)
    
    # --- 2. Annotate Significant Seqlets Only ---
    ymin, ymax = ax_logo.get_ylim() # Re-fetch final limits
    
    for _, r in seqlets.iterrows():
        if r.motif_e_value <= MAX_E_VALUE and r.motif_name != "NA":
            s, e = int(r.start), int(r.end)
            sign = int(r.sign) if "sign" in r else 1

            if sign > 0:
                y0 = 0
                height = ymax - 0
                txt_y = ymax * 0.92
                va = "top"
            else:
                y0 = ymin
                height = 0 - ymin
                txt_y = ymin * 0.92
                va = "bottom"

            # Highlight only the correct side
            ax_logo.add_patch(
                Rectangle(
                    (s - 0.5, y0),
                    e - s,
                    height,
                    color="gray",
                    alpha=0.12,
                    linewidth=0
                )
            )

            # Label
            ax_logo.text(
                (s + e) / 2 - 0.5,
                txt_y,
                r.motif_name,
                ha="center",
                va=va,
                fontsize=9,
                weight="bold"
            )

    return logo

def plot_comparison(seq, attr_gen, seqlets_gen, attr_rand, seqlets_rand, ws, we, motif_name, out_path):
    """
    Plots comparison with unified Y-axis scaling.
    """
    w_seq = seq[ws:we]
    
    # --- 1. Calculate Unified Y-Limits ---
    # Get the specific data slices we are about to plot
    slice_rand = attr_rand[ws:we]
    slice_gen = attr_gen[ws:we]
    
    # Find the maximum absolute value across BOTH models
    max_val_rand = np.max(np.abs(slice_rand))
    max_val_gen  = np.max(np.abs(slice_gen))
    global_max   = max(max_val_rand, max_val_gen)
    
    # Add a 20% buffer for the text labels (similar to your previous 1.2 multiplier)
    limit = global_max * 1.2
    
    # Ensure limit isn't zero (in case of empty/flat signal)
    if limit == 0: limit = 0.1
        
    common_ylim = (-limit, limit)

    # --- 2. Setup Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    
    # --- 3. Plot Random Model ---
    w_s_rand = seqlets_rand[(seqlets_rand.start >= ws) & (seqlets_rand.end <= we)].copy()
    w_s_rand["start"] -= ws
    w_s_rand["end"] -= ws
    
    _plot_track(
        ax1, w_seq, 
        slice_rand,      # Pass the sliced data
        w_s_rand, 
        f"Random Model: {motif_name} region ({ws}-{we})",
        custom_ylim=common_ylim  # <--- PASS THE SHARED LIMIT
    )
    
    # --- 4. Plot Genomic Model ---
    w_s_gen = seqlets_gen[(seqlets_gen.start >= ws) & (seqlets_gen.end <= we)].copy()
    w_s_gen["start"] -= ws
    w_s_gen["end"] -= ws
    
    _plot_track(
        ax2, w_seq, 
        slice_gen,       # Pass the sliced data
        w_s_gen, 
        "Genomic Model",
        custom_ylim=common_ylim  # <--- PASS THE SHARED LIMIT
    )
    
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()



def collapse_seqlet_motif_labels(df):
    """
    Collapse multiple motif hits per seqlet to a single best motif.
    Best = lowest motif_e_value (or motif_p_value as fallback).
    """
    if df.empty:
        return df

    group_cols = ["start", "end"]
    if "sign" in df.columns:
        group_cols.append("sign")
    if "strand" in df.columns:
        group_cols.append("strand")

    def pick_best(x):
        if "motif_e_value" in x.columns:
            return x.sort_values("motif_e_value").iloc[0]
        elif "motif_p_value" in x.columns:
            return x.sort_values("motif_p_value").iloc[0]
        else:
            return x.iloc[0]

    out = (
        df.groupby(group_cols, as_index=False)
          .apply(pick_best)
          .reset_index(drop=True)
    )

    return out


def keep_central_seqlet(seqlets, window_len):
    if len(seqlets) == 0:
        return seqlets

    center = window_len / 2
    seqlets = seqlets.copy()
    seqlets["mid"] = (seqlets.start + seqlets.end) / 2
    idx = (seqlets["mid"] - center).abs().idxmin()
    return seqlets.loc[[idx]]


def _kb_formatter(x, pos):
    """
    Format base-pair coordinates as kb with commas.
    Example:
      1234     -> 1 kb
      1234567  -> 1,234 kb
    """
    kb = int(round(x / 1000.0))
    return f"{kb:,} kb"


def _compute_bin_coords_bp(a, b, L, strand, crop_bp, bin_size_bp, n_bins, genome_start_bp=0):
    genome_start_bp = int(genome_start_bp)

    first_bin_view = a + crop_bp
    bin_lefts_view = first_bin_view + np.arange(n_bins) * bin_size_bp
    bin_centers_view = bin_lefts_view + (bin_size_bp / 2.0)

    if strand == "fwd":
        bin_centers_orig = bin_centers_view
    else:
        bin_centers_orig = (L - 1) - bin_centers_view
        bin_centers_orig = bin_centers_orig[::-1]

    return (genome_start_bp + bin_centers_orig).astype(float)

def _plot_cov_track(ax, y, title, x_bp=None, color="#6A5ACD", show_xlabel=False, y_label="Coverage"):
    """
    Filled coverage/profile track like genome browser plots.
    x_bp is in bp (absolute genome coordinates); axis labels show kb.
    """
    if y is None:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    y = np.asarray(y, dtype=float)

    if x_bp is None:
        x_bp = np.arange(len(y), dtype=float)
    else:
        x_bp = np.asarray(x_bp, dtype=float)

    # Filled profile
    ax.fill_between(x_bp, 0, y, color=color, alpha=0.9, linewidth=0)
    ax.plot(x_bp, y, color=color, linewidth=1.0)

    # Clean "track" look
    # ax.set_title(title, fontsize=9.5, loc="left", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # y axis minimal
    ax.tick_params(axis="y", labelsize=9.5)
    ymax = np.nanmax(y)
    ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # x axis as kb
    ax.xaxis.set_major_formatter(FuncFormatter(_kb_formatter))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.set_ylabel(y_label, fontsize=9)

    # <<< add this >>>
    ax.set_xlim(x_bp[0], x_bp[-1])
    ax.margins(x=0)

    if show_xlabel:
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Genomic coordinates", fontsize=10)
        ax.spines["bottom"].set_visible(True)
    else:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)


def _plot_track(ax_logo, window_seq, attr, seqlets, title, custom_ylim=None):
    """
    Logo + seqlet annotations on the same axis.
    """
    Lw = len(window_seq)

    # --- Logo ---
    logo_df = pd.DataFrame(index=range(Lw), columns=BASES, data=0.0)
    for i, base in enumerate(window_seq):
        logo_df.at[i, base] = float(attr[i])

    logo = logomaker.Logo(logo_df, color_scheme="classic", ax=ax_logo)
    logo.style_spines(visible=False)
    ax_logo.spines["left"].set_visible(True)
    ax_logo.set_xticks([])
    # ax_logo.set_title(title, fontsize=10, loc="left", pad=2)
    ax_logo.set_ylabel("Contribution score", fontsize=9)

    # --- Unified y scaling ---
    if custom_ylim is not None:
        ax_logo.set_ylim(custom_ylim)
    else:
        ymin, ymax = ax_logo.get_ylim()
        y_range = max(abs(ymin), abs(ymax))
        ax_logo.set_ylim(-y_range * 1.2, y_range * 1.2)

    ymin, ymax = ax_logo.get_ylim()

    # --- Seqlet highlighting + labels ---
    for _, r in seqlets.iterrows():
        if r.motif_e_value <= MAX_E_VALUE and r.motif_name != "NA":
            s, e = int(r.start), int(r.end)
            sign = int(r.sign) if "sign" in r else 1

            if sign > 0:
                y0 = 0
                height = ymax - 0
                txt_y = ymax * 0.92
                va = "top"
            else:
                y0 = ymin
                height = 0 - ymin
                txt_y = ymin * 0.92
                va = "bottom"

            ax_logo.add_patch(
                Rectangle(
                    (s - 0.5, y0),
                    e - s,
                    height,
                    color="gray",
                    alpha=0.12,
                    linewidth=0
                )
            )

            ax_logo.text(
                (s + e) / 2 - 0.5,
                txt_y,
                r.motif_name,
                ha="center",
                va=va,
                fontsize=9,
                weight="bold"
            )

    return logo


def plot_comparison_with_metrics(
    seq, strand, attr_gen, seqlets_gen, attr_rand, seqlets_rand,
    ws, we, out_dir,
    model_random, model_genomic,
    npz_truth_path, chrom,
    mu_g, sigma_g, mu_r, sigma_r, correlation, genome_start_bp=0
):
    """
    5-row figure:
      1) ExoShorkie attribution logo
      2) NatShorkie attribution logo
      3) ExoShorkie predicted track (filled)
      4) NatShorkie predicted track (filled)
      5) True coverage track (filled)

    X-axes of tracks are absolute genomic coordinates (kb).
    Reverse strand uses L - coord mapping automatically (L = len(seq)).
    """
    ws = int(ws); we = int(we)
    L = len(seq)

    genome_start_bp = int(genome_start_bp)

    def to_abs(i):
        # i is 0-based index into seq
        if strand == "fwd":
            return genome_start_bp + i
        else:
            return genome_start_bp + (L - 1 - i)

    start_abs = min(to_abs(ws), to_abs(we-1))
    end_abs   = max(to_abs(ws), to_abs(we-1)) + 1   # make it half-open

    out_path = out_dir / f"disagreement_win_{start_abs}_{end_abs}_{strand}.png"

    # -----------------------------
    # A) Attribution slices (ws:we)
    w_seq = seq[ws:we]
    strand_idx = 0 if strand == "fwd" else 1

    slice_rand = attr_rand[strand_idx, ws:we]
    slice_gen  = attr_gen[strand_idx, ws:we]

    global_max = max(
        float(np.max(slice_rand)) if len(slice_rand) else 0.0,
        float(np.max(slice_gen))  if len(slice_gen)  else 0.0,
    )
    global_min = min(
        float(np.min(slice_rand)) if len(slice_rand) else 0.0,
        float(np.min(slice_gen))  if len(slice_gen)  else 0.0,
    )

    # Clamp so we keep the semantics: pos >= 0, neg <= 0
    global_max = max(global_max, 0.0)
    global_min = min(global_min, 0.0)

    limit_pos = max(global_max * 1.2, 1e-3)
    limit_neg = min(global_min * 1.2, -1e-3)

    common_ylim = (limit_neg, limit_pos)
    # -----------------------------
    # B) 16384-centered evaluation
    # -----------------------------
    pr = pg = y0 = x_bp = None
    header_txt = "metrics unavailable"

    cov = fetch_coverage(npz_truth_path, chrom)
    cov = cov if strand == "fwd" else cov[::-1]
    L_cov = min(len(seq), len(cov))

    mid = (ws + we) // 2
    a = mid - WINDOW_BP // 2
    b = a + WINDOW_BP

    if a >= 0 and b <= L_cov:
        # Truth (binned)
        y_bins = crop_and_bin_cov(cov[a:b], CROP_BP, BIN_SIZE_BP)

        # Model input
        X = build_shorkie_features(seq[a:b]).astype("float32")[None, :, :]

        pred_g = model_genomic(X, training=False).numpy().squeeze()
        pred_r = model_random(X, training=False).numpy().squeeze()

        prof_g = np.expm1(pred_g * sigma_g + mu_g).astype(np.float64)
        prof_r = np.expm1(pred_r * sigma_r + mu_r).astype(np.float64)

        m = min(len(y_bins), len(prof_g), len(prof_r))
        y0 = y_bins[:m]
        pg = prof_g[:m]
        pr = prof_r[:m]

        # absolute genomic x coords (bp) for each output bin
        x_bp = _compute_bin_coords_bp(
            a=a, b=b, L=L, strand=strand,
            crop_bp=CROP_BP, bin_size_bp=BIN_SIZE_BP, n_bins=m, genome_start_bp=genome_start_bp
        )


        header_txt = (
            r"$\mathrm{NatShorkie}\ \rho=%.2f \;|\; \mathrm{ExoShorkie}\ \rho=%.2f$"
            % (spearmanr(y0, pg).correlation, spearmanr(y0, pr).correlation)
        )

    # -----------------------------
    # C) Seqlets (ONLY this strand)
    # -----------------------------
    w_s_rand = seqlets_rand[
        (seqlets_rand["strand"] == strand) &
        (seqlets_rand.start >= ws) &
        (seqlets_rand.end <= we)
    ].copy()
    w_s_rand = collapse_seqlet_motif_labels(w_s_rand)
    w_s_rand["start"] -= ws
    w_s_rand["end"]   -= ws
    w_s_rand = keep_central_seqlet(w_s_rand, we - ws)

    w_s_gen = seqlets_gen[
        (seqlets_gen["strand"] == strand) &
        (seqlets_gen.start >= ws) &
        (seqlets_gen.end <= we)
    ].copy()
    w_s_gen = collapse_seqlet_motif_labels(w_s_gen)
    w_s_gen["start"] -= ws
    w_s_gen["end"]   -= ws
    w_s_gen = keep_central_seqlet(w_s_gen, we - ws)

    # -----------------------------
    # D) Plot (1 col, 5 rows)
    # -----------------------------
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=False)
    fig.subplots_adjust(top=0.90, hspace=0.25)
    fig.align_ylabels(axes)
    ax1, ax2, ax3, ax4, ax5 = axes

    _plot_track(
        ax1, w_seq, slice_rand, w_s_rand,
        f"ExoShorkie attribution  ({chrom}:{start_abs:,}-{end_abs:,})  strand={strand}  window-corr={correlation:.2f}",
        custom_ylim=common_ylim
    )

    _plot_track(
        ax2, w_seq, slice_gen, w_s_gen,
        f"NatShorkie attribution  ({chrom}:{start_abs:,}-{end_abs:,})  strand={strand}",
        custom_ylim=common_ylim
    )

    # Filled “browser” tracks
    _plot_cov_track(ax3, y0, "True coverage",              x_bp=x_bp, color="#1976D2",y_label="RNA-seq coverage")
    _plot_cov_track(ax4, pr, "ExoShorkie predicted coverage", x_bp=x_bp, color="#7B1FA2",y_label="Predicted coverage")
    _plot_cov_track(ax5, pg, "NatShorkie predicted coverage", x_bp=x_bp, color="#F57C00",y_label="Predicted coverage",show_xlabel=True)


    # Header metrics
    fig.text(0.5, 0.965, header_txt, ha="center", va="top", fontsize=12)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
