#!/usr/bin/env python3
"""
Create a publication-style heatmap of dinucleotide enrichment (vs yeast) from
Results/ISMB_results/Genome_stats/genome_GC_CpG_dinuc_enrichment.csv.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# Publication-friendly defaults
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12

# Dinucleotide order matching the CSV columns
DINUCLEOTIDES = [
    "AA", "AC", "AG", "AT",
    "CA", "CC", "CG", "CT",
    "GA", "GC", "GG", "GT",
    "TA", "TC", "TG", "TT"
]

# Paper-style genome labels
GENOME_DISPLAY_NAMES = {
    "Yeast_genome": "Yeast genome",
    "chr7": "Human chr. 7",
    "Mpneumo": "M. pneumoniae",
    "Mmmyco": "M. mycoides",
    "HPRT1": "HPRT1",
    "HPRT1R": "HPRT1R",
    "dChr": "Data storage chr.",
}

EXTRA_COL_1_NAME = "% GC"
EXTRA_COL_2_NAME = "CpG (obs/exp)"

EXTRA_COL_1 = {
    "Yeast_genome": 38.30,
    "chr7": 36.85,
    "Mpneumo": 40.01,
    "Mmmyco": 24.16,
    "HPRT1": 42.13,
    "HPRT1R": 42.13,
    "dChr": 49.70,
}

EXTRA_COL_2 = {
    "Yeast_genome": 0.8004,
    "chr7": 0.2054,
    "Mpneumo": 0.8177,
    "Mmmyco": 0.3623,
    "HPRT1": 0.3807,
    "HPRT1R": 1.0327,
    "dChr": 0.9977,
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot dinucleotide enrichment heatmap for paper")
    p.add_argument(
        "--csv",
        type=str,
        default="Results/genome_GC_CpG_dinuc_enrichment.csv",
        help="Path to enrichment CSV",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="Results/plots",
        help="Output directory (default: same as CSV directory)",
    )
    p.add_argument(
        "--exclude-yeast",
        action="store_true",
        help="Exclude Yeast_genome row (reference, all 1.0)",
    )
    p.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["png"],
        help="Output formats (e.g. pdf png)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs (default 300)",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate enrichment cells with values",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base = Path(__file__).resolve().parent.parent
    csv_path = base / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    enrich_cols = [f"enrich_{d}_vs_yeast" for d in DINUCLEOTIDES if f"enrich_{d}_vs_yeast" in df.columns]
    if len(enrich_cols) != 16:
        enrich_cols = sorted([c for c in df.columns if c.startswith("enrich_") and c.endswith("_vs_yeast")])

    plot_df = df[["genome"] + enrich_cols].copy()
    plot_df = plot_df.set_index("genome")
    plot_df.columns = [c.replace("enrich_", "").replace("_vs_yeast", "") for c in plot_df.columns]

    if args.exclude_yeast and "Yeast_genome" in plot_df.index:
        plot_df = plot_df.drop(index="Yeast_genome")

    y_labels = [GENOME_DISPLAY_NAMES.get(g, g) for g in plot_df.index]
    n_rows = len(plot_df.index)
    col_labels = list(plot_df.columns)

    data = plot_df.values.astype(float)
    data = np.ma.masked_invalid(data)
    vmin = max(0.2, np.nanmin(data))
    vmax = min(2.5, np.nanmax(data))
    norm_enrich = Normalize(vmin=vmin, vmax=vmax)

    gc_vals = np.array([EXTRA_COL_1.get(g, np.nan) for g in plot_df.index], dtype=float).reshape(n_rows, 1)
    cpg_vals = np.array([EXTRA_COL_2.get(g, np.nan) for g in plot_df.index], dtype=float).reshape(n_rows, 1)

    norm_gc = Normalize(vmin=19, vmax=51)
    cpg_min = max(0.2, np.nanmin(cpg_vals))
    cpg_max = min(2.5, np.nanmax(cpg_vals)) if np.any(np.isfinite(cpg_vals)) else 1.2
    norm_cpg = Normalize(vmin=cpg_min, vmax=cpg_max)

    # Three different palettes (truncated for text readability)
    def _truncate_cmap(name, stop=0.84):
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(f"{name}_light", base(np.linspace(0, stop, 256)))
    cmap_gc = _truncate_cmap("Blues")       # Blues for % GC
    cmap_cpg = _truncate_cmap("Purples")    # Purples for CpG (obs/exp)
    cmap_enrich = _truncate_cmap("YlOrRd")  # original yellow-orange-red for Enrichment vs. yeast

    # Figure layout: one row of 3 colorbars (GC | CpG | Enrichment) on top, data panels below
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.22, 4], hspace=0.2)

    # Top row: 3 colorbars side by side (GC left, CpG middle, Enrichment right)
    gs_cbars = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.18)
    ax_cbar_gc = fig.add_subplot(gs_cbars[0])
    ax_cbar_cpg = fig.add_subplot(gs_cbars[1])
    ax_cbar_enrich = fig.add_subplot(gs_cbars[2])

    gs_panels = gs[1].subgridspec(1, 3, width_ratios=[0.75, 0.75, 16], wspace=0.08)
    ax_gc = fig.add_subplot(gs_panels[0])
    ax_cpg = fig.add_subplot(gs_panels[1], sharey=ax_gc)
    ax_main = fig.add_subplot(gs_panels[2], sharey=ax_gc)

    # GC panel
    ax_gc.imshow(
        gc_vals,
        aspect="auto",
        cmap=cmap_gc,
        norm=norm_gc,
        interpolation="nearest",
        extent=(0, 1, n_rows - 0.5, -0.5),
    )
    for i in range(n_rows):
        v = gc_vals[i, 0]
        s = f"{v:.1f}" if np.isfinite(v) else "—"
        ax_gc.text(0.5, i, s, ha="center", va="center", color="black", fontsize=10)

    ax_gc.set_xticks([0.5])
    ax_gc.set_xticklabels([EXTRA_COL_1_NAME], fontsize=9)
    plt.setp(ax_gc.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax_gc.set_ylabel("Genome", fontsize=11)
    ax_gc.set_yticks(np.arange(n_rows))
    ax_gc.set_yticklabels(y_labels, fontsize=10)

    # CpG panel
    ax_cpg.imshow(
        cpg_vals,
        aspect="auto",
        cmap=cmap_cpg,
        norm=norm_cpg,
        interpolation="nearest",
        extent=(0, 1, n_rows - 0.5, -0.5),
    )
    for i in range(n_rows):
        v = cpg_vals[i, 0]
        s = f"{v:.2f}" if np.isfinite(v) else "—"
        ax_cpg.text(0.5, i, s, ha="center", va="center", color="black", fontsize=10)

    ax_cpg.set_xticks([0.5])
    ax_cpg.set_xticklabels([EXTRA_COL_2_NAME], fontsize=9)
    plt.setp(ax_cpg.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax_cpg.get_yticklabels(), visible=False)
    ax_cpg.tick_params(left=False)

    # Main enrichment heatmap
    im_main = ax_main.imshow(
        data,
        aspect="auto",
        cmap=cmap_enrich,
        norm=norm_enrich,
        interpolation="nearest",
        extent=(-0.5, 15.5, n_rows - 0.5, -0.5),
    )

    ax_main.set_xticks(np.arange(16))
    ax_main.set_xticklabels(col_labels, fontsize=9)
    ax_main.set_xlabel("Dinucleotide", fontsize=11)
    plt.setp(ax_main.get_yticklabels(), visible=False)
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax_main.tick_params(left=False)

    if args.annotate:
        for i in range(n_rows):
            for j in range(16):
                val = data[i, j]
                if np.ma.is_masked(val):
                    continue
                ax_main.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=6
                )

    # Top horizontal colorbars: slightly thicker, much shorter (shrink=width fraction, aspect=length/height)
    cbar_enrich = fig.colorbar(
        im_main, cax=ax_cbar_enrich, orientation="horizontal", shrink=0.30, aspect=5
    )
    cbar_enrich.set_label("Enrichment vs. yeast", fontsize=10, labelpad=1)
    cbar_enrich.ax.tick_params(labelsize=8, length=2, pad=1)

    cbar_cpg = fig.colorbar(
        ScalarMappable(norm=norm_cpg, cmap=cmap_cpg),
        cax=ax_cbar_cpg,
        orientation="horizontal",
        shrink=0.30,
        aspect=5,
    )
    cbar_cpg.set_label("CpG (obs/exp)", fontsize=10, labelpad=1)
    cbar_cpg.ax.tick_params(labelsize=8, length=2, pad=1)

    cbar_gc = fig.colorbar(
        ScalarMappable(norm=norm_gc, cmap=cmap_gc),
        cax=ax_cbar_gc,
        orientation="horizontal",
        shrink=0.30,
        aspect=5,
    )
    cbar_gc.set_label("% GC", fontsize=10, labelpad=1)
    cbar_gc.ax.tick_params(labelsize=8, length=2, pad=1)
    # Slight margin so edge labels "20" and "50" don't extend beyond the bar
    cbar_gc.ax.set_xlim(19, 51)

    # fig.suptitle("Dinucleotide enrichment relative to yeast genome", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96], pad=0.3)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92)

    stem = "dinucleotide_enrichment_heatmap"
    if args.exclude_yeast:
        stem += "_no_yeast"

    for fmt in args.format:
        out_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out_path}")

    plt.close()


if __name__ == "__main__":
    main()