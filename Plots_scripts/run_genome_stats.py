#!/usr/bin/env python3
"""
Compute GC content, CpG content, and dinucleotide enrichment for each genome
relative to the yeast genome (chrI–chrXVI). Loads yeast chroms from a multi-FASTA,
chr7 in two parts, and the rest of genome FASTAs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

# Yeast chromosomes (S. cerevisiae)
YEAST_CHROMS = [
    "chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI", "chrVII", "chrVIII",
    "chrIX", "chrX", "chrXI", "chrXII", "chrXIII", "chrXIV", "chrXV", "chrXVI",
]

# Paths: yeast multi-FASTA (chrI–chrXVI), chr7 two parts, rest of genomes
YEAST_FASTA = "Data/Yeast genome/S288C_R64.fa"
CHR7_PART1_FASTA = "Data/Human chr 7/chr7_part1.fa"
CHR7_PART2_FASTA = "Data/Human chr 7/chr7_part2.fa"
REST_GENOME_FASTAS = [
    ("Mpneumo", "Data/M. pneumoniae/Mpneumo.fa"),
    ("Mmmyco", "Data/M. mycoides/Mmmyco.fa"),
    ("dChr", "Data/Data-storage chr/dChr.fa"),
    ("HPRT1", "Data/HPRT1/HPRT1.fa"),
    ("HPRT1R", "Data/HPRT1R/HPRT1R.fa"),
]

# Dinucleotide order (same as typical 4x4 row-major: AA, AC, AG, AT, CA, ..., TT)
BASES = "ACGT"
DINUCLEOTIDES = [a + b for a in BASES for b in BASES]


def load_seq_from_fasta(path: str, chrom: str) -> str:
    """Load a single sequence from FASTA. If chrom is None, use the first (or only) record."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    with open(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            if chrom is None or rec.id == chrom:
                return str(rec.seq).upper()
    if chrom is not None:
        raise KeyError(f"Chromosome '{chrom}' not found in {path}")
    raise ValueError(f"No sequence in {path}")


def load_yeast_genome(path: str) -> str:
    """Load and concatenate all yeast chromosomes chrI–chrXVI from a multi-FASTA."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Yeast FASTA not found: {path}")
    seqs = []
    with open(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            if rec.id in YEAST_CHROMS:
                seqs.append(str(rec.seq).upper())
    if not seqs:
        raise ValueError(f"None of {YEAST_CHROMS} found in {path}")

    print(f"loaded {len(seqs)}/{len(YEAST_CHROMS)} chromosomes from {path}")
    return "".join(seqs)


def gc_content(seq: str) -> float:
    """Fraction of bases that are G or C (excluding N and other)."""
    seq = seq.upper()
    valid = sum(1 for c in seq if c in "ACGT")
    if valid == 0:
        return np.nan
    gc = sum(1 for c in seq if c in "GC")
    return gc / valid


def cpg_oe(seq: str) -> float:
    """CpG observed/expected ratio. O/E = (n_CpG * L) / (n_C * n_G)."""
    seq = seq.upper()
    n_c = seq.count("C")
    n_g = seq.count("G")
    n_cpg = 0
    for i in range(len(seq) - 1):
        if seq[i : i + 2] == "CG":
            n_cpg += 1
    L = len(seq)
    if n_c * n_g == 0:
        return np.nan
    expected = (n_c * n_g) / L
    if expected == 0:
        return np.nan
    return n_cpg / expected


def dinucleotide_frequencies(seq: str) -> np.ndarray:
    """Return length-16 array of dinucleotide frequencies (same order as DINUCLEOTIDES)."""
    seq = seq.upper()
    counts = np.zeros(16, dtype=np.float64)
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a in "ACGT" and b in "ACGT":
            idx = BASES.index(a) * 4 + BASES.index(b)
            counts[idx] += 1
    total = counts.sum()
    if total == 0:
        return np.full(16, np.nan)
    return counts / total


def main():
    yeast_fasta = YEAST_FASTA if isinstance(YEAST_FASTA, str) else YEAST_FASTA
    chr7_p1 = CHR7_PART1_FASTA if isinstance(CHR7_PART1_FASTA, str) else CHR7_PART1_FASTA
    chr7_p2 = CHR7_PART2_FASTA if isinstance(CHR7_PART2_FASTA, str) else CHR7_PART2_FASTA
    rest = [(n, p if isinstance(p, str) else p) for n, p in REST_GENOME_FASTAS]

    # 1) Yeast genome (chrI–chrXVI)
    yeast_seq = load_yeast_genome(str(yeast_fasta))
    yeast_gc = gc_content(yeast_seq)
    yeast_cpg = cpg_oe(yeast_seq)
    yeast_dinuc = dinucleotide_frequencies(yeast_seq)

    rows = []
    # Add yeast row
    row = {
        "genome": "Yeast_genome",
        "length_bp": len(yeast_seq),
        "GC_content": round(yeast_gc, 4),
        "CpG_OE": round(yeast_cpg, 4),
    }
    for i, dinuc in enumerate(DINUCLEOTIDES):
        # row[f"dinuc_{dinuc}"] = round(yeast_dinuc[i], 6)
        row[f"enrich_{dinuc}_vs_yeast"] = 1.0
    rows.append(row)

    # 2) Chr7 (concatenate part1 + part2, report stats on concatenated sequence only)
    seq_p1 = load_seq_from_fasta(str(chr7_p1), chrom='chr7')
    seq_p2 = load_seq_from_fasta(str(chr7_p2), chrom='chr7')
    chr7_seq = seq_p1 + seq_p2
    g = gc_content(chr7_seq)
    c = cpg_oe(chr7_seq)
    d = dinucleotide_frequencies(chr7_seq)
    r = {
        "genome": "chr7",
        "length_bp": len(chr7_seq),
        "GC_content": round(g, 4),
        "CpG_OE": round(c, 4),
    }
    for i, dinuc in enumerate(DINUCLEOTIDES):
        # r[f"dinuc_{dinuc}"] = round(d[i], 6)
        r[f"enrich_{dinuc}_vs_yeast"] = round(d[i] / yeast_dinuc[i], 4) if yeast_dinuc[i] > 0 else np.nan
    rows.append(r)

    # 3) Rest of genome FASTAs
    for name, path in rest:
        seq = load_seq_from_fasta(str(path), chrom=None)
        g = gc_content(seq)
        c = cpg_oe(seq)
        d = dinucleotide_frequencies(seq)
        r = {
            "genome": name,
            "length_bp": len(seq),
            "GC_content": round(g, 4),
            "CpG_OE": round(c, 4),
        }
        for i, dinuc in enumerate(DINUCLEOTIDES):
            # r[f"dinuc_{dinuc}"] = round(d[i], 6)
            r[f"enrich_{dinuc}_vs_yeast"] = round(d[i] / yeast_dinuc[i], 4) if yeast_dinuc[i] > 0 else np.nan
        rows.append(r)

    df = pd.DataFrame(rows)

    out_dir = Path("Results/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "genome_GC_CpG_dinuc_enrichment.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df[["genome", "length_bp", "GC_content", "CpG_OE"]].to_string(index=False))


if __name__ == "__main__":
    main()
