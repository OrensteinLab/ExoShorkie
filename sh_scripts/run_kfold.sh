#!/bin/bash
set -e  # stop on error

echo "=== Running Mpneumo fine-tune ==="
python k_fold_cross.py \
    --src1-name Mpneumo \
    --src1-chrom Mpneumo \
    --src1-npz-fwd Data/normalized_expression/Mpneumo_fwd_norm.npz \
    --src1-npz-rev Data/normalized_expression/Mpneumo_rev_norm.npz \
    --src1-fasta Data/genome/S288c_Mpneumo.fa 

echo "=== Finished Mpneumo ==="

echo "=== Running Human chr7 fine-tune ==="
python k_fold_cross.py \
    --src1-name GSM6726666 \
    --src1-chrom chr7 \
    --src1-npz-fwd Data/normalized_expression/Human_part1_fwd_norm.npz \
    --src1-npz-rev Data/normalized_expression/Human_part1_rev_norm.npz \
    --src1-fasta Data/genome/chr7_part1.fa \
    --src2-name GSM7671181_481 \
    --src2-chrom chr7 \
    --src2-npz-fwd Data/normalized_expression/Human_part2_fwd_norm.npz \
    --src2-npz-rev Data/normalized_expression/Human_part2_rev_norm.npz \
    --src2-fasta Data/genome/chr7_part2.fa

echo "=== Finished Human ==="

echo "=== Running Mmmyco fine-tune ==="
python k_fold_cross.py \
    --src1-name Mmmyco \
    --src1-chrom Mmmyco \
    --src1-npz-fwd Data/normalized_expression/Mmmyco_fwd_norm.npz \
    --src1-npz-rev Data/normalized_expression/Mmmyco_rev_norm.npz \
    --src1-fasta Data/genome/W303_Mmmyco.fa

echo "=== Finished Mmmyco ==="


echo "=== Running HPRT1 fine-tune ==="
python k_fold_cross.py \
    --src1-name HPRT1 \
    --src1-chrom HPRT1 \
    --src1-npz-fwd Data/normalized_expression/HPRT1_fwd_norm.npz \
    --src1-npz-rev Data/normalized_expression/HPRT1_rev_norm.npz \
    --src1-fasta Data/genome/HPRT1.fa

echo "=== Finished HPRT1 ==="

echo "=== Running HPRT1R fine-tune ==="
python k_fold_cross.py \
    --src1-name HPRT1R \
    --src1-chrom HPRT1R \
    --src1-npz-fwd Data/normalized_expression/HPRT1R_fwd_norm.npz \
    --src1-npz-rev Data/normalized_expression/HPRT1R_rev_norm.npz \
    --src1-fasta Data/genome/HPRT1R.fa

echo "=== Finished HPRT1R ==="


echo "=== Running dChr fine-tune ==="
python k_fold_cross.py \
    --src1-name dChr \
    --src1-chrom dChr \
    --src1-npz-fwd Data/normalized_expression/dChr_fwd_norm.npz \
    --src1-npz-rev Data/normalized_expression/dChr_rev_norm.npz \
    --src1-fasta Data/genome/dChr.fa

echo "=== Finished dChr ==="


echo "=== ALL DONE ==="