#!/bin/bash
set -e  # stop on error

python create_distillation_set.py \
    --genomic \
    --src1-chrom genomic \
    --src1-fasta Data/genome/S288c_Mpneumo.fa \

python train_student.py \
    --chrom genomic


echo "=== Running Mpneumo   ==="
python create_distillation_set.py \
    --src1-chrom Mpneumo \
    --src1-fasta Data/genome/Mpneumo.fa \

python train_student.py \
    --chrom Mpneumo

echo "=== Running Human   ==="

python create_distillation_set.py \
    --src1-chrom chr7 \
    --src1-fasta Data/genome/chr7_part1.fa \
    --src2-chrom chr7 \
    --src2-fasta Data/genome/chr7_part2.fa

python train_student.py \
    --chrom chr7 

echo "=== Finished Human ==="

echo "=== Running dChr ==="
python create_distillation_set.py \
    --src1-chrom dChr \
    --src1-fasta Data/genome/dChr.fa

python train_student.py \
    --chrom dChr

echo "=== Finished dChr ==="

echo "=== Running HPRT1 ==="

python create_distillation_set.py \
    --src1-chrom HPRT1 \
    --src1-fasta Data/genome/HPRT1.fa

python train_student.py \
    --chrom HPRT1

echo "=== Finished HPRT1 ==="

echo "=== ALL DONE ==="

echo "=== Running HPRT1R ==="
python create_distillation_set.py \
    --src1-chrom HPRT1R \
    --src1-fasta Data/genome/HPRT1R.fa

python train_student.py \
    --chrom HPRT1R

echo "=== Running Mmmyco ==="
python create_distillation_set.py \
    --src1-chrom Mmmyco \
    --src1-fasta Data/genome/Mmmyco.fa

python train_student.py \
    --chrom Mmmyco


echo "=== Finished Mmmyco ==="
