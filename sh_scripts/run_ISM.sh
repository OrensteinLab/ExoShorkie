#!/bin/bash
set -e  # stop on error

#Mpneumo chromosome

# python -u ISM_student.py   --fasta Data/genome/Mpneumo.fa   --chrom Mpneumo  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/Mpneumo.fa   --chrom Mpneumo   --mu 3.5228  --sigma 1.7232 --model Mpneumo

# python -u ISM_student.py   --fasta Data/genome/Mpneumo.fa   --rev  --chrom Mpneumo  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/Mpneumo.fa   --rev  --chrom Mpneumo   --mu 3.5228  --sigma 1.7232 --model Mpneumo

# #chr7 chromosome

# #part 1
# python -u ISM_student.py   --fasta Data/genome/chr7_part1.fa   --chrom chr7  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/chr7_part1.fa   --chrom chr7   --mu 2.0900  --sigma 1.5280 --model chr7

# python -u ISM_student.py   --fasta Data/genome/chr7_part1.fa   --rev  --chrom chr7  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/chr7_part1.fa   --rev  --chrom chr7   --mu 2.0900  --sigma 1.5280 --model chr7

# #part 2
# python -u ISM_student.py   --fasta Data/genome/chr7_part2.fa   --chrom chr7  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/chr7_part2.fa   --chrom chr7   --mu 2.0900  --sigma 1.5280 --model chr7

# python -u ISM_student.py   --fasta Data/genome/chr7_part2.fa   --rev  --chrom chr7  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/chr7_part2.fa   --rev  --chrom chr7   --mu 2.0900  --sigma 1.5280 --model chr7


# #HPRT1 chromosome
# python -u ISM_student.py   --fasta Data/genome/HPRT1.fa   --chrom HPRT1  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/HPRT1.fa   --chrom HPRT1   --mu  4.4673  --sigma 1.8032 --model HPRT1

# python -u ISM_student.py   --fasta Data/genome/HPRT1.fa   --rev  --chrom HPRT1  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/HPRT1.fa   --rev  --chrom HPRT1   --mu 4.4673  --sigma 1.8032 --model HPRT1


# # dChr chromosome
# python -u ISM_student.py   --fasta Data/genome/dChr.fa   --chrom dChr  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/dChr.fa   --chrom dChr  --mu  4.1016  --sigma 1.9481 --model dChr

# python -u ISM_student.py   --fasta Data/genome/dChr.fa   --rev  --chrom dChr  --mu 1.7390   --sigma 1.8168 --model genomic
# python -u ISM_student.py   --fasta Data/genome/dChr.fa   --rev  --chrom dChr   --mu 4.1016  --sigma 1.9481 --model dChr


# #HPRT1R chromosome
# python -u ISM_student.py   --fasta Data/genome/HPRT1R.fa   --chrom HPRT1R  --mu 1.7390   --sigma 1.8168 --model genomic

python -u ISM_student.py   --fasta Data/genome/HPRT1R.fa   --chrom HPRT1R   --mu  4.5184  --sigma 1.7808 --model HPRT1R

python -u ISM_student.py   --fasta Data/genome/HPRT1R.fa   --rev  --chrom HPRT1R  --mu 1.7390   --sigma 1.8168 --model genomic
python -u ISM_student.py   --fasta Data/genome/HPRT1R.fa   --rev  --chrom HPRT1R   --mu 4.5184  --sigma 1.7808 --model HPRT1R

# Mmmyco chromosome
python -u ISM_student.py   --fasta Data/genome/Mmmyco.fa   --chrom Mmmyco  --mu 1.7390   --sigma 1.8168 --model genomic
python -u ISM_student.py   --fasta Data/genome/Mmmyco.fa   --chrom Mmmyco   --mu  1.2864   --sigma 1.5034 --model Mmmyco

python -u ISM_student.py   --fasta Data/genome/Mmmyco.fa   --rev  --chrom Mmmyco  --mu 1.7390   --sigma 1.8168 --model genomic
python -u ISM_student.py   --fasta Data/genome/Mmmyco.fa   --rev  --chrom Mmmyco   --mu 1.2864   --sigma 1.5034 --model Mmmyco

