#!/bin/bash
set -euo pipefail

# Define variables
SRR="SRR25474920"
TMP_DIR="tmp2"
FASTQ_DIR="${TMP_DIR}/Fastq"
STAR_INDEX_DIR="${TMP_DIR}/STAR_index"
METRICS_DIR="${TMP_DIR}/metrics"
OUTPUT_DIR="${TMP_DIR}/output"

# Create directories
mkdir -p "$TMP_DIR" "$FASTQ_DIR" "$STAR_INDEX_DIR" "$METRICS_DIR" "$OUTPUT_DIR"

# === Step 1: Download (uncomment if needed) ===
fasterq-dump "$SRR" --threads 32 -O "$FASTQ_DIR"

# === Step 2: Trimming paired-end reads ===
java -jar ../Trimmomatic-0.36/trimmomatic-0.36.jar PE -phred33 \
  "$FASTQ_DIR/${SRR}_1.fastq" "$FASTQ_DIR/${SRR}_2.fastq" \
  "$FASTQ_DIR/trimmed_${SRR}_1_paired.fastq" "$FASTQ_DIR/trimmed_${SRR}_1_unpaired.fastq" \
  "$FASTQ_DIR/trimmed_${SRR}_2_paired.fastq" "$FASTQ_DIR/trimmed_${SRR}_2_unpaired.fastq" \
  ILLUMINACLIP:../Trimmomatic-0.36/adapters/TruSeq3-PE-2.fa:2:30:10:2:true \
  SLIDINGWINDOW:4:10 MINLEN:36

# FastQC Quality Check ===
echo "Running FastQC for $SRR..."
QC_DIR="${TMP_DIR}/${SRR}/QC"
mkdir -p "$QC_DIR"

# Corrected FastQC line
fastqc "$FASTQ_DIR/trimmed_${SRR}_1_paired.fastq" "$FASTQ_DIR/trimmed_${SRR}_2_paired.fastq" \
        -o "$QC_DIR" -t 8

# === Step 3: STAR indexing ===
STAR --runMode genomeGenerate \
     --genomeDir "$STAR_INDEX_DIR" \
     --genomeFastaFiles YAC481_reference.fa \
     --genomeSAindexNbases 10 \
     --runThreadN 32

# === Step 4: STAR alignment ===
STAR --genomeDir "$STAR_INDEX_DIR" \
     --readFilesIn "$FASTQ_DIR/trimmed_${SRR}_1_paired.fastq" "$FASTQ_DIR/trimmed_${SRR}_2_paired.fastq" \
     --runThreadN 32 \
     --twopassMode Basic \
     --outSAMtype BAM SortedByCoordinate \
     --outFilterType BySJout \
     --alignIntronMax 5000 \
     --outFileNamePrefix "${TMP_DIR}/Aligned." \
     --limitBAMsortRAM 2000000000 

# === CLEANUP PART 1: Remove FASTQ files ===
# Alignment is complete. We delete the massive FASTQ files now.
echo "Cleaning up FASTQ files..."
rm -f "$FASTQ_DIR"/*.fastq

# === Step 6: Index the BAM ===
samtools index "${TMP_DIR}/Aligned.Aligned.sortedByCoord.out.bam"

# === Step 7: Generate BigWig files for forward strand ===
bamCoverage -b "${TMP_DIR}/Aligned.Aligned.sortedByCoord.out.bam" \
    -o "${OUTPUT_DIR}/part2_fwd_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand forward \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

# === Step 8: Generate BigWig files for reverse strand ===
bamCoverage -b "${TMP_DIR}/Aligned.Aligned.sortedByCoord.out.bam" \
    -o "${OUTPUT_DIR}/part2_rev_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand reverse \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

# === CLEANUP PART 2: Remove BAM files ===
# BigWigs are generated. We delete the massive BAM files now.
echo "Cleaning up BAM files..."
rm -f "${TMP_DIR}/Aligned.Aligned.sortedByCoord.out.bam"
rm -f "${TMP_DIR}/Aligned.Aligned.sortedByCoord.out.bam.bai"

# Optional: Remove STAR temporary directory
rm -rf "${TMP_DIR}/Aligned._STARtmp"

echo "Pipeline finished successfully. Intermediate files deleted."