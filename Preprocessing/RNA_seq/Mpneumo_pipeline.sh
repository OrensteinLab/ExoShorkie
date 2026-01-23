#!/bin/bash
set -euo pipefail

# === Define replicate accessions ===
SRR_LIST=("SRR22130093" "SRR22130092" "SRR22130091")

# === Define directories ===
TMP_DIR="tmp"
STAR_INDEX_DIR="${TMP_DIR}/STAR_index"
OUTPUT_DIR="${TMP_DIR}/output"
GENOME_FA="Mpneumo_reference.fa"

# If you want to keep FastQC reports, set to 0
DELETE_FASTQC=0

mkdir -p "$TMP_DIR" "$STAR_INDEX_DIR" "$OUTPUT_DIR"

# === Step 1: STAR genome indexing (only once) ===
STAR --runMode genomeGenerate \
     --genomeDir "$STAR_INDEX_DIR" \
     --genomeFastaFiles "$GENOME_FA" \
     --genomeSAindexNbases 10 \
     --runThreadN 32

# === Step 2–4: Process each replicate ===
for SRR in "${SRR_LIST[@]}"; do
    echo "Processing $SRR"
    FASTQ_DIR="${TMP_DIR}/${SRR}/Fastq"
    QC_DIR="${TMP_DIR}/${SRR}/QC"
    OUT_PREFIX="${TMP_DIR}/${SRR}/Aligned."
    mkdir -p "$FASTQ_DIR" "$QC_DIR"

    # === Download FASTQ files ===
    fasterq-dump "$SRR" --threads 32 -O "$FASTQ_DIR"

    # === Trimming paired-end reads ===
    java -jar ../Trimmomatic-0.36/trimmomatic-0.36.jar PE -phred33 \
        "$FASTQ_DIR/${SRR}_1.fastq" "$FASTQ_DIR/${SRR}_2.fastq" \
        "$FASTQ_DIR/trimmed_1_paired.fastq" "$FASTQ_DIR/trimmed_1_unpaired.fastq" \
        "$FASTQ_DIR/trimmed_2_paired.fastq" "$FASTQ_DIR/trimmed_2_unpaired.fastq" \
        ILLUMINACLIP:../Trimmomatic-0.36/adapters/TruSeq3-PE-2.fa:2:30:10:2:true \
        SLIDINGWINDOW:4:10 MINLEN:36

    # === FastQC Quality Check ===
    echo "Running FastQC for $SRR..."
    fastqc "$FASTQ_DIR/trimmed_1_paired.fastq" "$FASTQ_DIR/trimmed_2_paired.fastq" \
           -o "$QC_DIR" -t 8

    # === STAR alignment ===
    STAR --genomeDir "$STAR_INDEX_DIR" \
         --readFilesIn "$FASTQ_DIR/trimmed_1_paired.fastq" "$FASTQ_DIR/trimmed_2_paired.fastq" \
         --runThreadN 32 \
         --twopassMode Basic \
         --outSAMtype BAM SortedByCoordinate \
         --outFilterType BySJout \
         --alignIntronMax 5000 \
         --outFileNamePrefix "$OUT_PREFIX" \
         --limitBAMsortRAM 2000000000

    # === CLEANUP PART 1: remove FASTQs for this replicate ===
    echo "Cleaning up FASTQ files for $SRR..."
    rm -f "$FASTQ_DIR"/*.fastq

    # STAR sometimes leaves a temp folder; safe to remove after BAM exists
    rm -rf "${OUT_PREFIX}_STARtmp" 2>/dev/null || true

    # Optional: remove FastQC outputs to save more space
    if [[ "$DELETE_FASTQC" -eq 1 ]]; then
        echo "Deleting FastQC outputs for $SRR..."
        rm -rf "$QC_DIR"
    fi
done

# === Step 5: Merge BAMs from all replicates ===
MERGED_BAM_LIST=""
for SRR in "${SRR_LIST[@]}"; do
    MERGED_BAM_LIST+="I=${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam "
done

picard MergeSamFiles $MERGED_BAM_LIST \
       O="${TMP_DIR}/merged.bam" \
       USE_THREADING=true \
       SORT_ORDER=coordinate

# === CLEANUP PART 2: remove individual BAMs now that merged exists ===
echo "Cleaning up individual replicate BAMs..."
for SRR in "${SRR_LIST[@]}"; do
    rm -f "${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam"
    rm -f "${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam.bai" 2>/dev/null || true
done

# === Step 6: Index the merged BAM file ===
samtools index "${TMP_DIR}/merged.bam"

# === Step 7–8: Generate strand-specific BigWigs ===
bamCoverage -b "${TMP_DIR}/merged.bam" \
    -o "${OUTPUT_DIR}/Mpneumo_merged_fwd_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand forward \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

bamCoverage -b "${TMP_DIR}/merged.bam" \
    -o "${OUTPUT_DIR}/Mpneumo_merged_rev_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand reverse \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

# === CLEANUP PART 3: remove merged BAM ===
echo "Cleaning up merged BAM..."
rm -f "${TMP_DIR}/merged.bam" "${TMP_DIR}/merged.bam.bai"

echo "Pipeline finished successfully. BigWigs saved in: ${OUTPUT_DIR}"