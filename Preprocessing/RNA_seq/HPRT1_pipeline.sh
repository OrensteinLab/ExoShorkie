#!/bin/bash
set -euo pipefail

SRR_LIST=("SRR27411960" "SRR27411959")

TMP_DIR="tmp"
STAR_INDEX_DIR="${TMP_DIR}/STAR_index"
OUTPUT_DIR="${TMP_DIR}/output"
GENOME_FA="HPRT1_reference.fa"

# Set to 1 to delete FastQC output folders too
DELETE_FASTQC=0

mkdir -p "$TMP_DIR" "$STAR_INDEX_DIR" "$OUTPUT_DIR"

# === STAR genome indexing (only once) ===
STAR --runMode genomeGenerate \
     --genomeDir "$STAR_INDEX_DIR" \
     --genomeFastaFiles "$GENOME_FA" \
     --genomeSAindexNbases 10 \
     --runThreadN 32

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

    # === FastQC ===
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

    BAM_FILE="${OUT_PREFIX}Aligned.sortedByCoord.out.bam"
    samtools index "$BAM_FILE"
    samtools idxstats "$BAM_FILE"
    echo ""

    # === CLEANUP PART 1: delete FASTQs (raw + trimmed) ===
    echo "Cleaning up FASTQs for $SRR..."
    rm -f "$FASTQ_DIR"/*.fastq

    # Remove STAR temp folder if created
    rm -rf "${OUT_PREFIX}_STARtmp" 2>/dev/null || true

    # Optional: delete FastQC output
    if [[ "$DELETE_FASTQC" -eq 1 ]]; then
        rm -rf "$QC_DIR"
    fi
done

# === Merge BAMs ===
MERGED_BAM_LIST=""
for SRR in "${SRR_LIST[@]}"; do
    MERGED_BAM_LIST+="I=${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam "
done

picard MergeSamFiles $MERGED_BAM_LIST \
       O="${TMP_DIR}/merged.bam" \
       USE_THREADING=true \
       SORT_ORDER=coordinate

# === CLEANUP PART 2: delete replicate BAMs now that merged exists ===
echo "Cleaning up replicate BAMs..."
for SRR in "${SRR_LIST[@]}"; do
    rm -f "${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam"
    rm -f "${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam.bai" 2>/dev/null || true
done

# === Index merged BAM ===
samtools index "${TMP_DIR}/merged.bam"
samtools idxstats "${TMP_DIR}/merged.bam"

# === BigWigs ===
bamCoverage -b "${TMP_DIR}/merged.bam" \
    -o "${OUTPUT_DIR}/HPRT1_merged_fwd_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand forward \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

bamCoverage -b "${TMP_DIR}/merged.bam" \
    -o "${OUTPUT_DIR}/HPRT1_merged_rev_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand reverse \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

# === CLEANUP PART 3: delete merged BAM ===
echo "Cleaning up merged BAM..."
rm -f "${TMP_DIR}/merged.bam" "${TMP_DIR}/merged.bam.bai"

echo "Done. BigWigs are in: ${OUTPUT_DIR}"