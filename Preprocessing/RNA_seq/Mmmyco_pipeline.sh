#!/bin/bash
set -euo pipefail

SRR_LIST=("SRR22130096" "SRR22130095" "SRR22130094")

TMP_DIR="tmp"
STAR_INDEX_DIR="${TMP_DIR}/STAR_index"
OUTPUT_DIR="${TMP_DIR}/output"
GENOME_FA="Mmmyco_reference.fa"


mkdir -p "$TMP_DIR" "$STAR_INDEX_DIR" "$OUTPUT_DIR"

# === STAR genome indexing ===
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

    fasterq-dump "$SRR" --threads 32 -O "$FASTQ_DIR"

    java -jar ../Trimmomatic-0.36/trimmomatic-0.36.jar PE -phred33 \
        "$FASTQ_DIR/${SRR}_1.fastq" "$FASTQ_DIR/${SRR}_2.fastq" \
        "$FASTQ_DIR/trimmed_1_paired.fastq" "$FASTQ_DIR/trimmed_1_unpaired.fastq" \
        "$FASTQ_DIR/trimmed_2_paired.fastq" "$FASTQ_DIR/trimmed_2_unpaired.fastq" \
        ILLUMINACLIP:../Trimmomatic-0.36/adapters/TruSeq3-PE-2.fa:2:30:10:2:true \
        SLIDINGWINDOW:4:10 MINLEN:36

    echo "Running FastQC for $SRR..."
    fastqc "$FASTQ_DIR/trimmed_1_paired.fastq" "$FASTQ_DIR/trimmed_2_paired.fastq" \
           -o "$QC_DIR" -t 8

    STAR --genomeDir "$STAR_INDEX_DIR" \
         --readFilesIn "$FASTQ_DIR/trimmed_1_paired.fastq" "$FASTQ_DIR/trimmed_2_paired.fastq" \
         --runThreadN 32 \
         --twopassMode Basic \
         --outSAMtype BAM SortedByCoordinate \
         --outFilterType BySJout \
         --alignIntronMax 5000 \
         --outFileNamePrefix "$OUT_PREFIX" \
         --limitBAMsortRAM 2000000000

    # === CLEANUP: delete FASTQs ===
    echo "Cleaning up FASTQs for $SRR..."
    rm -f "$FASTQ_DIR"/*.fastq

    # remove STAR tmp folder if present
    rm -rf "${OUT_PREFIX}_STARtmp" 2>/dev/null || true

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

# === CLEANUP: delete replicate BAMs ===
echo "Cleaning up replicate BAMs..."
for SRR in "${SRR_LIST[@]}"; do
    rm -f "${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam"
    rm -f "${TMP_DIR}/${SRR}/Aligned.Aligned.sortedByCoord.out.bam.bai" 2>/dev/null || true
done

samtools index "${TMP_DIR}/merged.bam"

# === BigWigs ===
bamCoverage -b "${TMP_DIR}/merged.bam" \
    -o "${OUTPUT_DIR}/Mmmyco_merged_fwd_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand forward \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

bamCoverage -b "${TMP_DIR}/merged.bam" \
    -o "${OUTPUT_DIR}/Mmmyco_merged_rev_cpm.bw" \
    --outFileFormat bigwig \
    --filterRNAstrand reverse \
    --binSize 1 \
    --normalizeUsing CPM \
    --numberOfProcessors 16 \
    --blackListFileName rDNA_mask.bed

# === CLEANUP: delete merged BAM ===
echo "Cleaning up merged BAM..."
rm -f "${TMP_DIR}/merged.bam" "${TMP_DIR}/merged.bam.bai"

echo "Done. BigWigs are in: ${OUTPUT_DIR}"