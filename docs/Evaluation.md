# Evaluations

## 5-fold Cross-validation ensemble evaluation

This script evaluates a **5-fold cross-validation ensemble** of ExoShorkie fine-tuned models on strand-resolved RNA-seq coverage.

It loads pretrained ensemble members from:

```text
Models/<chrom>/cv<fold>/f<member>/model_finetune.h5
```

and computes window-level **Spearman correlation** between predicted and measured coverage.

---

### Program arguments (`scripts/predict_kfold.py`)

**Primary source (required)**

- **`--src1-chrom`** — Genome or chromosome identifier for source 1.
- **`--src1-fasta`** — FASTA path for source 1.
- **`--src1-npz-fwd`**, **`--src1-npz-rev`** — Forward- and reverse-strand coverage `.npz` files for source 1.

**Second source (optional)** — If any `--src2-*` argument is used, provide the full set below so evaluation can run on two genomes

- **`--src2-chrom`**, **`--src2-fasta`**, **`--src2-npz-fwd`**, **`--src2-npz-rev`**

**Other**

- **`--ensemble`** — Number of ensemble members to load per fold (default: 8).

---

### Model directory structure

Expected layout:

```text
Models/
└── Mpneumo/
    ├── cv0/
    │   ├── f0/model_finetune.h5
    │   ├── ...
    │   └── f7/model_finetune.h5
    ├── cv1/
    ├── ...
    └── cv4/
```

---

### Example usage

```bash
python scripts/predict_kfold.py \
  --src1-chrom Mpneumo \
  --src1-fasta Data/genome/Mpneumo.fa \
  --src1-npz-fwd Data/normalized_expression/Mpneumo_fwd_norm.npz \
  --src1-npz-rev Data/normalized_expression/Mpneumo_rev_norm.npz \
  --ensemble 8
```

---

### Output

A CSV file is written containing per-fold metrics:

```text
cv_fold,median_spearman,mean_spearman
0,0.52,0.50
1,0.55,0.53
...
Overall_Mean,0.54,0.52
Overall_Std,0.02,0.01
```

It is saved in Results/Cross_validation/results_<chrom_name>.csv

### Metrics

For each fold:

- **Spearman correlation** is computed per window
- Median and mean are reported

Overall summary:

- Mean ± std across folds

## Leave-One-Genome-Out Evaluation

This script evaluates **cross-genome generalization** by predicting RNA-seq coverage on one held-out exogenous genome using an ensemble of models trained on **all other exogenous genomes**.  

---

### Expected data structure (Figshare)

After downloading the processed datasets from Figshare, the expected layout is:

```text
Data/
├── M. pneumoniae/
│   ├── Mpneumo.fa
│   ├── Mpneumo_fwd_norm.npz
│   └── Mpneumo_rev_norm.npz
├── M. mycoides/
│   ├── Mmmyco.fa
│   ├── Mmmyco_fwd_norm.npz
│   └── Mmmyco_rev_norm.npz
├── HPRT1/
│   ├── HPRT1.fa
│   ├── HPRT1_fwd_norm.npz
│   └── HPRT1_rev_norm.npz
├── HPRT1R/
│   ├── HPRT1R.fa
│   ├── HPRT1R_fwd_norm.npz
│   └── HPRT1R_rev_norm.npz
├── Data-storage chr/
│   ├── dChr.fa
│   ├── dChr_fwd_norm.npz
│   └── dChr_rev_norm.npz
└── Human chr 7/
    ├── chr7_part1.fa
    ├── chr7_part2.fa
    ├── Human_part1_fwd_norm.npz
    ├── Human_part1_rev_norm.npz
    ├── Human_part2_fwd_norm.npz
    └── Human_part2_rev_norm.npz
```

Each dataset includes:
- Genome FASTA
- Forward-strand RNA-seq coverage (`*_fwd_norm.npz`)
- Reverse-strand RNA-seq coverage (`*_rev_norm.npz`)

---

### Expected model directory structure

The script expects fine-tuned ExoShorkie models organized exactly as downloaded from Hugging Face.

Each dataset directory contains **5 cross-validation folds** (`cv0–cv4`), and each fold contains **8 ensemble members** (`f0–f7`):

```text
Models/
├── Data_storage_chr/
│   ├── cv0/
│   │   ├── f0/model_finetune.h5
│   │   ├── f1/model_finetune.h5
│   │   ├── ...
│   │   └── f7/model_finetune.h5
│   ├── cv1/
│   │   └── ...
│   └── cv4/
│       └── ...
├── HPRT1/
├── HPRT1R/
├── Human_chr_7/
├── M_mycoides/
├── M_pneumoniae/
```

All available CV folds (`cv0–cv4`) and ensemble members (`f0–f7`) are used, except those trained on the held-out genome.

---

### Running the script

```bash
python scripts/leave_one_genome_out.py [--ablation {full|human|bacteria|syn}]
```

`--ablation` (default `full`): `full` = leave-one-out over all genomes; `human`, `bacteria`, `syn` = ensemble restricted to the specific group of genomes

Results and window correlations are written to:

```text
Results/Correlations/correlations_<genome>_<ablation>_stride<bp>.npz
Results/leave_one_genome_out_results_<ablation>.csv
```
