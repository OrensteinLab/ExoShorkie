# Cross-Validation Ensemble Evaluation

This script evaluates a **5-fold cross-validation ensemble** of ExoShorkie fine-tuned models on strand-resolved RNA-seq coverage.

It loads pretrained ensemble members from:

```text
Models/<chrom>/cv<fold>/f<member>/model_finetune.h5
```

and computes window-level **Spearman correlation** between predicted and measured coverage.

---

## What the script does

1. Loads one or two genome sources (FASTA + forward/reverse coverage `.npz`)
2. Builds overlapping windows across the genome
3. Splits windows into **5 cross-validation folds**
4. Runs an ensemble of models per fold (`f0..f7`)
5. Computes Spearman correlation for each window
6. Reports median + mean correlation per CV fold
7. Saves a CSV summary to:

```text
Results/Cross_validation/results_<chrom>.csv
```

---

## Inputs

### Required arguments

| Argument | Description |
|---------|-------------|
| `--src1-chrom` | Genome/chromosome identifier |
| `--src1-fasta` | FASTA file for source genome |
| `--src1-npz-fwd` | Forward-strand RNA-seq coverage `.npz` |
| `--src1-npz-rev` | Reverse-strand RNA-seq coverage `.npz` |

### Optional second source

| Argument | Description |
|---------|-------------|
| `--src2-chrom` | Second genome identifier |
| `--src2-fasta` | FASTA file |
| `--src2-npz-fwd` | Forward coverage `.npz` |
| `--src2-npz-rev` | Reverse coverage `.npz` |

### Ensemble size

| Argument | Default |
|---------|---------|
| `--ensemble` | 8 models |

---

## Model directory structure

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

## Example usage

```bash
python scripts/predict_kfold.py \
  --src1-chrom Mpneumo \
  --src1-fasta Data/genome/Mpneumo.fa \
  --src1-npz-fwd Data/normalized_expression/Mpneumo_fwd_norm.npz \
  --src1-npz-rev Data/normalized_expression/Mpneumo_rev_norm.npz \
  --ensemble 8
```

---

## Output

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

## Metrics

For each fold:

- **Spearman correlation** is computed per window
- Median and mean are reported

Overall summary:

- Mean ± std across folds

---

## Notes

- Uses deterministic TensorFlow settings (`SEED=42`)
- Runs predictions in mixed precision (`mixed_bfloat16`)
- Designed for evaluation of ExoShorkie fine-tuned ensembles

