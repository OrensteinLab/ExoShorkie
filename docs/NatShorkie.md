
## Fine-tune on native yeast genomic data (NatShorkie)

ExoShorkie provides a training script for fine-tuning the pretrained **Shorkie** models on strand-resolved yeast genomic RNA-seq data to create the native-genome baseline **NatShorkie**.  
This corresponds to the genomic adaptation step described in the paper.

---


**Required arguments**

- **`--npz-fwd`**, **`--npz-rev`** — Forward- and reverse-strand yeast genomic coverage `.npz` files.
- **`--fasta`** — Yeast genome FASTA.

**Optional**

- **`--ensemble`** — Number of ensemble members to train (default: 8).

---

### Output format

Fine-tuned models are saved as `.h5` files under the `Models/` directory.

Expected directory structure:

```text
Models/
└── NatShorkie/
    ├── f0/model_finetune.h5
    ├── f1/model_finetune.h5
    ├── ...
    └── f7/model_finetune.h5
```

Example usage

```bash
python scripts/fine_tune_NatShorkie.py \
  --fasta 'Data/Yeast genome/S288C_R64.fa' \
  --npz-fwd 'Data/Yeast genome/genomic_fwd_norm.npz' \
  --npz-rev 'Data/Yeast genome/genomic_rev_norm.npz'
```

## Evaluation (NatShorkie)

The NatShorkie evaluation script computes window-level Spearman correlations between predicted and true RNA-seq coverage.

---


** Required arguments **

- **`--src1-name`** — Dataset label for outputs and logging.
- **`--src1-chrom`** — Chromosome or genome identifier (FASTA key and coverage keys as used by the loader).
- **`--src1-npz-fwd`**, **`--src1-npz-rev`** — Strand-resolved coverage `.npz` files.
- **`--src1-fasta`** — Genome FASTA.

**Second source (optional)** — Supply all `--src2-*` fields together if you evaluate two sources in one run.

- **`--src2-name`**, **`--src2-chrom`**, **`--src2-npz-fwd`**, **`--src2-npz-rev`**, **`--src2-fasta`**

**Other**

- **`--ensemble`** — Number of ensemble members to average (default: 8).

---

### Example usage

```bash
python scripts/predict_NatShorkie.py \
  --src1-name Mpneumo \
  --src1-chrom Mpneumo \
  --src1-fasta 'Data/M. pneumoniae/Mpneumo.fa'  \
  --src1-npz-fwd 'Data/M. pneumoniae/Mpneumo_fwd_norm.npz' \
  --src1-npz-rev 'Data/M. pneumoniae/Mpneumo_rev_norm.npz'
```

### Output format

Correlation results are saved as compressed NumPy `.npz` files under:

```text
Results/Correlations/correlations_NatShorkie_<chrom_name>.npz
```

The output file contains:

- `starts` : start coordinates for all windows in the source for which correlations are calculated  
- `correlations` : Spearman correlation values for all evaluated windows  
- `median_spearman` : median Spearman correlation across all windows  
- `std_spearman` : standard deviation of Spearman correlations across all windows  