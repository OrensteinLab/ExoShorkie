
## Fine-tune on native yeast genomic data (NatShorkie)

ExoShorkie provides a training script for fine-tuning the pretrained **Shorkie** models on strand-resolved yeast genomic RNA-seq data to create the native-genome baseline **NatShorkie**.  
This corresponds to the genomic adaptation step described in the paper.

---

### Script arguments

The fine-tuning script expects the following inputs:

- `--npz-fwd` : forward-strand yeast RNA-seq coverage `.npz` file  
- `--npz-rev` : reverse-strand yeast RNA-seq coverage `.npz` file  
- `--fasta`   : FASTA file of the yeast genome  
- `--ensemble` : number of models in ensemble (default: 8)

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
  --fasta Data/genome/S288c_reference.fa \
  --npz-fwd Data/normalized_expression/genomic_fwd_norm.npz \
  --npz-rev Data/normalized_expression/genomic_rev_norm.npz
```

## Evaluation (NatShorkie)

The NatShorkie evaluation script computes window-level Spearman correlations between predicted and true RNA-seq coverage.

---

### Script arguments

The NatShorkie prediction/evaluation script expects the following inputs:

- `--ensemble` : size of ensemble to test

#### Required source (source 1)

- `--src1-name` : name of the dataset  
- `--src1-chrom` : chromosome / genome identifier  
- `--src1-npz-fwd` : forward-strand RNA-seq coverage `.npz` file  
- `--src1-npz-rev` : reverse-strand RNA-seq coverage `.npz` file  
- `--src1-fasta` : FASTA file of the genome  

#### Optional second source (source 2)

- `--src2-name` : name of the dataset  
- `--src2-chrom` : chromosome / genome identifier  
- `--src2-npz-fwd` : forward-strand RNA-seq coverage `.npz` file  
- `--src2-npz-rev` : reverse-strand RNA-seq coverage `.npz` file  
- `--src2-fasta` : FASTA file of the genome  

---

### Example usage

```bash
python scripts/predict_NatShorkie.py \
  --src1-name Mpneumo \
  --src1-chrom Mpneumo \
  --src1-fasta Data/genome/Mpneumo.fa \
  --src1-npz-fwd Data/normalized_expression/Mpneumo_fwd_norm.npz \
  --src1-npz-rev Data/normalized_expression/Mpneumo_rev_norm.npz

### Output format

Correlation results are saved as compressed NumPy `.npz` files under:

```text
Results/Correlations/correlations_NatShorkie_<chrom_name>.npz

The output file contains:

- `starts` : start coordinates for all windows in the source for which correlations are calculated  
- `correlations` : Spearman correlation values for all evaluated windows  
- `median_spearman` : median Spearman correlation across all windows  
- `std_spearman` : standard deviation of Spearman correlations across all windows  