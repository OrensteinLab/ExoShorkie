# ExoShorkie

ExoShorkie is a method for accurately predicting RNA-seq coverage of exogenous genomes in yeast using transfer learning, as proposed in the paper:

*ExoShorkie: Predicting RNA-seq coverage of exogenous genomes in yeast by transfer learning*

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Trained Models](#trained-models)
- [Pretrained Shorkie Models](#pretrained-shorkie-models)
- [Prediction](#prediction)
- [Training](#training)
- [Model Training Data](#model-training-data)
- [Contact](#contact)

---

## Introduction

ExoShorkie is the first method that leverages transfer learning from a native-genome-trained yeast model to predict RNA-seq coverage of exogenous DNA.

It is based on the Shorkie model introduced in:

*Predicting dynamic expression patterns in budding yeast with a fungal DNA language model*  
Chao et al.  
https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1

---

## Getting Started

ExoShorkie is designed to run inside a Docker environment.

An NVIDIA GPU is recommended for efficient training and inference, but ExoShorkie can also run on CPU with reduced performance.

---

### Prerequisites

- **Docker** installed on your system  
- **(Optional) GPU Support:** NVIDIA GPU with CUDA drivers and NVIDIA Container Toolkit installed

---

### Dependencies

ExoShorkie relies on the Baskerville sequence modeling framework:

https://github.com/calico/baskerville-yeast

All dependencies are installed automatically inside the Docker image.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/OrensteinLab/ExoShorkie.git
cd ExoShorkie
```

---

### 2. Build the Docker image

```bash
docker build -t exoshorkie .
```

This creates a local Docker image named `exoshorkie` containing all required dependencies.

---

### 3. Enter the Docker workspace

```bash
docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)":/workspace -w /workspace \
  exoshorkie bash
```

This opens an interactive shell inside the container, with the repository mounted at `/workspace`.

---

## Trained Models

All trained ExoShorkie ensemble models are available on Hugging Face:

https://huggingface.co/Jonathan-Mandl/ExoShorkie-models

After downloading, the expected directory structure is:

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

Each dataset contains an ensemble of 8 fine-tuned models (f0–f7) for each cross-validation fold.

---

## Pretrained Shorkie Models

ExoShorkie fine-tunes an ensemble of pretrained Shorkie models.  
These pretrained models are **not included** in this repository.

They are hosted on Google Cloud Storage by the original authors:

```text
gs://seqnn-share/shorkie/
```

To download them, you need `gsutil` (Google Cloud SDK installed):

```bash
gsutil -m cp -r gs://seqnn-share/shorkie Models/shorkie
```

Expected structure:

```text
Models/
└── shorkie/
    ├── f0/
    ├── f1/
    ├── f2/
    ├── f3/
    ├── f4/
    ├── f5/
    ├── f6/
    └── f7/
```

---

## Prediction

ExoShorkie provides a prediction script for generating RNA-seq coverage predictions over an input FASTA genome.

### Script arguments

The prediction script expects:

- `--chrom` : dataset name the model was trained on  
- `--cv`    : cross-validation fold index  
- `--fold`  : ensemble member index (0–7)  
- `--fasta` : FASTA file of the genome to predict on  
- `--out`   : output `.npz` file path  
- `--rc`    : (optional) generate predictions on the reverse-complement strand  

---

### Output format

Predictions are saved as compressed NumPy `.npz` files in the `Results/` directory.

- `pred_bp` contains expanded predictions at **base-pair resolution**

---

### Quick prediction example

```bash
python predict.py \
  --chrom M_pneumoniae\
  --cv 0 \
  --fold 0 \
  --fasta Data/genome/Mpneumo.fa \
  --out pred_Mpneumo_cv0_f0.npz
```

---
## Training

ExoShorkie provides a training script for fine-tuning the native-genome baseline **NatShorkie** models on an exogenous genome using **5-fold cross-validation**.

### Script arguments

The training script expects the following inputs:

- `--name` : name of the exogenous dataset  
- `--chrom` : chromosome / genome identifier  
- `--npz-fwd` : forward-strand RNA-seq coverage `.npz` file  
- `--npz-rev` : reverse-strand RNA-seq coverage `.npz` file  
- `--fasta` : FASTA file of the exogenous genome  
- `--ensemble` : number of ensemble members to train per fold  
- `--target-wins' : (optional) target number of training windows per fold (default: 10000)

---

### Output format

Fine-tuned models are saved as `.h5` files under the `Models/` directory.

Expected directory structure:

```text
Models/
├── <chrom>/cv0/
│   ├── f0/model_finetune.h5
│   ├── f1/model_finetune.h5
│   ├── ...
│   └── f7/model_finetune.h5
├── <chrom>/cv1/
│   └── ...
└── <chrom>/cv4/
    └── ...
```

Where `<chrom>` is the genome/dataset name (e.g., Mpneumo, HPRT1).


###  Example usage
```bash
python train.py \
  --name Mpneumo \
  --chrom Mpneumo \
  --fasta Data/genome/Mpneumo.fa \
  --npz-fwd Data/normalized_expression/Mpneumo_fwd_norm.npz \
  --npz-rev Data/normalized_expression/Mpneumo_rev_norm.npz \
  --ensemble 8
  ```

## Model Training Data

ExoShorkie is trained on six exogenous RNA-seq datasets described in the main paper.

All preprocessed datasets used for training and evaluation are available on Figshare:

https://doi.org/10.6084/m9.figshare.31075375

---

## Additional Documentation

More detailed pipeline guides are available in the `docs/` folder:

- [NatShorkie: Fine-tuning on the native yeast genome](docs/NatShorkie.md)
- [Distillation: Synthetic data + student training](docs/Distillation.md)
- [Evaluation: Cross-genome benchmarking](docs/Evaluation.md)

## Contact

For issues or questions regarding ExoShorkie, please contact: jonathan.mandl2@gmail.com

---
