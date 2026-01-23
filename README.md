# ExoShorkie

ExoShorkie is a method for accurately predicting RNA-seq coverage of exogenous genomes in yeast using transfer learning, as proposed in the paper *ExoShorkie: Predicting RNA-seq coverage of exogenous genomes in yeast by transfer learning*.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Training and Evaluation Scripts](#training-and-evaluation-scripts)
- [Model Training Data](#model-training-data)
- [Trained Models](#trained-models)
- [Contact](#contact)

## Introduction

ExoShorkie is the first method that leverages transfer learning from a native-genome-trained model to predict RNA-seq coverage of exogenous DNA in yeast. It is based on the architecture and pretrained model introduced in the paper *Predicting dynamic expression patterns in budding yeast with a fungal DNA language model* by Chao et al.  
https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1

## Getting Started

ExoShorkie is run using a Conda environment. While an NVIDIA GPU is recommended for faster training and inference, ExoShorkie can also be run on a standard CPU (with reduced performance).

### Prerequisites

- **Conda** installed on your system (Miniconda or Anaconda).
- **(Optional) GPU Support:** A compatible NVIDIA GPU with appropriate CUDA drivers installed. GPU support depends on the local system configuration. The provided Conda environment is CPU-compatible by default.

### Setup

1. Clone the repository:
```bash
git clone <REPO_URL>
cd ExoShorkie
```

2. Create the Conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate shorki
```

## Training and Evaluation Scripts

Training scripts for ExoShorkie models, along with evaluation and interpretability scripts, are available in the `scripts/` directory. RNA-seq preprocessing scripts are located in the `Preprocessing/` directory. Plotting scripts used to recreate the figures in the paper are located in the `Plots/` directory. Motif visualization and related analyses are located in the `Motif_visualization/` directory.

## Model Training Data

ExoShorkie is trained on six exogenous RNA-seq datasets described in the main paper. All datasets used for model training and evaluation are archived on Figshare at:  
https://doi.org/10.6084/m9.figshare.31075375

## Trained Models

Trained ExoShorkie ensemble models are available on Hugging Face at:  
https://huggingface.co/Jonathan-Mandl/ExoShorkie-models

## Contact

For issues or questions regarding ExoShorkie, please contact:  
**Jonathan Mandl**  
📧 jonathan.mandl2@gmail.com
