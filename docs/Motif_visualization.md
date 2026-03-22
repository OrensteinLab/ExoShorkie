# Motif visualization

Notebooks compare genomic vs. genome-specific (ISM) attributions, scan motifs, and plot low-correlation windows with model predictions vs. RNA-seq coverage.

## Dependencies (not in the main `Dockerfile`)

`Motif_visualization/` uses **PyTorch** (tangermeme / seqlets), **tangermeme**, **logomaker**, plus **Jupyter** if you use JupyterLab or classic notebooks. The base `exoshorkie` image matches the rest of the repo (TensorFlow + Baskerville); install the following in **your conda/venv** or **inside a running container** before opening the notebooks:

```bash
# PyTorch with CUDA 12.x wheels (adjust index if you use CPU-only or a different CUDA)
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

python -m pip install --no-cache-dir logomaker tangermeme jupyterlab ipykernel
```

Install **after** any TensorFlow image setup so pip does not replace the NVIDIA TensorFlow build. For a local environment, pick a [PyTorch install](https://pytorch.org/get-started/locally/) that matches your CUDA/driver.

## How to run

Use the **repository root** as the working directory so `from src....` imports resolve, then open and run `Motif_visualization/Mpneumo_correlations.ipynb` (or `chr7_correlations.ipynb`, etc.) in your usual Jupyter or IDE setup.

## Configuration (first code cell)

Adjust paths and IDs for your assembly. Example names follow `Mpneumo_correlations.ipynb`.

| Variable | Role |
|----------|------|
| `CHROM_ID` | Sequence name to read from the FASTA (e.g. `Mpneumo`). |
| `FA_PATH` | Path to the FASTA containing that sequence. |
| `ISM_GENOMIC_FWD`, `ISM_GENOMIC_REV` | Genomic-model ISM maps (`.dat`, memmap `(L, 4)` float32). |
| `ISM_RANDOM_FWD`, `ISM_RANDOM_REV` | Genome-specific model ISM maps (same layout). |
| `GENOMIC_MODEL` | Distilled genomic student weights (`.h5`). |
| `RANDOM_MODEL` | Distilled exogenous genome student weights (`.h5`). |
| `PARAMS_JSON` | Shorkie `params.json` (trunk architecture). |
| `NPZ_COV_FWD`, `NPZ_COV_REV` | Normalized strand coverage `.npz` files. |
| `MEME_DB_PATH` | Motif database (MEME format, e.g. SwissRegulon). |
| `WINDOW_SIZE` | Window length (bp) for correlation/plotting helpers. |
| `threshold` | Seqlet / motif hit threshold. |
| `mu_r`, `sigma_r` | Mean and std for inverse-transforming **random-model** predictions. |
| `mu_g`, `sigma_g` | Mean and std for **genomic-model** predictions. |
| `OUT_DIR` | Directory for saved figures (created under `ISM/Notebook_output/<genome>/…`). |
| `start_genomic_bp` | (chr7 notebook only) Genome coordinate offset for plots. |
