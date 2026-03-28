# Motif visualization

Notebooks compare genomic vs. genome-specific (ISM) attributions, scan motifs, and plot low-correlation windows with model predictions vs. RNA-seq coverage.

## Docker image (`Dockerfile.motifs`)

From the repository root, build the main image, then the motif layer:

```bash
docker build -t exoshorkie .
docker build -f Dockerfile.motifs -t exoshorkie-motifs .
```

The image sets `JUPYTER_*` paths under `/workspace/.jupyter` so Jupyter does not try to create `/.local` when the container runs as a non-root `--user` (missing `HOME`).

### JupyterLab in Docker

From the **repository root** on the host (so `-v "$(pwd)":/workspace` is the repo):

```bash
docker run -it --rm --gpus all \
  -p 8888:8888 \
  -v "$(pwd)":/workspace -w /workspace \
  exoshorkie-motifs \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open the URL printed in the terminal (includes an access token). Remote server: use SSH port forwarding, e.g. `ssh -L 8888:localhost:8888 user@server`.

If you still see permission errors (custom `--user`), add `-e HOME=/workspace` to the `docker run` line.

## How to run (local IDE / no Jupyter in Docker)

Use the **repository root** as the working directory so `from src....` imports resolve, then open `Motif_visualization/Mpneumo_correlations.ipynb` (or `chr7_correlations.ipynb`, etc.) in your editor or a local Jupyter.

## Configuration (first code cell)

Adjust paths and IDs for your assembly. Example names follow `Mpneumo_correlations.ipynb`.

**Sequence and inputs**

- **`CHROM_ID`** — FASTA record id to load (e.g. `Mpneumo`).
- **`FA_PATH`** — Path to that FASTA.
- **`ISM_GENOMIC_FWD`**, **`ISM_GENOMIC_REV`** — Genomic-model ISM maps (`.dat`, memmap shape `(L, 4)`, float32).
- **`ISM_RANDOM_FWD`**, **`ISM_RANDOM_REV`** — Genome-specific (exogenous) ISM maps, same layout.
- **`NPZ_COV_FWD`**, **`NPZ_COV_REV`** — Normalized strand coverage `.npz` files.

**Models and motif resources**

- **`GENOMIC_MODEL`**, **`RANDOM_MODEL`** — Distilled student weights (`.h5`) for genomic vs. genome-specific models.
- **`PARAMS_JSON`** — Shorkie `params.json` (trunk architecture).
- **`MEME_DB_PATH`** — Motif database in MEME format (e.g. SwissRegulon).

**Plotting and normalization**

- **`WINDOW_SIZE`** — Window length in bp for correlation and plotting helpers.
- **`threshold`** — Seqlet / motif hit threshold.
- **`mu_r`**, **`sigma_r`** — Mean and standard deviation for inverse-transforming random-model predictions.
- **`mu_g`**, **`sigma_g`** — Mean and standard deviation for genomic-model predictions.
- **`OUT_DIR`** — Output directory for figures (typically under `ISM/Notebook_output/<genome>/…`).
- **`start_genomic_bp`** — *(Human chr7 notebook only.)* Genome coordinate offset for axis-aligned plots.
