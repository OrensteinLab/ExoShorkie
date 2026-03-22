FROM nvcr.io/nvidia/tensorflow:24.06-tf2-py3
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Keep NVIDIA’s TensorFlow intact: do NOT let pip replace it
RUN python -m pip install --no-cache-dir \
    biopython \
    natsort==7.1.1 \
    intervaltree==3.1.0 \
    networkx==2.8.4 \
    pandas==1.5.3 \
    pybigwig==0.3.18 \
    pybedtools==0.9.0 \
    pysam==0.22.0 \
    qnorm==0.8.1 \
    seaborn==0.12.2 \
    scikit-learn==1.2.2 \
    statsmodels==0.13.5 \
    tabulate==0.8.10 \
    tqdm==4.65.0 \
    joblib==1.1.1 \
    h5py==3.10.0 \
    matplotlib==3.7.1 \
    scipy==1.9.1

# Freeze baskerville to commit used in ExoShorkie paper
RUN python -m pip install --no-cache-dir --no-deps \
    "git+https://github.com/calico/baskerville-yeast.git@88e89f48e7df73c0856ce93ae35e8878794d19e9"

CMD ["bash"]
