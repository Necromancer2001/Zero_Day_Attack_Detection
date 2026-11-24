# Zero-Day Attack Detection in IoT

Detecting unseen / zero-day attacks in IoT traffic using Genetic Algorithm + VAE + ECOA + Isolation Forest.

Dataset: CICIoT2023
https://www.unb.ca/cic/datasets/iotdataset-2023.html

## Pipeline

1.  `preprocessing.py` - Load and clean CICIoT2023 data, train/test split
2.  `genetic.py` + `featureselection.py` + `laplacian.py` - Genetic Algorithm for feature selection (with Laplacian score)
3.  `VAE_final.py` - Train Variational Autoencoder on normal traffic only, compute reconstruction error
4.  `thersh.py` - Threshold analysis (e.g. 95th percentile) for zero-day detection
5.  `seed1IF.py` - Isolation Forest on VAE latent / reconstruction features for final anomaly detection

## How to Run

```bash
# 1. Download CICIoT2023 CSVs and put them in dataset/ folder
# 2. Install deps
pip install torch scikit-learn pandas numpy

# 3. Run in order
python preprocessing.py
python genetic.py
python VAE_final.py
python thersh.py
python seed1IF.py
