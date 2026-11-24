# tune_if_seed1.py
import pandas as pd, numpy as np, os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

BASE = "VAE_output_seed1"
train = pd.read_csv(os.path.join(BASE, "vae_train_latent_seed1.csv"))
val = pd.read_csv(os.path.join(BASE, "vae_validation_latent_seed1.csv"))
test = pd.read_csv(os.path.join(BASE, "vae_test_zero_day_latent_seed1.csv"))

latent_cols = [c for c in train.columns if c.startswith("latent_")]
X_train, X_val, X_test = train[latent_cols].values, val[latent_cols].values, test[latent_cols].values

model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
model.fit(X_train)

# Use anomaly score, not hard predict
val_scores = model.decision_function(X_val)
test_scores = model.decision_function(X_test)

# Try 10 thresholds - lower score = more anomalous
for percentile in [5, 10, 15, 20, 30]:
    thresh = np.percentile(val_scores, percentile)
    val_pred = (val_scores < thresh).astype(int)
    test_pred = (test_scores < thresh).astype(int)
    fpr = val_pred.mean()*100
    det = test_pred.mean()*100
    print(f"Percentile {percentile}% -> Thresh {thresh:.4f} | FPR {fpr:.2f}% | Detection {det:.2f}%")