# if_final_seed1.py
import pandas as pd, numpy as np, os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

BASE = "VAE_output_seed1"
train = pd.read_csv(os.path.join(BASE, "vae_train_latent_seed1.csv"))
val = pd.read_csv(os.path.join(BASE, "vae_validation_latent_seed1.csv"))
test = pd.read_csv(os.path.join(BASE, "vae_test_zero_day_latent_seed1.csv"))

latent_cols = [c for c in train.columns if c.startswith("latent_")]
X_train, X_val, X_test = train[latent_cols].values, val[latent_cols].values, test[latent_cols].values

# FIXED: contamination 0.10, not 0.05
model = IsolationForest(n_estimators=200, contamination=0.10, random_state=42, n_jobs=-1)
model.fit(X_train)

val_pred = (model.predict(X_val) == -1).astype(int)
test_pred = (model.predict(X_test) == -1).astype(int)

print(f"FPR: {val_pred.mean()*100:.2f}% | Detection: {test_pred.mean()*100:.2f}%")
y_true = np.concatenate([np.zeros(len(val_pred)), np.ones(len(test_pred))])
y_pred = np.concatenate([val_pred, test_pred])
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["Benign","Zero-day"]))

pd.DataFrame({"pred": test_pred}).to_csv(os.path.join(BASE, "if_test_pred_seed1.csv"), index=False)
pd.DataFrame({"pred": val_pred}).to_csv(os.path.join(BASE, "if_val_pred_seed1.csv"), index=False)
print("Saved FINAL IF for Seed 1 at 10%")