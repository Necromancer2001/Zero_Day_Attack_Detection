import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ==================== CONFIG (SEED 1 ONLY) ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

RANDOM_SEED = 1
TRAIN_ONLY_BENIGN = True
BENIGN_LABEL = 2
ZERO_DAY_LABEL = 0

ECOA_ITER = 20
ECOA_POP = 5
ECOA_EPOCHS = 3
FINAL_EPOCHS = 50
BATCH_SIZE = 1024
PATIENCE = 5
LATENT_DIM = 10

# ==================== MODEL + HELPERS (DEFINED ONCE) ====================
class TabularDataset(Dataset):
    def __init__(self, X, y, w=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.w = torch.ones(len(y), dtype=torch.float32) if w is None else torch.tensor(w, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]

class WeightedBetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.drop = nn.Dropout(dropout)
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        self.act = nn.LeakyReLU(0.2)
    def encode(self, x):
        h = self.act(self.bn1(self.enc1(x)))
        h = self.drop(h)
        h = self.act(self.bn2(self.enc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def decode(self, z):
        h = self.act(self.bn3(self.dec1(z)))
        h = self.drop(h)
        h = self.act(self.bn4(self.dec2(h)))
        return torch.sigmoid(self.dec_out(h))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def vae_loss(recon, x, mu, logvar, beta, weights, gamma=2.0):
    mse = ((recon - x) ** 2).mean(dim=1)
    focal = (mse.detach() ** gamma)
    recon_loss = (focal * mse * weights).mean()
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = (kl * weights).mean()
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

def train_one_epoch(model, loader, optimizer, beta):
    model.train()
    total = 0
    for xb, yb, wb in loader:
        xb, wb = xb.to(DEVICE), wb.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar, _ = model(xb)
        loss, _, _ = vae_loss(recon, xb, mu, logvar, beta, wb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

def eval_loss(model, loader, beta):
    model.eval()
    total = 0
    with torch.no_grad():
        for xb, yb, wb in loader:
            xb, wb = xb.to(DEVICE), wb.to(DEVICE)
            recon, mu, logvar, _ = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar, beta, wb)
            total += loss.item() * len(xb)
    return total / len(loader.dataset)

def decode_params(vec):
    hidden = int(32 + vec[0] * 64)
    dropout = 0.1 + vec[1] * 0.4
    lr = 10 ** (-4 + vec[2] * 2.3)
    beta = 0.5 + vec[3] * 1.5
    return hidden, dropout, lr, beta

# ==================== SEED 1 ONLY - NO LOOP ====================
print(f"\n\n########## VAE - SEED {RANDOM_SEED} ##########")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# INPUT = GA output for this seed, OUTPUT = separate folder per seed
GA_FILE = f"GA_feature_selection/seed{RANDOM_SEED}/GA_reduced_dataset_seed{RANDOM_SEED}.csv"
OUT_DIR = f"VAE_output_seed{RANDOM_SEED}"
os.makedirs(OUT_DIR, exist_ok=True)

# ==================== LOAD ====================
print("\nLoading GA-selected dataset...")
df = pd.read_csv(GA_FILE)
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

if "Sub_Category" in df.columns:
    weight_labels = df["Sub_Category"].values
    y = df["label"].values.astype(np.int64) if "label" in df.columns else df.iloc[:, -2].values.astype(np.int64)
    X = df.drop(columns=["label", "Sub_Category"]).values.astype(np.float32)
    feature_names = list(df.drop(columns=["label", "Sub_Category"]).columns)
    print("Sub_Category found, using it for weighting")
else:
    target_col = df.columns[-1]
    y = df[target_col].values.astype(np.int64)
    weight_labels = y
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    feature_names = list(df.drop(columns=[target_col]).columns)
    print("Sub_Category not found, using label for weighting")

print(f"Input features: {X.shape[1]}")

# ==================== WEIGHTS ====================
unique, counts = np.unique(weight_labels, return_counts=True)
class_weight_dict = {}
for u, c in zip(unique, counts):
    w = len(weight_labels) / (len(unique) * c)
    class_weight_dict[int(u)] = float(w)
sample_weights_all = np.array([class_weight_dict[int(v)] for v in weight_labels], dtype=np.float64)

# ==================== SPLITTING ====================
print("\nSplitting dataset...")
if TRAIN_ONLY_BENIGN:
    print("MODE: ZERO-DAY - Training on Benign only")
    mask_zero = (y == ZERO_DAY_LABEL)
    X_zero, y_zero = X[mask_zero], y[mask_zero]
    mask_rest = ~mask_zero
    X_rest, y_rest = X[mask_rest], y[mask_rest]
    mask_benign = (y_rest == BENIGN_LABEL)
    X_benign, y_benign = X_rest[mask_benign], y_rest[mask_benign]
    print(f"Benign pool: {X_benign.shape} | Zero-day held out: {X_zero.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X_benign, y_benign, test_size=0.1, random_state=42)
    X_test, y_test = X_zero, y_zero
    w_train = np.ones(len(y_train))
else:
    mask_zero = (y == ZERO_DAY_LABEL)
    X_zero, y_zero = X[mask_zero], y[mask_zero]
    X_rest, y_rest = X[~mask_zero], y[~mask_zero]
    w_rest = sample_weights_all[~mask_zero]
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_rest, y_rest, w_rest, test_size=0.1, random_state=42, stratify=y_rest)
    X_test, y_test = X_zero, y_zero

print(f"Training: {X_train.shape} | Validation: {X_val.shape} | Test zero-day: {X_test.shape}")

# ==================== SCALING ====================
print("\nApplying Min-Max scaling...")
if X_train.max() > 1.5:
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print("Scaling applied")
else:
    print("Data already 0-1, skipping re-scaling")
    scaler_path = f"data/minmax_scaler_seed{RANDOM_SEED}.pkl"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else MinMaxScaler()
joblib.dump(scaler, os.path.join(OUT_DIR, f"vae_scaler_seed{RANDOM_SEED}.pkl"))

# ==================== LOADERS ====================
def make_loaders(Xtr, ytr, wtr, Xv, yv):
    train_ds = TabularDataset(Xtr, ytr, wtr)
    val_ds = TabularDataset(Xv, yv)
    if TRAIN_ONLY_BENIGN:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    else:
        sampler = WeightedRandomSampler(weights=torch.tensor(wtr, dtype=torch.double), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = make_loaders(X_train, y_train, w_train, X_val, y_val)

# ==================== ECOA ====================
print("\n========== ECOA OPTIMIZATION ==========")
pop = np.random.rand(ECOA_POP, 4)
best_vec = None
best_loss = float("inf")
best_params = None

for it in range(ECOA_ITER):
    for i in range(ECOA_POP):
        hidden, dropout, lr, beta = decode_params(pop[i])
        model = WeightedBetaVAE(X_train.shape[1], hidden, LATENT_DIM, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(ECOA_EPOCHS):
            train_one_epoch(model, train_loader, opt, beta)
        val_loss = eval_loss(model, val_loader, beta)
        if val_loss < best_loss:
            best_loss = val_loss
            best_vec = pop[i].copy()
            best_params = (hidden, dropout, lr, beta)
        print(f"ECOA {it+1}/{ECOA_ITER} | Crayfish {i+1}/{ECOA_POP} | hidden={hidden} dropout={dropout:.3f} lr={lr:.5f} beta={beta:.3f} | val={val_loss:.5f} | best={best_loss:.5f}")
        del model
    temp = 20 + 15 * np.random.rand()
    for i in range(ECOA_POP):
        if temp > 30:
            pop[i] = np.clip(pop[i] + np.random.randn(4) * 0.2, 0, 1)
        else:
            r = np.random.rand(4)
            pop[i] = np.clip(pop[i] + r * (best_vec - pop[i]) + np.random.randn(4) * 0.05, 0, 1)

print(f"\nBest params Seed {RANDOM_SEED}: hidden={best_params[0]}, dropout={best_params[1]:.4f}, lr={best_params[2]:.6f}, beta={best_params[3]:.4f}")

# ==================== FINAL TRAINING ====================
print("\n========== FINAL TRAINING ==========")
hidden_f, dropout_f, lr_f, beta_f = best_params
model = WeightedBetaVAE(X_train.shape[1], hidden_f, LATENT_DIM, dropout_f).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_f)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
history = []
best_val = float("inf")
patience_counter = 0
for epoch in range(FINAL_EPOCHS):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, beta_f)
    val_loss = eval_loss(model, val_loader, beta_f)
    scheduler.step(val_loss)
    elapsed = time.time() - t0
    print(f"Epoch [{epoch+1}/{FINAL_EPOCHS}] Train: {train_loss:.6f} Val: {val_loss:.6f} Time: {elapsed:.1f}s")
    history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"best_vae_model_seed{RANDOM_SEED}.pth"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break
pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, f"vae_training_history_seed{RANDOM_SEED}.csv"), index=False)

# ==================== LATENT + THRESHOLD ====================
print("\nLoading best VAE model...")
model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"best_vae_model_seed{RANDOM_SEED}.pth"), map_location=DEVICE))
model.eval()

def extract_latent(X_np, y_np):
    loader = DataLoader(TabularDataset(X_np, y_np), batch_size=1024, shuffle=False)
    latents, recons = [], []
    with torch.no_grad():
        for xb, yb, wb in loader:
            xb = xb.to(DEVICE)
            recon, mu, logvar, z = model(xb)
            mse = ((recon - xb) ** 2).mean(dim=1).cpu().numpy()
            latents.append(mu.cpu().numpy())
            recons.append(mse)
    lat = np.concatenate(latents, axis=0)
    rec = np.concatenate(recons, axis=0)
    out = np.column_stack([lat, y_np, rec])
    return out, rec

print("\nExtracting latent features...")
train_lat, train_rec = extract_latent(X_train, y_train)
val_lat, val_rec = extract_latent(X_val, y_val)
test_lat, test_rec = extract_latent(X_test, y_test)
print(f"train: {train_lat.shape} mean recon {train_rec.mean():.5f}")
print(f"val: {val_lat.shape} mean recon {val_rec.mean():.5f}")
print(f"test_zero_day: {test_lat.shape} mean recon {test_rec.mean():.5f}")

cols = [f"latent_{i}" for i in range(LATENT_DIM)] + ["label", "recon_error"]
pd.DataFrame(train_lat, columns=cols).to_csv(os.path.join(OUT_DIR, f"vae_train_latent_seed{RANDOM_SEED}.csv"), index=False)
pd.DataFrame(val_lat, columns=cols).to_csv(os.path.join(OUT_DIR, f"vae_validation_latent_seed{RANDOM_SEED}.csv"), index=False)
pd.DataFrame(test_lat, columns=cols).to_csv(os.path.join(OUT_DIR, f"vae_test_zero_day_latent_seed{RANDOM_SEED}.csv"), index=False)

thr95 = np.percentile(val_rec, 95)
thr99 = np.percentile(val_rec, 99)
flag95 = (test_rec > thr95).mean() * 100
flag99 = (test_rec > thr99).mean() * 100
print(f"\nThreshold 95%: {thr95:.5f} | 99%: {thr99:.5f}")
print(f"Zero-day flagged at 95%: {flag95:.2f}% | at 99%: {flag99:.2f}%")

config = {
    "seed": int(RANDOM_SEED),
    "input_dim": int(X_train.shape[1]), "hidden_dim": int(hidden_f), "latent_dim": int(LATENT_DIM),
    "dropout": float(dropout_f), "lr": float(lr_f), "beta": float(beta_f),
    "best_val_loss": float(best_val), "train_only_benign": bool(TRAIN_ONLY_BENIGN),
    "benign_label": int(BENIGN_LABEL), "zero_day_label": int(ZERO_DAY_LABEL),
    "threshold_95": float(thr95), "threshold_99": float(thr99),
    "zero_day_flag_95": float(flag95), "zero_day_flag_99": float(flag99),
    "feature_names": feature_names
}
with open(os.path.join(OUT_DIR, f"feature_extraction_config_seed{RANDOM_SEED}.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n========== VAE SEED {RANDOM_SEED} COMPLETED ==========")