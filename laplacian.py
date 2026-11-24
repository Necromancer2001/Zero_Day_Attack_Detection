# laplacian_5_seeds.py - Normalized Laplacian + SPEC Fusion for 5 Seeds
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
import joblib
import os

TOP_K = 20

# ==================== LOOP FOR SEED 1 TO 5 ====================
for SEED in [1,2,3,4,5]:
    print(f"\n\n########## LAPLACIAN - SEED {SEED} ##########")

    INPUT = f"data/ciciot_minmax_seed{SEED}.csv"
    OUTPUT_CSV = f"data/ciciot_selected_seed{SEED}.csv"
    OUTPUT_PKL = f"data/selected_features_seed{SEED}.pkl"

    print(f"[1] Loading {INPUT}")
    df_full = pd.read_csv(INPUT)
    print(f"    Shape: {df_full.shape}")

    # FIX: Drop BOTH label and Sub_Category from features
    target_cols = []
    if "label" in df_full.columns:
        target_cols.append("label")
    if "Sub_Category" in df_full.columns:
        target_cols.append("Sub_Category")
    # Fallback if names different - take last 2 cols
    if not target_cols:
        target_cols = list(df_full.columns[-2:])

    print(f"    Target cols: {target_cols}")
    label = target_cols[0]  # main label for saving

    X_full = df_full.drop(columns=target_cols)
    y_full = df_full[target_cols]  # keep both for saving

    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_full = X_full.fillna(X_full.mean())
    X_full = X_full.dropna(axis=1, how='all')
    feature_names = X_full.columns.tolist()
    print(f"    Features: {len(feature_names)}")

    if X_full.shape[0] > 10000:
        X_sample = X_full.sample(n=10000, random_state=42)
    else:
        X_sample = X_full

    X_np = np.nan_to_num(X_sample.values, nan=0.0, posinf=0.0, neginf=0.0)

    print("\n[2] Normalized Laplacian Score...")
    W = kneighbors_graph(X_np, n_neighbors=5, mode='connectivity', include_self=False)
    W = 0.5 * (W + W.T)
    W = W.toarray()

    # --- CORRECTED PART ---
    D_sum = W.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D_sum + 1e-9)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    L = np.eye(W.shape[0]) - D_inv_sqrt_mat @ W @ D_inv_sqrt_mat
    D = np.diag(D_sum)  # for formula
    # ----------------------

    lap_scores = []
    for i in range(X_np.shape[1]):
        f = X_np[:, i] - X_np[:, i].mean()
        num = f.T @ L @ f
        den = f.T @ f + 1e-9  # normalized den
        lap_scores.append(num / den)

    lap_scores = np.array(lap_scores)
    # invert: small lap = good
    lap_imp = 1 - (lap_scores - lap_scores.min())/(lap_scores.max()-lap_scores.min()+1e-9)

    print("[3] SPEC Score...")
    eig_vals, eig_vecs = np.linalg.eigh(L)
    F = eig_vecs[:, 1:3]
    spec_scores = []
    for i in range(X_np.shape[1]):
        f = X_np[:, i]
        if np.std(f)==0:
            spec_scores.append(0)
            continue
        c1 = np.corrcoef(f, F[:,0])[0,1] if not np.isnan(np.corrcoef(f, F[:,0])[0,1]) else 0
        c2 = np.corrcoef(f, F[:,1])[0,1] if not np.isnan(np.corrcoef(f, F[:,1])[0,1]) else 0
        spec_scores.append(abs(c1) + abs(c2))

    spec_scores = np.array(spec_scores)
    spec_norm = (spec_scores - spec_scores.min())/(spec_scores.max()-spec_scores.min()+1e-9)

    print("[4] Fusion 60% Lap + 40% SPEC...")
    final_score = 0.6*lap_imp + 0.4*spec_norm
    sorted_idx = np.argsort(final_score)[::-1]

    # --- STATISTICAL PROOF CORRECTED ---
    print(f"\n--- STATISTICAL ACCURACY PROOF - SEED {SEED} ---")
    print(f"Laplacian scores mean: {np.mean(lap_scores):.4f}, std: {np.std(lap_scores):.4f}")
    print(f"SPEC scores mean: {np.mean(spec_scores):.4f}, std: {np.std(spec_scores):.4f}")
    print(f"Fusion scores mean: {np.mean(final_score):.4f}, std: {np.std(final_score):.4f}")
    gap = final_score[sorted_idx[TOP_K-1]] - final_score[sorted_idx[TOP_K]]
    print(f"Top 20 fusion gap vs 21st: {gap:.4f}")

    selected = [feature_names[i] for i in sorted_idx[:TOP_K]]
    print(f"\nSelected {TOP_K} for Seed {SEED}:")
    for i, idx in enumerate(sorted_idx[:TOP_K], 1):
        print(f"  {i}. {feature_names[idx]} - {final_score[idx]:.4f}")

    joblib.dump(selected, OUTPUT_PKL)
    df_final = X_full[selected].copy()
    # Save back both label + Sub_Category
    for col in target_cols:
        df_final[col] = y_full[col].values
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDONE: {OUTPUT_CSV}")

print("\n\nALL 5 SEEDS DONE!")