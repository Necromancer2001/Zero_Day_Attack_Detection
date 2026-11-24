# hippo_feature_selector_5_seeds.py - For All 5 Seeds
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==================== LOOP FOR SEED 1 TO 5 ====================
for SEED in [1,2,3,4,5]:
    print(f"\n\n########## HIPPO - SEED {SEED} ##########")

    # ============ CONFIG ============
    INPUT = f"data/ciciot_minmax_seed{SEED}.csv"
    OUTPUT = f"data/ciciot_hippo_selected_seed{SEED}.csv"
    # ================================

    print(f"[1] Loading {INPUT}")
    df_full = pd.read_csv(INPUT)
    print(f"    Shape: {df_full.shape}")

    # Take small sample for fast test
    df = df_full.sample(n=20000, random_state=42) # 20k only

    # FIX: Drop BOTH label and Sub_Category from features
    # Your minmax file = [48 features + label + Sub_Category]
    # So last 2 columns are targets, not features
    target_cols = ['label', 'Sub_Category']
    # Fallback if names are different - take last 2 cols
    if not all(c in df.columns for c in target_cols):
        target_cols = list(df.columns[-2:])
    
    print(f"    Target cols to drop: {target_cols}")

    # For fitness, use Main label (first of the two)
    label_col = target_cols[0]
    X = df.drop(columns=target_cols).values
    y = df[label_col].values
    feature_names = df.drop(columns=target_cols).columns.tolist()
    n_feat = len(feature_names)
    print(f"    Features for Hippo: {n_feat}")

    def fitness(vec):
        if np.sum(vec) == 0: return 1
        idx = np.where(vec==1)[0]
        X_train, X_test, y_train, y_test = train_test_split(X[:,idx], y, test_size=0.3, random_state=42)
        m = RandomForestClassifier(n_estimators=10, n_jobs=-1).fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        return 0.99*(1-acc) + 0.01*(len(idx)/n_feat)

    print(f"[2] Starting Hippo on seed {SEED}: {n_feat} features")
    herd = np.random.randint(0,2,size=(10, n_feat)) # 10 hippos for speed
    fit = [fitness(h) for h in herd]
    best = herd[np.argmin(fit)].copy()
    best_f = min(fit)

    for it in range(10): # 10 iterations for speed
        for i in range(10):
            new = best.copy()
            for j in range(n_feat):
                if random.random() < 0.5: new[j] = random.randint(0,1)
            f_new = fitness(new)
            if f_new < fit[i]:
                herd[i]=new; fit[i]=f_new
                if f_new < best_f: best=new.copy(); best_f=f_new
        print(f"Iter {it+1}/10 - Best: {best_f:.4f} Features: {np.sum(best)}")

    sel = [feature_names[i] for i in np.where(best==1)[0]]
    print(f"Selected {len(sel)} features: {sel}")

    # Save FULL data (not just 20k sample) with selected features + targets
    df_full[sel + target_cols].to_csv(OUTPUT, index=False)
    print(f"DONE: {OUTPUT} | Selected: {len(sel)} features")

print("\n\nALL 5 SEEDS DONE!")