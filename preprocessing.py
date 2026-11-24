import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==================== LOOP FOR SEED 1 TO 5 ====================
for RANDOM_SEED in [1,2,3,4,5]:
    print(f"\n\n########## PROCESSING SEED {RANDOM_SEED} ##########")

    # ==================== 1. CONFIG ====================
    RAW_DATA_FILE = f"data/ciciot2023_subset_seed{RANDOM_SEED}.csv"
    INTERMEDIATE_FILE = f"data/ciciot_cleaned_seed{RANDOM_SEED}.csv"
    FINAL_OUTPUT_FILE = f"data/ciciot_minmax_seed{RANDOM_SEED}.csv"
    os.makedirs("data", exist_ok=True)

    # ==================== 2. LOAD ====================
    print("\n>>> Step 1: Loading Raw Data...")
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Source file: {RAW_DATA_FILE}")
    print(f"Initial dimensions: {df.shape}")

    # ==================== 3. CLEAN ====================
    print("\n>>> Step 2: Data Quality Checks...")
    print(f"Total null elements: {df.isnull().sum().sum()}")
    print(f"Duplicate rows found: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Dimensions after cleaning: {df.shape}")

    # ==================== 4. HIERARCHICAL LABELS ====================
    print("\n>>> Step 3: Hierarchical Categorization...")
    target_col = df.columns[-1]
    print(f"Target Feature: {target_col}")

    df[target_col] = df[target_col].astype(str).str.replace("DD0S", "DDoS", regex=False)
    df["Sub_Category"] = df[target_col].copy()

    def map_attack_category(attack_str):
        text = str(attack_str).lower()
        if "ddos" in text:
            return "DDoS"
        elif text.startswith(("dos-", "dos_")):
            return "DoS"
        elif "mirai" in text:
            return "Mirai"
        elif "brute" in text or "dictionary" in text:
            return "BruteForce"
        elif "spoof" in text or "mitm" in text:
            return "Spoofing"
        elif any(k in text for k in ["recon", "hostdiscovery", "osscan", "pingsweep", "portscan", "vulscan"]):
            return "Recon"
        elif any(k in text for k in ["backdoor", "browser", "hijack", "command", "injection", "sql", "xss", "upload"]):
            return "WebBased"
        elif "benign" in text:
            return "Benign"
        else:
            return "Other"

    df["Main_Category"] = df[target_col].apply(map_attack_category)
    print(df["Main_Category"].value_counts())
    df[target_col] = df["Main_Category"]
    df.drop(columns=["Main_Category"], inplace=True)

    # ==================== 5. ENCODING ====================
    print("\n>>> Step 4: Encoding...")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Categorical features: {categorical_cols}")

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        joblib.dump(encoder, f"data/encoder_{col}_seed{RANDOM_SEED}.pkl")
        print(f"Attribute: {col} | Classes: {len(encoder.classes_)} | {list(encoder.classes_)}")

    df.to_csv(INTERMEDIATE_FILE, index=False)
    print(f"Cleaned saved to: {INTERMEDIATE_FILE}")

    # ==================== 6. SPLIT FEATURES / TARGET ====================
    print("\n>>> Step 5: Separating Features and Target...")
    features_df = df.drop(columns=[target_col, "Sub_Category"])
    labels_df = df[[target_col]]
    sub_df = df[["Sub_Category"]]
    print(f"Total features count: {features_df.shape[1]}")
    print(f"Target shape: {labels_df.shape}")

    # ==================== 7. SCALING ====================
    print("\n>>> Step 6: Min-Max Scaling...")
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(features_df)
    scaled_features_df = pd.DataFrame(scaled_array, columns=features_df.columns, index=features_df.index)

    final_dataset = pd.concat([scaled_features_df, labels_df, sub_df], axis=1)
    joblib.dump(scaler, f"data/minmax_scaler_seed{RANDOM_SEED}.pkl")
    final_dataset.to_csv(FINAL_OUTPUT_FILE, index=False)

    print("\n>>> Pipeline Successful! <<<")
    print(f"Output: {FINAL_OUTPUT_FILE}")
    print(f"Final Shape: {final_dataset.shape}")
    print(final_dataset[target_col].value_counts())

print("\n\nALL 5 SEEDS DONE!")