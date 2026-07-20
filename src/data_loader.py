import os

import numpy as np
import pandas as pd

TARGET = "target"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# Short description of every column, used by the EDA/report.
FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1=male, 0=female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1/0)",
    "restecg": "Resting ECG result (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1/0)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)",
    "target": "Heart disease present (1) / absent (0)",
}

KAGGLE_DATASET = "johnsmith88/heart-disease-dataset"


def download_dataset():
    # Download the dataset with kagglehub and return its folder.
    import kagglehub

    return kagglehub.dataset_download(KAGGLE_DATASET)


def _find_csv(local_csv):
    # Use the given CSV if it exists, otherwise download and search for heart.csv.
    if local_csv is not None and os.path.exists(local_csv):
        return local_csv

    folder = download_dataset()
    direct = os.path.join(folder, "heart.csv")
    if os.path.exists(direct):
        return direct

    # Fall back to the first CSV in the downloaded folder.
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".csv"):
                return os.path.join(root, name)

    raise FileNotFoundError(f"Could not find heart.csv under {folder}")


def load_data(dedup=True, local_csv=None):
    # Load the dataset as a clean DataFrame; drop duplicate rows when dedup is True.
    csv_path = _find_csv(local_csv)
    df = pd.read_csv(csv_path)

    # Make sure the columns we expect are present.
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df[FEATURE_COLUMNS + [TARGET]].copy()
    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_Xy(df):
    # Split the DataFrame into feature matrix, label vector and feature names.
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    return X, y, list(FEATURE_COLUMNS)


def dataset_summary(local_csv=None):
    # Compare the raw dataset with the de-duplicated one (used in the report).
    raw = load_data(dedup=False, local_csv=local_csv)
    clean = load_data(dedup=True, local_csv=local_csv)
    return {
        "n_rows_raw": int(raw.shape[0]),
        "n_rows_unique": int(clean.shape[0]),
        "n_duplicates": int(raw.shape[0] - clean.shape[0]),
        "n_features": len(FEATURE_COLUMNS),
        "target_balance_raw": raw[TARGET].value_counts().to_dict(),
        "target_balance_unique": clean[TARGET].value_counts().to_dict(),
        "n_missing": int(raw.isna().sum().sum()),
    }


if __name__ == "__main__":
    print("Downloading + summarizing dataset...")
    for key, value in dataset_summary().items():
        print(f"  {key}: {value}")
