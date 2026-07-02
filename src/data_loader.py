"""
Data loading and cleaning for the Heart Disease dataset.

The dataset is the Kaggle "johnsmith88/heart-disease-dataset" (heart.csv).
It is the classic UCI Cleveland heart-disease data (303 patients) that was
*up-sampled with duplicates* to 1025 rows. A naive random train/test split
therefore leaks identical rows between train and test, which massively inflates
accuracy. We expose a `dedup` option (default True) to handle this correctly.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Column metadata
# ----------------------------------------------------------------------------
TARGET = "target"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# Continuous features (benefit from standardization).
CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Discrete / categorical features.
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Multi-valued categoricals that are good candidates for one-hot encoding.
ONEHOT_COLUMNS = ["cp", "restecg", "slope", "ca", "thal"]

# Human-readable descriptions (used by the EDA / report).
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


def download_dataset() -> str:
    """Download the dataset via kagglehub and return the path to its folder."""
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    return path


def _resolve_csv_path(local_csv: Optional[str]) -> str:
    """Return a path to heart.csv, downloading via kagglehub if needed."""
    if local_csv is not None and os.path.exists(local_csv):
        return local_csv

    folder = download_dataset()
    candidate = os.path.join(folder, "heart.csv")
    if os.path.exists(candidate):
        return candidate

    # Fall back to the first .csv found in the downloaded folder.
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".csv"):
                return os.path.join(root, name)

    raise FileNotFoundError(f"Could not locate heart.csv under {folder!r}")


def load_data(
    dedup: bool = True,
    local_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the Heart Disease dataset as a cleaned DataFrame.

    Parameters
    ----------
    dedup : bool
        If True (default), drop exact duplicate rows. This is essential: the
        Kaggle version contains 723 duplicate rows (1025 -> 302 unique). Keeping
        duplicates causes train/test leakage and unrealistically high accuracy.
    local_csv : str or None
        Optional path to a local heart.csv. If None, the file is downloaded.

    Returns
    -------
    pandas.DataFrame with columns FEATURE_COLUMNS + [TARGET].
    """
    csv_path = _resolve_csv_path(local_csv)
    df = pd.read_csv(csv_path)

    # Basic sanity: ensure expected columns are present.
    expected = set(FEATURE_COLUMNS + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = df[FEATURE_COLUMNS + [TARGET]].copy()

    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def get_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split a DataFrame into feature matrix X, label vector y, and feature names."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    return X, y, list(FEATURE_COLUMNS)


def dataset_summary(local_csv: Optional[str] = None) -> dict:
    """
    Return a summary dict comparing the raw vs de-duplicated dataset.

    Useful for the EDA / "challenges" section of the report.
    """
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
    summary = dataset_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
