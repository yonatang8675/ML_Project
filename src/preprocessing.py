"""
Preprocessing utilities implemented from scratch (NumPy only).

Includes:
- stratified train/test split
- StandardScaler (z-score standardization)
- one-hot encoding
- k-fold cross-validation index generator
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Train / test split
# ----------------------------------------------------------------------------
def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split X, y into train/test sets.

    If ``stratify`` is True, the class proportions are preserved in both splits.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if stratify:
        test_idx_parts = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            n_test = int(round(test_size * len(cls_idx)))
            test_idx_parts.append(cls_idx[:n_test])
        test_idx = np.concatenate(test_idx_parts)
    else:
        perm = rng.permutation(n)
        n_test = int(round(test_size * n))
        test_idx = perm[:n_test]

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------------
# Standardization
# ----------------------------------------------------------------------------
class StandardScaler:
    """Z-score standardization: (x - mean) / std, fit on training data only."""

    def __init__(self, columns: Optional[list[int]] = None):
        # If columns is None, scale all columns; otherwise scale a subset.
        self.columns = columns
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        cols = self._cols(X)
        self.mean_ = X[:, cols].mean(axis=0)
        std = X[:, cols].std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero for constant columns
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        cols = self._cols(X)
        Xt = X.astype(float).copy()
        Xt[:, cols] = (Xt[:, cols] - self.mean_) / self.std_
        return Xt

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _cols(self, X: np.ndarray) -> list[int]:
        if self.columns is None:
            return list(range(X.shape[1]))
        return self.columns


# ----------------------------------------------------------------------------
# One-hot encoding
# ----------------------------------------------------------------------------
def one_hot_encode(
    X: np.ndarray,
    feature_names: list[str],
    columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    One-hot encode the given categorical columns of X.

    Returns the new matrix and the new list of feature names. Categories are
    inferred from the data; encoding is deterministic (sorted category order).
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    encode_idx = [name_to_idx[c] for c in columns]

    out_cols: list[np.ndarray] = []
    out_names: list[str] = []

    for i, name in enumerate(feature_names):
        if i not in encode_idx:
            out_cols.append(X[:, i : i + 1])
            out_names.append(name)
        else:
            categories = np.unique(X[:, i])
            for cat in categories:
                out_cols.append((X[:, i] == cat).astype(float).reshape(-1, 1))
                out_names.append(f"{name}={int(cat) if float(cat).is_integer() else cat}")

    return np.hstack(out_cols), out_names


# ----------------------------------------------------------------------------
# K-fold cross-validation indices
# ----------------------------------------------------------------------------
def kfold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    stratify: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) tuples for k-fold CV.

    Stratified folds preserve class balance in each fold.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    if stratify:
        # Assign each sample a fold id, balanced within each class.
        fold_id = np.empty(n, dtype=int)
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            fold_id[cls_idx] = np.arange(len(cls_idx)) % n_splits
    else:
        perm = rng.permutation(n)
        fold_id = np.empty(n, dtype=int)
        fold_id[perm] = np.arange(n) % n_splits

    for k in range(n_splits):
        val_idx = np.where(fold_id == k)[0]
        train_idx = np.where(fold_id != k)[0]
        yield train_idx, val_idx
