"""From-scratch preprocessing helpers: train/test split, standardization, k-fold."""

import numpy as np


def train_test_split(X, y, test_size=0.2, seed=42, stratify=True):
    # Split X, y into train/test. Stratify keeps the class ratio in both parts.
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]

    if stratify:
        # Take a test slice from each class separately.
        test_parts = []
        for label in np.unique(y):
            label_idx = np.where(y == label)[0]
            rng.shuffle(label_idx)
            n_test = int(round(test_size * len(label_idx)))
            test_parts.append(label_idx[:n_test])
        test_idx = np.concatenate(test_parts)
    else:
        perm = rng.permutation(n_samples)
        n_test = int(round(test_size * n_samples))
        test_idx = perm[:n_test]

    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[test_idx] = True

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    return X_train, X_test, y_train, y_test


class StandardScaler:
    """Z-score standardization: (x - mean) / std, fit on the training data only."""

    def __init__(self, columns=None):
        # columns=None scales every column; otherwise scale the given subset.
        self.columns = columns
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # Learn the mean and std of each scaled column.
        cols = self._cols(X)
        self.mean_ = X[:, cols].mean(axis=0)
        std = X[:, cols].std(axis=0)
        std[std == 0] = 1.0  # avoid dividing by zero on constant columns
        self.std_ = std
        return self

    def transform(self, X):
        # Apply the learned standardization.
        cols = self._cols(X)
        X_scaled = X.astype(float).copy()
        X_scaled[:, cols] = (X_scaled[:, cols] - self.mean_) / self.std_
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _cols(self, X):
        # Column indices to scale.
        if self.columns is None:
            return list(range(X.shape[1]))
        return self.columns


def kfold_indices(y, n_splits=5, seed=42, stratify=True):
    # Yield (train_idx, val_idx) for each fold; stratified folds keep class balance.
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    fold_id = np.empty(n_samples, dtype=int)

    if stratify:
        # Spread each class evenly across the folds.
        for label in np.unique(y):
            label_idx = np.where(y == label)[0]
            rng.shuffle(label_idx)
            fold_id[label_idx] = np.arange(len(label_idx)) % n_splits
    else:
        perm = rng.permutation(n_samples)
        fold_id[perm] = np.arange(n_samples) % n_splits

    for fold in range(n_splits):
        val_idx = np.where(fold_id == fold)[0]
        train_idx = np.where(fold_id != fold)[0]
        yield train_idx, val_idx
