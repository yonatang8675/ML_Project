"""
k-Nearest-Neighbors classifier implemented from scratch.

Reference: lecture 10 (Nearest neighbor). Prediction is by (optionally
distance-weighted) majority vote among the k closest training points.
Distances are computed in a fully vectorized way with NumPy.
"""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.metric = metric
        self.weights = weights

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        self.n_classes_ = int(self.y_train_.max()) + 1
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Return an (n_query, n_train) distance matrix."""
        if self.metric == "euclidean":
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b  (numerically clipped at 0)
            aa = np.sum(X ** 2, axis=1).reshape(-1, 1)
            bb = np.sum(self.X_train_ ** 2, axis=1).reshape(1, -1)
            ab = X @ self.X_train_.T
            d2 = np.maximum(aa + bb - 2 * ab, 0.0)
            return np.sqrt(d2)
        # Manhattan distance (broadcasted).
        return np.sum(np.abs(X[:, None, :] - self.X_train_[None, :, :]), axis=2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dist = self._distances(X)
        # Indices of the k nearest neighbors for each query point.
        knn_idx = np.argpartition(dist, kth=min(self.k, dist.shape[1] - 1), axis=1)[:, : self.k]

        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i in range(X.shape[0]):
            idx = knn_idx[i]
            labels = self.y_train_[idx]
            if self.weights == "distance":
                w = 1.0 / (dist[i, idx] + 1e-12)
            else:
                w = np.ones_like(idx, dtype=float)
            for cls in range(self.n_classes_):
                proba[i, cls] = w[labels == cls].sum()
            s = proba[i].sum()
            if s > 0:
                proba[i] /= s
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
