"""k-Nearest-Neighbors classifier, from scratch (lecture 10).

Predicts by majority vote among the k closest training points, using
vectorized Euclidean distances.
"""

import numpy as np


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
        self.n_classes_ = 0

    def fit(self, X, y):
        # k-NN has no real training step: just remember the data.
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        self.n_classes_ = int(self.y_train_.max()) + 1
        return self

    def _distances(self, X):
        # Euclidean distance from each query row to every training row.
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b (clipped at 0 for safety).
        query_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        train_sq = np.sum(self.X_train_ ** 2, axis=1).reshape(1, -1)
        dot = X @ self.X_train_.T
        sq_dist = np.maximum(query_sq + train_sq - 2 * dot, 0.0)
        return np.sqrt(sq_dist)

    def predict_proba(self, X):
        # Class probability = share of votes among the k nearest neighbors.
        X = np.asarray(X, dtype=float)
        dist = self._distances(X)
        nearest = np.argpartition(dist, kth=min(self.k, dist.shape[1] - 1), axis=1)[:, : self.k]

        proba = np.zeros((X.shape[0], self.n_classes_))
        for i in range(X.shape[0]):
            labels = self.y_train_[nearest[i]]
            for label in range(self.n_classes_):
                proba[i, label] = np.sum(labels == label)
            proba[i] /= proba[i].sum()
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
