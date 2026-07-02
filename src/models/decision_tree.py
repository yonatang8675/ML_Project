"""
Decision Tree classifier (CART) implemented from scratch.

Reference: lecture 11 (Decision trees). Splits are chosen greedily to maximize
the impurity decrease, using either Gini impurity or entropy (information gain).
Supports `max_depth`, `min_samples_split`, `min_samples_leaf` for regularization,
and exposes impurity-based feature importances.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n_samples")

    def __init__(self):
        self.feature: Optional[int] = None      # split feature index
        self.threshold: Optional[float] = None  # split threshold (x <= t -> left)
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.value: Optional[np.ndarray] = None  # class probability at a leaf
        self.n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


def _gini(y: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(y, minlength=n_classes)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def _entropy(y: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(y, minlength=n_classes)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root_: Optional[_Node] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self._importances = np.zeros(self.n_features_, dtype=float)
        self.root_ = self._build(X, y, depth=0)

        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0 else self._importances
        )
        return self

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            return _gini(y, self.n_classes_)
        return _entropy(y, self.n_classes_)

    def _leaf(self, y: np.ndarray) -> _Node:
        node = _Node()
        counts = np.bincount(y, minlength=self.n_classes_)
        node.value = counts / counts.sum()
        node.n_samples = len(y)
        return node

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = len(y)

        # Stopping conditions -> leaf.
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return self._leaf(y)

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:
            return self._leaf(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Accumulate impurity-decrease weighted by samples (feature importance).
        self._importances[feature] += n_samples * gain

        node = _Node()
        node.feature = feature
        node.threshold = threshold
        node.n_samples = n_samples
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples = len(y)
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for feature in range(self.n_features_):
            values = X[:, feature]
            # Candidate thresholds: midpoints between consecutive unique values.
            uniq = np.unique(values)
            if uniq.size == 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for t in thresholds:
                left_mask = values <= t
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                imp_left = self._impurity(y[left_mask])
                imp_right = self._impurity(y[~left_mask])
                child_impurity = (n_left * imp_left + n_right * imp_right) / n_samples
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(t)

        return best_feature, best_threshold, best_gain

    # --------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty((X.shape[0], self.n_classes_), dtype=float)
        for i, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.value
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_depth(self) -> int:
        def _depth(node: Optional[_Node]) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root_)
