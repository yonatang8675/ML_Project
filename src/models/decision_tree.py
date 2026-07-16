"""Decision Tree classifier (CART), from scratch (lecture 11).

Splits are chosen greedily to maximize the impurity decrease (Gini or entropy),
with max_depth / min_samples_split / min_samples_leaf for regularization.
Also exposes impurity-based feature importances.
"""

import numpy as np


class _Node:
    def __init__(self):
        self.feature = None       # feature index used to split
        self.threshold = None     # split threshold (x <= threshold goes left)
        self.left = None
        self.right = None
        self.value = None         # class probabilities at a leaf
        self.n_samples = 0

    @property
    def is_leaf(self):
        return self.value is not None


def _gini(y, n_classes):
    # Gini impurity of a label vector.
    counts = np.bincount(y, minlength=n_classes)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def _entropy(y, n_classes):
    # Shannon entropy of a label vector.
    counts = np.bincount(y, minlength=n_classes)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root_ = None
        self.n_classes_ = 0
        self.n_features_ = 0
        self.feature_importances_ = None

    def fit(self, X, y):
        # Grow the tree, then normalize the accumulated feature importances.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self._importances = np.zeros(self.n_features_)
        self.root_ = self._build(X, y, depth=0)

        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0 else self._importances
        )
        return self

    def _impurity(self, y):
        # Impurity of the current node according to the chosen criterion.
        if self.criterion == "gini":
            return _gini(y, self.n_classes_)
        return _entropy(y, self.n_classes_)

    def _leaf(self, y):
        # Build a leaf node holding the class probabilities.
        node = _Node()
        counts = np.bincount(y, minlength=self.n_classes_)
        node.value = counts / counts.sum()
        node.n_samples = len(y)
        return node

    def _build(self, X, y, depth):
        # Recursively split until a stopping condition turns the node into a leaf.
        n_samples = len(y)
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

        # Add the sample-weighted impurity decrease to this feature's importance.
        self._importances[feature] += n_samples * gain

        node = _Node()
        node.feature = feature
        node.threshold = threshold
        node.n_samples = n_samples
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        # Try every feature/threshold and keep the one with the largest gain.
        n_samples = len(y)
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features_):
            values = X[:, feature]
            unique_values = np.unique(values)
            if unique_values.size == 1:
                continue
            # Candidate thresholds are midpoints between consecutive values.
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
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
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    def predict_proba(self, X):
        # Walk each row down to a leaf and read off its class probabilities.
        X = np.asarray(X, dtype=float)
        probabilities = np.empty((X.shape[0], self.n_classes_))
        for i, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            probabilities[i] = node.value
        return probabilities

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
