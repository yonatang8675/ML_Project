import numpy as np

from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Bagging ensemble of decision trees with per-split random feature subsets."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features   # features considered per split (decorrelates trees)
        self.bootstrap = bootstrap         # sample rows with replacement for each tree
        self.seed = seed

        self.trees_ = []
        self.n_classes_ = 0
        self.n_features_ = 0

    def fit(self, X, y):
        # Grow n_estimators trees, each on a bootstrap sample with a random feature subspace.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples = X.shape[0]
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]

        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n_samples, size=n_samples)
            else:
                idx = np.arange(n_samples)

            # Give each tree its own seed so the per-split feature subsets differ.
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        # Soft vote: average the class probabilities predicted by every tree.
        X = np.asarray(X, dtype=float)
        proba = np.zeros((X.shape[0], self.n_classes_))
        for tree in self.trees_:
            proba += tree.predict_proba(X)
        return proba / len(self.trees_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def feature_importances_(self):
        # Mean of the trees' importances, normalized to sum to 1.
        if not self.trees_:
            return np.array([])
        importances = np.zeros(self.n_features_)
        for tree in self.trees_:
            importances += tree.feature_importances_
        total = importances.sum()
        return importances / total if total > 0 else importances
