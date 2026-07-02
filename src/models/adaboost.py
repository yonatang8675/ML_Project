"""
AdaBoost (SAMME / discrete AdaBoost) with decision-stump weak learners,
implemented from scratch.

Reference: lecture 8 (Adaboost). Labels are handled internally in {-1, +1}.
Each round fits the stump that minimizes the weighted error, computes
alpha = 0.5 * ln((1 - err) / err), and re-weights the samples. The final
prediction is sign(sum_t alpha_t * h_t(x)).
"""

from __future__ import annotations

import numpy as np


class DecisionStump:
    """A one-level decision tree: threshold on a single feature with a polarity."""

    __slots__ = ("feature", "threshold", "polarity", "alpha")

    def __init__(self):
        self.feature: int = 0
        self.threshold: float = 0.0
        self.polarity: int = 1   # +1: predict +1 when x <= t ; -1: flipped
        self.alpha: float = 0.0  # weight of this stump in the ensemble

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = X[:, self.feature]
        pred = np.ones(X.shape[0])
        if self.polarity == 1:
            pred[values > self.threshold] = -1
        else:
            pred[values <= self.threshold] = -1
        return pred


class AdaBoostClassifier:
    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.stumps_: list[DecisionStump] = []
        self.train_errors_: list[float] = []  # ensemble training error per round
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        X = np.asarray(X, dtype=float)
        y01 = np.asarray(y, dtype=int)
        y = np.where(y01 == 1, 1, -1)  # map {0,1} -> {-1,+1}

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        w = np.full(n_samples, 1.0 / n_samples)
        self.stumps_ = []
        self.train_errors_ = []

        agg = np.zeros(n_samples)  # running weighted vote, for tracking train error

        for _ in range(self.n_estimators):
            stump, err, pred = self._best_stump(X, y, w)

            # Numerical guards on the weighted error.
            err = float(np.clip(err, 1e-10, 1 - 1e-10))
            stump.alpha = 0.5 * np.log((1 - err) / err)

            # Re-weight: increase weight of misclassified samples.
            w = w * np.exp(-stump.alpha * y * pred)
            w = w / w.sum()

            self.stumps_.append(stump)

            agg += stump.alpha * pred
            ensemble_pred = np.sign(agg)
            ensemble_pred[ensemble_pred == 0] = 1
            self.train_errors_.append(float(np.mean(ensemble_pred != y)))

        return self

    def _best_stump(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        n_samples, n_features = X.shape
        best_err = np.inf
        best_stump = DecisionStump()
        best_pred = np.ones(n_samples)

        for feature in range(n_features):
            values = X[:, feature]
            uniq = np.unique(values)
            if uniq.size == 1:
                candidate_thresholds = uniq
            else:
                candidate_thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for t in candidate_thresholds:
                for polarity in (1, -1):
                    pred = np.ones(n_samples)
                    if polarity == 1:
                        pred[values > t] = -1
                    else:
                        pred[values <= t] = -1

                    err = float(np.sum(w[pred != y]))
                    if err < best_err:
                        best_err = err
                        best_stump.feature = feature
                        best_stump.threshold = float(t)
                        best_stump.polarity = polarity
                        best_pred = pred

        return best_stump, best_err, best_pred

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        agg = np.zeros(X.shape[0])
        for stump in self.stumps_:
            agg += stump.alpha * stump.predict(X)
        return agg

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        out = np.sign(scores)
        out[out == 0] = 1
        return ((out + 1) // 2).astype(int)  # map {-1,+1} -> {0,1}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Squash the margin through a logistic to get pseudo-probabilities.
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-2.0 * scores))
        return np.column_stack([1 - p1, p1])

    @property
    def feature_importances_(self) -> np.ndarray:
        """Sum of stump alphas per feature, normalized to sum to 1."""
        if not self.stumps_:
            return np.array([])
        n_features = self.n_features_ or (max(s.feature for s in self.stumps_) + 1)
        imp = np.zeros(n_features)
        for s in self.stumps_:
            imp[s.feature] += abs(s.alpha)
        total = imp.sum()
        return imp / total if total > 0 else imp
