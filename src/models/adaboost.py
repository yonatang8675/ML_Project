"""AdaBoost (discrete SAMME) with decision-stump weak learners, from scratch.

Lecture 8. Labels are handled internally as {-1, +1}. Each round fits the stump
with the smallest weighted error, sets alpha = 0.5 * ln((1-err)/err), and
re-weights the samples. Final prediction is sign(sum_t alpha_t * h_t(x)).
"""

import numpy as np


class DecisionStump:
    """One-level tree: a threshold on a single feature, with a polarity."""

    def __init__(self):
        self.feature = 0
        self.threshold = 0.0
        self.polarity = 1   # +1: predict +1 when x <= threshold; -1: flipped
        self.alpha = 0.0    # weight of this stump in the ensemble

    def predict(self, X):
        # +1 / -1 prediction for each row.
        values = X[:, self.feature]
        pred = np.ones(X.shape[0])
        if self.polarity == 1:
            pred[values > self.threshold] = -1
        else:
            pred[values <= self.threshold] = -1
        return pred


class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps_ = []
        self.train_errors_ = []   # ensemble training error after each round
        self.n_features_ = 0

    def fit(self, X, y):
        # Boosting loop: add one stump per round and re-weight the samples.
        X = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        y = np.where(labels == 1, 1, -1)  # map {0,1} -> {-1,+1}

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        sample_weights = np.full(n_samples, 1.0 / n_samples)
        self.stumps_ = []
        self.train_errors_ = []

        running_score = np.zeros(n_samples)  # weighted vote so far, for tracking
        for _ in range(self.n_estimators):
            stump, error, stump_pred = self._best_stump(X, y, sample_weights)

            # Clip the error away from 0/1 before taking the log.
            error = float(np.clip(error, 1e-10, 1 - 1e-10))
            stump.alpha = 0.5 * np.log((1 - error) / error)

            # Increase the weight of the misclassified samples.
            sample_weights = sample_weights * np.exp(-stump.alpha * y * stump_pred)
            sample_weights = sample_weights / sample_weights.sum()
            self.stumps_.append(stump)

            running_score += stump.alpha * stump_pred
            ensemble_pred = np.sign(running_score)
            ensemble_pred[ensemble_pred == 0] = 1
            self.train_errors_.append(float(np.mean(ensemble_pred != y)))
        return self

    def _best_stump(self, X, y, sample_weights):
        # Pick the feature/threshold/polarity with the lowest weighted error.
        n_samples, n_features = X.shape
        best_error = np.inf
        best_stump = DecisionStump()
        best_pred = np.ones(n_samples)

        for feature in range(n_features):
            values = X[:, feature]
            unique_values = np.unique(values)
            if unique_values.size == 1:
                thresholds = unique_values
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                for polarity in (1, -1):
                    pred = np.ones(n_samples)
                    if polarity == 1:
                        pred[values > threshold] = -1
                    else:
                        pred[values <= threshold] = -1

                    error = float(np.sum(sample_weights[pred != y]))
                    if error < best_error:
                        best_error = error
                        best_stump.feature = feature
                        best_stump.threshold = float(threshold)
                        best_stump.polarity = polarity
                        best_pred = pred

        return best_stump, best_error, best_pred

    def decision_function(self, X):
        # Weighted vote of all the stumps.
        X = np.asarray(X, dtype=float)
        scores = np.zeros(X.shape[0])
        for stump in self.stumps_:
            scores += stump.alpha * stump.predict(X)
        return scores

    def predict(self, X):
        # Sign of the weighted vote, mapped back to {0, 1}.
        scores = self.decision_function(X)
        signs = np.sign(scores)
        signs[signs == 0] = 1
        return ((signs + 1) // 2).astype(int)

    def predict_proba(self, X):
        # Squash the margin through a logistic to get pseudo-probabilities.
        scores = self.decision_function(X)
        prob_positive = 1.0 / (1.0 + np.exp(-2.0 * scores))
        return np.column_stack([1 - prob_positive, prob_positive])

    @property
    def feature_importances_(self):
        # Sum of |alpha| per feature, normalized to sum to 1.
        if not self.stumps_:
            return np.array([])
        n_features = self.n_features_ or (max(stump.feature for stump in self.stumps_) + 1)
        importances = np.zeros(n_features)
        for stump in self.stumps_:
            importances[stump.feature] += abs(stump.alpha)
        total = importances.sum()
        return importances / total if total > 0 else importances
