import numpy as np


def _rbf_kernel(A, B, gamma):
    # Pairwise RBF kernel exp(-gamma * ||a - b||^2) for every row pair of A and B.
    a_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
    b_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
    sq_dist = np.maximum(a_sq + b_sq - 2 * A @ B.T, 0.0)
    return np.exp(-gamma * sq_dist)


def _linear_kernel(A, B):
    return A @ B.T


class SVMClassifier:
    """Soft-margin kernel SVM (binary) trained with a simplified SMO solver."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", tol=1e-3,
                 max_passes=5, max_iter=1000, seed=42):
        if kernel not in ("rbf", "linear"):
            raise ValueError("kernel must be 'rbf' or 'linear'")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes   # consecutive passes with no alpha change before stopping
        self.max_iter = max_iter       # hard cap on the number of passes
        self.seed = seed

        self.b_ = 0.0
        self.support_X_ = None
        self.support_y_ = None
        self.support_alpha_ = None
        self.gamma_ = None
        self.n_features_ = 0

    def _resolve_gamma(self, X):
        # "scale" -> 1/(n_features * var(X)); "auto" -> 1/n_features; otherwise numeric.
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                var = X.var()
                return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
            if self.gamma == "auto":
                return 1.0 / X.shape[1]
            raise ValueError("gamma must be 'scale', 'auto', or a number")
        return float(self.gamma)

    def _kernel(self, A, B):
        if self.kernel == "rbf":
            return _rbf_kernel(A, B, self.gamma_)
        return _linear_kernel(A, B)

    def fit(self, X, y):
        # Simplified SMO: repeatedly optimize pairs of alphas until they stop changing.
        X = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        y = np.where(labels == 1, 1.0, -1.0)  # map {0,1} -> {-1,+1}

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.gamma_ = self._resolve_gamma(X)

        K = self._kernel(X, X)                # precompute the full kernel matrix
        alpha = np.zeros(n_samples)
        b = 0.0
        rng = np.random.default_rng(self.seed)

        passes = 0
        iters = 0
        while passes < self.max_passes and iters < self.max_iter:
            iters += 1
            num_changed = 0
            for i in range(n_samples):
                E_i = np.sum(alpha * y * K[:, i]) + b - y[i]
                # Only act on samples that violate the KKT conditions.
                if (y[i] * E_i < -self.tol and alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and alpha[i] > 0):
                    j = i
                    while j == i:
                        j = int(rng.integers(0, n_samples))
                    E_j = np.sum(alpha * y * K[:, j]) + b - y[j]

                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                    # Box constraints that keep 0 <= alpha <= C on both updated alphas.
                    if y[i] != y[j]:
                        L = max(0.0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0.0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] = alpha_j_old - y[j] * (E_i - E_j) / eta
                    alpha[j] = min(H, max(L, alpha[j]))
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])

                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] \
                        - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] \
                        - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

        # Keep only the support vectors (alpha > 0) for prediction.
        sv = alpha > 1e-8
        self.b_ = b
        self.support_X_ = X[sv]
        self.support_y_ = y[sv]
        self.support_alpha_ = alpha[sv]
        return self

    def decision_function(self, X):
        # Signed distance to the margin: sum_i alpha_i y_i K(x_i, x) + b.
        X = np.asarray(X, dtype=float)
        if self.support_X_ is None or len(self.support_X_) == 0:
            return np.full(X.shape[0], self.b_)
        K = self._kernel(self.support_X_, X)   # (n_support, n_query)
        return (self.support_alpha_ * self.support_y_) @ K + self.b_

    def predict(self, X):
        # Sign of the margin, mapped back to {0, 1}.
        signs = np.sign(self.decision_function(X))
        signs[signs == 0] = 1
        return ((signs + 1) // 2).astype(int)

    def predict_proba(self, X):
        # Squash the margin through a logistic to get pseudo-probabilities.
        scores = self.decision_function(X)
        prob_positive = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1 - prob_positive, prob_positive])
