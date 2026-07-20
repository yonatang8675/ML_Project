import numpy as np


def _sigmoid(z):
    # Numerically stable logistic function.
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _relu(z):
    return np.maximum(0.0, z)


def _relu_grad(z):
    return (z > 0).astype(z.dtype)


def _tanh(z):
    return np.tanh(z)


def _tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "tanh": (_tanh, _tanh_grad),
}


class NeuralNetwork:
    """Binary-classifier MLP trained with mini-batch gradient descent."""

    def __init__(self, hidden_layers=(16,), activation="relu", lr=0.05,
                 epochs=200, batch_size=32, l2=1e-4, seed=42, verbose=False,
                 patience=None, val_fraction=0.1):
        if activation not in ACTIVATIONS:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose
        self.patience = patience          # None = disabled; int = epochs to wait
        self.val_fraction = val_fraction  # fraction of X used for early-stop val set

        self.weights_ = []
        self.biases_ = []
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.best_epoch_ = None

    def _init_params(self, n_features):
        # He init for ReLU, Xavier for tanh.
        rng = np.random.default_rng(self.seed)
        sizes = [n_features, *self.hidden_layers, 1]
        self.weights_, self.biases_ = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            if self.activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights_.append(rng.normal(0, scale, size=(sizes[i], sizes[i + 1])))
            self.biases_.append(np.zeros((1, sizes[i + 1])))

    def _forward(self, X):
        # Forward pass; keep activations and pre-activations for back-prop.
        activate, _ = ACTIVATIONS[self.activation]
        activations = [X]
        pre_activations = []
        current = X
        n_layers = len(self.weights_)
        for i in range(n_layers):
            z = current @ self.weights_[i] + self.biases_[i]
            pre_activations.append(z)
            # Hidden layers use the chosen activation; the output layer uses sigmoid.
            if i < n_layers - 1:
                current = activate(z)
            else:
                current = _sigmoid(z)
            activations.append(current)
        return activations, pre_activations

    def _loss(self, y_true, y_prob):
        # Binary cross-entropy plus an L2 weight penalty.
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        bce = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        l2_term = 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.weights_)
        return float(bce + l2_term / max(len(y_true), 1))

    def _backward(self, activations, pre_activations, y_true):
        # Back-propagate the loss gradient through every layer.
        _, activation_grad = ACTIVATIONS[self.activation]
        n_samples = y_true.shape[0]
        grads_w = [None] * len(self.weights_)
        grads_b = [None] * len(self.biases_)

        # Output layer: dL/dz = (a - y) for sigmoid + BCE.
        y_true = y_true.reshape(-1, 1)
        delta = (activations[-1] - y_true) / n_samples

        for i in reversed(range(len(self.weights_))):
            prev_activation = activations[i]
            grads_w[i] = prev_activation.T @ delta + self.l2 * self.weights_[i] / n_samples
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights_[i].T) * activation_grad(pre_activations[i - 1])
        return grads_w, grads_b

    def fit(self, X, y):
        # Mini-batch gradient descent with optional early stopping.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)

        # Split off a validation set for early stopping if patience is set.
        if self.patience is not None:
            n_val = max(1, int(len(X) * self.val_fraction))
            perm0 = rng.permutation(len(X))
            val_idx, train_idx = perm0[:n_val], perm0[n_val:]
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_tr, y_tr = X, y
            X_val, y_val = None, None

        self._init_params(X_tr.shape[1])
        n_samples = X_tr.shape[0]
        self.loss_history_ = []
        self.val_loss_history_ = []

        best_val_loss = np.inf
        best_weights = None
        best_biases = None
        no_improve = 0

        for epoch in range(self.epochs):
            # Shuffle the rows at the start of every epoch.
            perm = rng.permutation(n_samples)
            X_shuffled, y_shuffled = X_tr[perm], y_tr[perm]
            for start in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[start:start + self.batch_size]
                y_batch = y_shuffled[start:start + self.batch_size]
                activations, pre_activations = self._forward(X_batch)
                grads_w, grads_b = self._backward(activations, pre_activations, y_batch)
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.lr * grads_w[i]
                    self.biases_[i] -= self.lr * grads_b[i]

            acts, _ = self._forward(X_tr)
            loss = self._loss(y_tr, acts[-1].ravel())
            self.loss_history_.append(loss)

            if self.patience is not None:
                val_acts, _ = self._forward(X_val)
                val_loss = self._loss(y_val, val_acts[-1].ravel())
                self.val_loss_history_.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights_]
                    best_biases  = [b.copy() for b in self.biases_]
                    no_improve = 0
                    self.best_epoch_ = epoch
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if self.verbose:
                            print(f"Early stop at epoch {epoch} (best={self.best_epoch_})")
                        break

            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch:4d}  loss={loss:.4f}")

        # Restore the weights from the best epoch when early stopping is used.
        if self.patience is not None and best_weights is not None:
            self.weights_ = best_weights
            self.biases_  = best_biases
        return self

    def predict_proba(self, X):
        # Probability of the positive class for each row.
        X = np.asarray(X, dtype=float)
        activations, _ = self._forward(X)
        prob_positive = activations[-1].ravel()
        return np.column_stack([1 - prob_positive, prob_positive])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def gradient_check(self, X, y, epsilon=1e-6):
        # Compare analytic back-prop gradients with numerical ones on a small
        # batch. A correct implementation returns a relative difference ~1e-6.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._init_params(X.shape[1])

        activations, pre_activations = self._forward(X)
        grads_w, _ = self._backward(activations, pre_activations, y)

        analytic, numeric = [], []
        for layer in range(len(self.weights_)):
            W = self.weights_[layer]
            analytic_grad = grads_w[layer]
            for idx in np.ndindex(W.shape):
                original = W[idx]
                W[idx] = original + epsilon
                loss_plus = self._loss(y, self._forward(X)[0][-1].ravel())
                W[idx] = original - epsilon
                loss_minus = self._loss(y, self._forward(X)[0][-1].ravel())
                W[idx] = original
                numeric.append((loss_plus - loss_minus) / (2 * epsilon))
                analytic.append(analytic_grad[idx])

        analytic = np.array(analytic)
        numeric = np.array(numeric)
        denom = np.linalg.norm(analytic) + np.linalg.norm(numeric)
        if denom == 0:
            return 0.0
        return float(np.linalg.norm(analytic - numeric) / denom)
