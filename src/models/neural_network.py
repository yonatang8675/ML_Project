"""Feed-forward neural network (MLP), from scratch.

Generalizes the Perceptron (lecture 6) and logistic regression (lecture 14):
linear layers with non-linear activations, trained by mini-batch gradient
descent on the binary cross-entropy loss. gradient_check verifies back-prop.
"""

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
                 epochs=200, batch_size=32, l2=1e-4, seed=42, verbose=False):
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

        self.weights_ = []
        self.biases_ = []
        self.loss_history_ = []

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
        # Mini-batch gradient descent, recording the loss after each epoch.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._init_params(X.shape[1])
        rng = np.random.default_rng(self.seed)
        n_samples = X.shape[0]
        self.loss_history_ = []

        for epoch in range(self.epochs):
            # Shuffle the rows at the start of every epoch.
            perm = rng.permutation(n_samples)
            X_shuffled, y_shuffled = X[perm], y[perm]
            for start in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[start:start + self.batch_size]
                y_batch = y_shuffled[start:start + self.batch_size]
                activations, pre_activations = self._forward(X_batch)
                grads_w, grads_b = self._backward(activations, pre_activations, y_batch)
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.lr * grads_w[i]
                    self.biases_[i] -= self.lr * grads_b[i]

            activations, _ = self._forward(X)
            loss = self._loss(y, activations[-1].ravel())
            self.loss_history_.append(loss)
            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch:4d}  loss={loss:.4f}")
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
