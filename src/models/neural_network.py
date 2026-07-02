"""
Feed-forward Neural Network (multi-layer perceptron) implemented from scratch.

The network is the natural generalization of the Perceptron (lecture 6) and of
Logistic Regression (lecture 14): stacked layers of linear units followed by
non-linear activations, trained by back-propagation with mini-batch gradient
descent on the binary cross-entropy loss.

A numerical gradient check (`gradient_check`) is included to verify the
correctness of the back-propagation implementation.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic function.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_grad(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


_ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "tanh": (_tanh, _tanh_grad),
}


class NeuralNetwork:
    """
    Binary classifier MLP.

    Parameters
    ----------
    hidden_layers : tuple[int, ...]
        Sizes of the hidden layers, e.g. (16,) or (32, 16).
    activation : str
        Hidden-layer activation: 'relu' or 'tanh'.
    lr : float
        Learning rate for gradient descent.
    epochs : int
        Number of passes over the training data.
    batch_size : int
        Mini-batch size.
    l2 : float
        L2 regularization strength (weight decay).
    seed : int
        RNG seed for reproducible weight initialization and shuffling.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (16,),
        activation: str = "relu",
        lr: float = 0.05,
        epochs: int = 200,
        batch_size: int = 32,
        l2: float = 1e-4,
        seed: int = 42,
        verbose: bool = False,
    ):
        if activation not in _ACTIVATIONS:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.loss_history_: list[float] = []

    # --------------------------------------------------------- initialization
    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        sizes = [n_features, *self.hidden_layers, 1]
        self.weights_, self.biases_ = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            # He init for ReLU, Xavier for tanh.
            if self.activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights_.append(rng.normal(0, scale, size=(sizes[i], sizes[i + 1])))
            self.biases_.append(np.zeros((1, sizes[i + 1])))

    # ----------------------------------------------------------------- forward
    def _forward(self, X: np.ndarray):
        act_fn, _ = _ACTIVATIONS[self.activation]
        activations = [X]
        pre_activations = []
        a = X
        n_layers = len(self.weights_)
        for i in range(n_layers):
            z = a @ self.weights_[i] + self.biases_[i]
            pre_activations.append(z)
            if i < n_layers - 1:
                a = act_fn(z)
            else:
                a = _sigmoid(z)  # output layer
            activations.append(a)
        return activations, pre_activations

    # ------------------------------------------------------------------- loss
    def _loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        bce = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        l2_term = 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.weights_)
        return float(bce + l2_term / max(len(y_true), 1))

    # --------------------------------------------------------------- backward
    def _backward(self, activations, pre_activations, y_true):
        _, act_grad = _ACTIVATIONS[self.activation]
        n = y_true.shape[0]
        grads_w = [None] * len(self.weights_)
        grads_b = [None] * len(self.biases_)

        # Output layer: dL/dz = (a - y) for sigmoid + BCE.
        y_true = y_true.reshape(-1, 1)
        delta = (activations[-1] - y_true) / n

        for i in reversed(range(len(self.weights_))):
            a_prev = activations[i]
            grads_w[i] = a_prev.T @ delta + self.l2 * self.weights_[i] / n
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights_[i].T) * act_grad(pre_activations[i - 1])
        return grads_w, grads_b

    # -------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetwork":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._init_params(X.shape[1])
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self.loss_history_ = []

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            Xs, ys = X[perm], y[perm]
            for start in range(0, n, self.batch_size):
                xb = Xs[start : start + self.batch_size]
                yb = ys[start : start + self.batch_size]
                activations, pre = self._forward(xb)
                grads_w, grads_b = self._backward(activations, pre, yb)
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.lr * grads_w[i]
                    self.biases_[i] -= self.lr * grads_b[i]

            activations, _ = self._forward(X)
            loss = self._loss(y, activations[-1].ravel())
            self.loss_history_.append(loss)
            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch:4d}  loss={loss:.4f}")
        return self

    # ---------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        activations, _ = self._forward(X)
        p1 = activations[-1].ravel()
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # -------------------------------------------------- verification utility
    def gradient_check(
        self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-6
    ) -> float:
        """
        Compare analytic back-prop gradients with numerical gradients.

        Returns the relative difference; a correct implementation yields a value
        on the order of 1e-6 or smaller. Call on a *small* batch.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._init_params(X.shape[1])

        activations, pre = self._forward(X)
        grads_w, grads_b = self._backward(activations, pre, y)

        # Save a flat copy of analytic gradients and parameters.
        analytic, numeric = [], []
        for layer in range(len(self.weights_)):
            W = self.weights_[layer]
            gW = grads_w[layer]
            for idx in np.ndindex(W.shape):
                original = W[idx]
                W[idx] = original + epsilon
                loss_plus = self._loss(y, self._forward(X)[0][-1].ravel())
                W[idx] = original - epsilon
                loss_minus = self._loss(y, self._forward(X)[0][-1].ravel())
                W[idx] = original
                numeric.append((loss_plus - loss_minus) / (2 * epsilon))
                analytic.append(gW[idx])

        analytic = np.array(analytic)
        numeric = np.array(numeric)
        denom = np.linalg.norm(analytic) + np.linalg.norm(numeric)
        if denom == 0:
            return 0.0
        return float(np.linalg.norm(analytic - numeric) / denom)
