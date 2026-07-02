"""
Classification metrics implemented from scratch (NumPy only).

All functions assume binary labels in {0, 1} unless stated otherwise.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Return the 2x2 confusion matrix:

        [[TN, FP],
         [FN, TP]]
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ROC curve.

    Returns (fpr, tpr, thresholds) sorted by decreasing threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    # Sort scores in descending order.
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        # Degenerate: only one class present.
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    # Cumulative TP / FP as the threshold sweeps down.
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    # Keep one point per distinct score value.
    distinct = np.where(np.diff(y_score) != 0)[0]
    idx = np.r_[distinct, len(y_score) - 1]

    tpr = tps[idx] / P
    fpr = fps[idx] / N
    thresholds = y_score[idx]

    # Prepend the (0, 0) origin point.
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under a curve via the trapezoidal rule (expects increasing fpr)."""
    return float(np.trapezoid(tpr, fpr))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bundle the common metrics into a dict (handy for tables)."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
