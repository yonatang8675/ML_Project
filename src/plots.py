"""
Plotting helpers (matplotlib). Every function saves a PNG into ``outdir`` and
returns the saved path. Figure labels are in English (the written report is in
Hebrew, but figures use English to avoid RTL/font issues).

Includes from-scratch PCA (via SVD) and a Johnson-Lindenstrauss random
projection (lecture 15) for 2-D visualization.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe for headless / notebook execution
import matplotlib.pyplot as plt


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def _ensure_dir(outdir: Optional[str]) -> str:
    outdir = outdir or FIG_DIR
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _save(fig, outdir: Optional[str], name: str) -> str:
    outdir = _ensure_dir(outdir)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------- EDA
def plot_class_balance(y, outdir=None, name="class_balance.png") -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    classes, counts = np.unique(y, return_counts=True)
    ax.bar(["No disease (0)", "Disease (1)"][: len(classes)], counts,
           color=["#4C72B0", "#C44E52"])
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom")
    ax.set_ylabel("Count")
    ax.set_title("Target class balance")
    return _save(fig, outdir, name)


def plot_feature_histograms(df, feature_cols, target_col="target",
                            outdir=None, name="feature_histograms.png") -> str:
    n = len(feature_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).ravel()
    for ax, col in zip(axes, feature_cols):
        for cls, color in zip([0, 1], ["#4C72B0", "#C44E52"]):
            ax.hist(df.loc[df[target_col] == cls, col], bins=20, alpha=0.6,
                    label=f"target={cls}", color=color)
        ax.set_title(col)
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Feature distributions by target", y=1.02)
    return _save(fig, outdir, name)


def plot_correlation_heatmap(df, columns, outdir=None,
                             name="correlation_heatmap.png") -> str:
    corr = np.corrcoef(df[columns].to_numpy().T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation matrix")
    return _save(fig, outdir, name)


# --------------------------------------------------- dimensionality reduction
def pca_2d(X: np.ndarray) -> np.ndarray:
    """Project X onto its top-2 principal components (from scratch, via SVD)."""
    Xc = X - X.mean(axis=0)
    # Standardize so features share scale before PCA.
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xc = Xc / std
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def jl_projection_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Johnson-Lindenstrauss random projection to 2-D (lecture 15).

    Uses a Gaussian random matrix scaled by 1/sqrt(k).
    """
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xc = Xc / std
    k = 2
    R = rng.normal(0, 1, size=(X.shape[1], k)) / np.sqrt(k)
    return Xc @ R


def plot_2d_projection(X, y, method="pca", outdir=None, name=None) -> str:
    if method == "pca":
        Z = pca_2d(X)
        title = "PCA projection (2-D)"
    else:
        Z = jl_projection_2d(X)
        title = "Johnson-Lindenstrauss random projection (2-D)"
    name = name or f"projection_{method}.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, color in zip([0, 1], ["#4C72B0", "#C44E52"]):
        mask = y == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], s=15, alpha=0.6, color=color,
                   label=f"target={cls}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend()
    return _save(fig, outdir, name)


# ------------------------------------------------------------------- results
def plot_confusion_matrix(cm, outdir=None, name="confusion_matrix.png",
                          title="Confusion matrix") -> str:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, outdir, name)


def plot_roc_curves(roc_data: dict, outdir=None, name="roc_curves.png") -> str:
    """roc_data: name -> (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (fpr, tpr, auc_val) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8)
    return _save(fig, outdir, name)


def plot_model_comparison(results: dict, metric="accuracy", outdir=None,
                          name=None) -> str:
    name = name or f"model_comparison_{metric}.png"
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color="#55A868")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    plt.xticks(rotation=15)
    return _save(fig, outdir, name)


def plot_feature_importances(importances: dict, feature_names, outdir=None,
                             name="feature_importances.png") -> str:
    """importances: model_name -> importance vector (same length as feature_names)."""
    models = list(importances.keys())
    n_feat = len(feature_names)
    x = np.arange(n_feat)
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(models):
        vals = np.asarray(importances[m], dtype=float)
        if vals.sum() > 0:
            vals = vals / vals.sum()  # normalize for comparability
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Normalized importance")
    ax.set_title("Feature importance across models")
    ax.legend()
    return _save(fig, outdir, name)


# ----------------------------------------------------- learning / tuning curves
def plot_curve(x, y, xlabel, ylabel, title, outdir=None, name="curve.png",
               marker="o") -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, outdir, name)


def plot_nn_loss(loss_history, outdir=None, name="nn_loss.png") -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Neural network training loss")
    ax.grid(alpha=0.3)
    return _save(fig, outdir, name)
