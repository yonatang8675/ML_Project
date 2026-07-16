"""Matplotlib figures. Each function saves a PNG into outdir and returns its path.

Labels are in English (the written report is in Hebrew) to avoid RTL/font issues.
Includes a from-scratch Johnson-Lindenstrauss random projection (lecture 15).
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # works without a display / inside notebooks
import matplotlib.pyplot as plt


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def _save(fig, outdir, name):
    # Save the figure into outdir (defaults to figures/) and return its path.
    outdir = outdir or FIG_DIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(y, outdir=None, name="class_balance.png"):
    # Bar chart of how many samples fall in each target class.
    fig, ax = plt.subplots(figsize=(5, 4))
    classes, counts = np.unique(y, return_counts=True)
    ax.bar(["No disease (0)", "Disease (1)"][: len(classes)], counts,
           color=["#4C72B0", "#C44E52"])
    for i, count in enumerate(counts):
        ax.text(i, count, str(int(count)), ha="center", va="bottom")
    ax.set_ylabel("Count")
    ax.set_title("Target class balance")
    return _save(fig, outdir, name)


def plot_feature_histograms(df, feature_cols, target_col="target",
                            outdir=None, name="feature_histograms.png"):
    # One histogram per feature, split by target class.
    n_features = len(feature_cols)
    ncols = 4
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).ravel()
    for ax, col in zip(axes, feature_cols):
        for label, color in zip([0, 1], ["#4C72B0", "#C44E52"]):
            ax.hist(df.loc[df[target_col] == label, col], bins=20, alpha=0.6,
                    label=f"target={label}", color=color)
        ax.set_title(col)
        ax.legend(fontsize=7)
    # Hide the unused axes in the last row.
    for ax in axes[n_features:]:
        ax.axis("off")
    fig.suptitle("Feature distributions by target", y=1.02)
    return _save(fig, outdir, name)


def plot_correlation_heatmap(df, columns, outdir=None,
                             name="correlation_heatmap.png"):
    # Heatmap of the feature correlation matrix with the values written on top.
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


def jl_projection_2d(X, seed=42):
    # Johnson-Lindenstrauss random projection to 2-D (lecture 15):
    # standardize, then multiply by a Gaussian matrix scaled by 1/sqrt(2).
    rng = np.random.default_rng(seed)
    centered = X - X.mean(axis=0)
    std = centered.std(axis=0)
    std[std == 0] = 1.0
    centered = centered / std
    random_matrix = rng.normal(0, 1, size=(X.shape[1], 2)) / np.sqrt(2)
    return centered @ random_matrix


def plot_2d_projection(X, y, outdir=None, name="projection_jl.png"):
    # Scatter of the 2-D random projection, colored by class.
    projected = jl_projection_2d(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color in zip([0, 1], ["#4C72B0", "#C44E52"]):
        mask = y == label
        ax.scatter(projected[mask, 0], projected[mask, 1], s=15, alpha=0.6,
                   color=color, label=f"target={label}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Johnson-Lindenstrauss random projection (2-D)")
    ax.legend()
    return _save(fig, outdir, name)


def plot_model_comparison(results, metric="accuracy", outdir=None, name=None):
    # Bar chart comparing one metric across models.
    name = name or f"model_comparison_{metric}.png"
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color="#55A868")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    plt.xticks(rotation=15)
    return _save(fig, outdir, name)


def plot_feature_importances(importances, feature_names, outdir=None,
                             name="feature_importances.png"):
    # Grouped bars: normalized importance of each feature for each model.
    model_names = list(importances.keys())
    n_features = len(feature_names)
    positions = np.arange(n_features)
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(model_names):
        values = np.asarray(importances[model_name], dtype=float)
        if values.sum() > 0:
            values = values / values.sum()  # normalize so models are comparable
        ax.bar(positions + i * width, values, width, label=model_name)
    ax.set_xticks(positions + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Normalized importance")
    ax.set_title("Feature importance across models")
    ax.legend()
    return _save(fig, outdir, name)


def plot_curve(x, y, xlabel, ylabel, title, outdir=None, name="curve.png",
               marker="o"):
    # Generic single-line curve used for the tuning/learning curves.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, outdir, name)


def plot_nn_loss(loss_history, outdir=None, name="nn_loss.png"):
    # Neural-network training loss over epochs.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Neural network training loss")
    ax.grid(alpha=0.3)
    return _save(fig, outdir, name)
