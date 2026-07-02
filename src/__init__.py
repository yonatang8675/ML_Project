"""
Heart Disease ML project — from-scratch (NumPy) implementations.

Sub-packages / modules:
- data_loader:   download + load + clean the Heart Disease dataset.
- preprocessing: train/test split, standardization, k-fold helpers (from scratch).
- metrics:       classification metrics + ROC/AUC (from scratch).
- evaluation:    cross-validation, model comparison, permutation importance.
- plots:         matplotlib figures.
- models:        Decision Tree, k-NN, AdaBoost, Neural Network (all from scratch).
"""

__all__ = [
    "data_loader",
    "preprocessing",
    "metrics",
    "evaluation",
    "plots",
    "models",
]
