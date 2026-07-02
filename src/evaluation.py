"""
Evaluation utilities: cross-validation, model comparison, permutation feature
importance, and the duplicate-leakage experiment. All from scratch.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import metrics
from .preprocessing import StandardScaler, kfold_indices, train_test_split


def cross_val_score(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    scale: bool = False,
    scorer: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy,
) -> np.ndarray:
    """
    Run stratified k-fold CV and return the per-fold scores.

    ``model_factory`` must return a fresh, unfitted model each call.
    If ``scale`` is True, a StandardScaler is fit on each training fold.
    """
    scores = []
    for train_idx, val_idx in kfold_indices(y, n_splits=n_splits, seed=seed):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if scale:
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_val = scaler.transform(X_tr), scaler.transform(X_val)

        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        scores.append(scorer(y_val, y_pred))
    return np.array(scores)


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, evaluate on test; return a metrics dict including AUC."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = metrics.classification_report(y_test, y_pred)

    # AUC if the model exposes probabilities / scores.
    try:
        proba = model.predict_proba(X_test)[:, 1]
        report["auc"] = metrics.roc_auc_score(y_test, proba)
    except AttributeError:
        report["auc"] = float("nan")
    return report


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
    scorer: Callable[[np.ndarray, np.ndarray], float] = metrics.accuracy,
) -> np.ndarray:
    """
    Model-agnostic feature importance.

    For each feature, shuffle its column and measure the drop in score. The
    mean drop over ``n_repeats`` shuffles is the importance. The model must
    already be fitted.
    """
    rng = np.random.default_rng(seed)
    baseline = scorer(y, model.predict(X))
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            score = scorer(y, model.predict(X_perm))
            drops.append(baseline - score)
        importances[j] = np.mean(drops)
    return importances


def compare_models(
    model_factories: dict[str, Callable[[], object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_flags: dict[str, bool] | None = None,
) -> dict[str, dict]:
    """
    Train and evaluate several models, returning a dict name -> metrics dict.

    ``scale_flags`` lets distance/gradient-based models (k-NN, NN) receive
    standardized inputs while trees/AdaBoost use raw features.
    """
    scale_flags = scale_flags or {}
    results: dict[str, dict] = {}

    for name, factory in model_factories.items():
        if scale_flags.get(name, False):
            scaler = StandardScaler().fit(X_train)
            X_tr, X_te = scaler.transform(X_train), scaler.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test
        results[name] = evaluate_model(factory(), X_tr, y_train, X_te, y_test)
    return results


def duplicate_leakage_experiment(
    model_factory: Callable[[], object],
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    X_unique: np.ndarray,
    y_unique: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    scale: bool = False,
) -> dict:
    """
    Demonstrate the impact of duplicate rows.

    Trains/evaluates the same model on (a) the raw data with duplicates and
    (b) the de-duplicated data, returning both test accuracies. The gap
    illustrates train/test leakage.
    """
    def _run(X, y):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, seed=seed, stratify=True
        )
        if scale:
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
        model = model_factory()
        model.fit(X_tr, y_tr)
        return metrics.accuracy(y_te, model.predict(X_te))

    return {
        "accuracy_with_duplicates": _run(X_raw, y_raw),
        "accuracy_deduplicated": _run(X_unique, y_unique),
    }
