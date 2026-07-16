"""Cross-validation, model evaluation and the duplicate-leakage experiment."""

import numpy as np

from . import metrics
from .preprocessing import StandardScaler, kfold_indices, train_test_split


def cross_val_score(model_factory, X, y, n_splits=5, seed=42, scale=False,
                    scorer=metrics.accuracy):
    # Stratified k-fold CV. model_factory() must return a fresh, unfitted model.
    scores = []
    for train_idx, val_idx in kfold_indices(y, n_splits=n_splits, seed=seed):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit the scaler on the training fold only.
        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

        model = model_factory()
        model.fit(X_train, y_train)
        scores.append(scorer(y_val, model.predict(X_val)))
    return np.array(scores)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Fit on train, score on test.
    model.fit(X_train, y_train)
    return {"accuracy": metrics.accuracy(y_test, model.predict(X_test))}


def duplicate_leakage_experiment(model_factory, X_raw, y_raw, X_unique, y_unique,
                                 test_size=0.2, seed=42, scale=False):
    # Train the same model with and without duplicate rows; the accuracy gap
    # shows the train/test leakage the duplicates cause.
    def run(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, seed=seed, stratify=True
        )
        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        model = model_factory()
        model.fit(X_train, y_train)
        return metrics.accuracy(y_test, model.predict(X_test))

    return {
        "accuracy_with_duplicates": run(X_raw, y_raw),
        "accuracy_deduplicated": run(X_unique, y_unique),
    }
