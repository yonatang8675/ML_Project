"""Build the project notebook (notebooks/heart_disease.ipynb) with nbformat.

Generating the notebook from a script keeps the JSON valid and lets us rebuild
it deterministically. Run:  python3 tools/build_notebook.py
"""

import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "heart_disease.ipynb")

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    # Add a markdown cell.
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src):
    # Add a code cell.
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# Title
md(r"""
# Heart Disease Prediction — Machine Learning Project

This notebook implements **from scratch (NumPy only)** every stage of the project on the
*Heart Disease* dataset (Kaggle: `johnsmith88/heart-disease-dataset`).
The goal is **binary classification**: does a patient have heart disease (`target=1`) or not (`target=0`)?

## Research questions
1. **Predicting heart disease** — can we predict whether the disease is present from the patient's data?
2. **Comparing models** — which model achieves the best performance?
3. **Feature importance** — which features most affect the prediction, and do the models agree?

## The models (implemented from scratch)
1. **k-Nearest Neighbors** (lecture 10)
2. **Decision Tree** (lecture 11)
3. **AdaBoost** (lecture 8)
4. **Neural Network** — MLP, based on the Perceptron (lecture 6) and logistic regression (lecture 14)
""")

# Setup
md("## 1. Setup")
code(r"""
import sys
import os
sys.path.append(os.path.abspath(".."))  # make `src` importable

import numpy as np
import pandas as pd
from IPython.display import Image, display

from src import data_loader as dl
from src import metrics, evaluation as ev, plots
from src.preprocessing import train_test_split, StandardScaler, kfold_indices
from src.models.knn import KNNClassifier
from src.models.decision_tree import DecisionTreeClassifier
from src.models.adaboost import AdaBoostClassifier
from src.models.neural_network import NeuralNetwork

SEED = 42
np.random.seed(SEED)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print("Environment ready.")
""")

# Data
md(r"""
## 2. Loading the data and describing the dataset

The dataset has 13 medical features (age, sex, chest pain type, blood pressure, cholesterol,
maximum heart rate, and more) and a binary target column. We load the raw version and the
de-duplicated version and compare them.
""")
code(r"""
raw_df = dl.load_data(dedup=False)
df = dl.load_data(dedup=True)          # de-duplicated — used for honest evaluation
X, y, feature_names = dl.get_Xy(df)

summary = dl.dataset_summary()
print("Raw rows:        ", summary["n_rows_raw"])
print("Unique rows:     ", summary["n_rows_unique"])
print("Duplicate rows:  ", summary["n_duplicates"])
print("Missing values:  ", summary["n_missing"])
print("Target (unique): ", summary["target_balance_unique"])
df.head()
""")
code(r"""
# Descriptions of each feature
pd.DataFrame(
    [(c, dl.FEATURE_DESCRIPTIONS[c]) for c in feature_names + ["target"]],
    columns=["feature", "description"],
)
""")
code(r"""
df.describe().T
""")

# EDA
md("## 3. Exploratory Data Analysis (EDA)")
code("display(Image(plots.plot_class_balance(y)))")
code("display(Image(plots.plot_feature_histograms(df, feature_names)))")
code("display(Image(plots.plot_correlation_heatmap(df, feature_names + ['target'])))")
md(r"""
### 3.1 Dimensionality reduction for visualization (lecture 15)

We project the data to 2-D using a **Johnson–Lindenstrauss random projection** (lecture 15).
The projection shows partial separation between the sick and healthy patients.
""")
code(r"""
display(Image(plots.plot_2d_projection(X, y)))
""")

# Leakage
md(r"""
## 4. Main challenge: duplicates and data leakage

The Kaggle version contains **723 duplicate rows** (1025 → 302 unique). A random split puts
identical copies in both train and test, so a "memorizing" model gets an inflated accuracy.
We demonstrate this: models that can memorize (1-NN, a full tree) reach 100% on the data with
duplicates, and only a realistic accuracy once they are removed.
""")
code(r"""
X_raw, y_raw, _ = dl.get_Xy(raw_df)      # with duplicates
X_unique, y_unique, _ = dl.get_Xy(df)    # without duplicates

experiments = {
    "1-NN":                 (lambda: KNNClassifier(k=1), True),
    "5-NN":                 (lambda: KNNClassifier(k=5), True),
    "Decision Tree (full)": (lambda: DecisionTreeClassifier(max_depth=None), False),
    "Decision Tree (d=4)":  (lambda: DecisionTreeClassifier(max_depth=4), False),
}

rows = []
for name, (factory, scale) in experiments.items():
    experiment = ev.duplicate_leakage_experiment(
        factory, X_raw, y_raw, X_unique, y_unique, seed=SEED, scale=scale
    )
    rows.append({
        "model": name,
        "acc_with_duplicates": experiment["accuracy_with_duplicates"],
        "acc_deduplicated": experiment["accuracy_deduplicated"],
        "gap": experiment["accuracy_with_duplicates"] - experiment["accuracy_deduplicated"],
    })

leakage_df = pd.DataFrame(rows)
leakage_df
""")
md(r"""
**Conclusion:** the gap (`gap`) demonstrates the leakage. Therefore **all evaluation from here on
uses the de-duplicated data only** — this is the honest way to measure true generalization.
""")

# Preprocessing
md(r"""
## 5. Preprocessing and splitting

We split into train/test while preserving the class ratio (stratified) and standardize (z-score)
using the train set only. Distance/gradient-based models (k-NN, neural network) use the
standardized data; trees and AdaBoost use the original values (threshold splits are not sensitive
to scale).
""")
code(r"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, seed=SEED, stratify=True
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train balance:", np.bincount(y_train), " Test balance:", np.bincount(y_test))
""")

# kNN
md("## 6. Model 1 — k-Nearest Neighbors (lecture 10)")
code(r"""
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
cv_acc = []
for k in k_values:
    scores = ev.cross_val_score(lambda k=k: KNNClassifier(k=k), X_train, y_train,
                                n_splits=5, seed=SEED, scale=True)
    cv_acc.append(scores.mean())

best_k = k_values[int(np.argmax(cv_acc))]
print("CV accuracy per k:", {k: round(a, 3) for k, a in zip(k_values, cv_acc)})
print("Best k =", best_k)
display(Image(plots.plot_curve(k_values, cv_acc, "k (neighbors)",
            "5-fold CV accuracy", "k-NN: choosing k", name="knn_k_curve.png")))
""")

# Tree
md("## 7. Model 2 — Decision Tree (lecture 11)")
code(r"""
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
train_acc = []
val_acc = []
for depth in depths:
    train_scores = []
    val_scores = []
    for train_idx, val_idx in kfold_indices(y_train, n_splits=5, seed=SEED):
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        tree.fit(X_train[train_idx], y_train[train_idx])
        train_scores.append(metrics.accuracy(y_train[train_idx], tree.predict(X_train[train_idx])))
        val_scores.append(metrics.accuracy(y_train[val_idx], tree.predict(X_train[val_idx])))
    train_acc.append(np.mean(train_scores))
    val_acc.append(np.mean(val_scores))

labels = [str(depth) if depth is not None else "None" for depth in depths]
best_depth = depths[int(np.argmax(val_acc))]
print("Best max_depth =", best_depth, " (CV acc=%.3f)" % max(val_acc))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(labels, train_acc, "o-", label="train")
ax.plot(labels, val_acc, "s-", label="validation")
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.set_title("Decision tree: bias-variance vs depth")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("../figures/tree_depth_curve.png", dpi=130)
plt.close(fig)
display(Image("../figures/tree_depth_curve.png"))
""")
md(r"""
The overfitting curve illustrates the generalization principle from the **PAC and VC-dimension**
lectures: a deeper tree fits the train set almost perfectly (accuracy→1), but the validation
performance starts to drop — too much complexity hurts generalization.
""")

# AdaBoost
md("## 8. Model 3 — AdaBoost (lecture 8)")
code(r"""
ada = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
display(Image(plots.plot_curve(range(1, len(ada.train_errors_) + 1),
            ada.train_errors_, "boosting round", "training error",
            "AdaBoost: training error vs rounds", name="adaboost_rounds.png",
            marker="")))

# Choose the number of rounds by 5-fold CV.
round_grid = [1, 5, 10, 20, 30, 50, 75, 100]
cv_round = []
for n_rounds in round_grid:
    scores = ev.cross_val_score(lambda n=n_rounds: AdaBoostClassifier(n_estimators=n),
                                X_train, y_train, n_splits=5, seed=SEED, scale=False)
    cv_round.append(scores.mean())
best_n_rounds = round_grid[int(np.argmax(cv_round))]
print("CV accuracy per n_estimators:", {n: round(acc, 3) for n, acc in zip(round_grid, cv_round)})
print("Best n_estimators =", best_n_rounds)
""")
md(r"""
The AdaBoost training-error bound from lecture 8 is $\prod_t 2\sqrt{\epsilon_t(1-\epsilon_t)}$,
so the training error decreases (almost) monotonically with the number of rounds — as the curve shows.
""")

# NN
md("## 9. Model 4 — Neural Network (MLP, lectures 6 and 14)")
code(r"""
# Verify back-propagation with a numerical gradient check.
grad_diff = NeuralNetwork(hidden_layers=(8,), seed=0).gradient_check(X_train_s[:20], y_train[:20])
print(f"Numerical gradient check (relative diff) = {grad_diff:.2e}  -> {'PASS' if grad_diff < 1e-5 else 'FAIL'}")

nn = NeuralNetwork(hidden_layers=(16,), activation="relu", lr=0.1,
                   epochs=300, batch_size=32, l2=1e-4, seed=SEED)
nn.fit(X_train_s, y_train)
print("Final training loss = %.4f" % nn.loss_history_[-1])
print("Test accuracy = %.3f" % metrics.accuracy(y_test, nn.predict(X_test_s)))
display(Image(plots.plot_nn_loss(nn.loss_history_)))
""")

# Comparison
md("## 10. Model comparison (research question 2)")
code(r"""
# Final tuned models.
final_models = {
    "k-NN":          (lambda: KNNClassifier(k=best_k), True),
    "Decision Tree": (lambda: DecisionTreeClassifier(criterion="entropy", max_depth=best_depth), False),
    "AdaBoost":      (lambda: AdaBoostClassifier(n_estimators=best_n_rounds), False),
    "Neural Net":    (lambda: NeuralNetwork(hidden_layers=(16,), lr=0.1, epochs=300,
                                            batch_size=32, l2=1e-4, seed=SEED), True),
}

results = {}
fitted = {}
for name, (factory, scale) in final_models.items():
    X_fit, X_eval = (X_train_s, X_test_s) if scale else (X_train, X_test)
    model = factory()
    model.fit(X_fit, y_train)
    fitted[name] = (model, scale)
    results[name] = ev.evaluate_model(model, X_fit, y_train, X_eval, y_test)

results_df = pd.DataFrame(results).T[["accuracy"]]
results_df = results_df.sort_values("accuracy", ascending=False)
results_df
""")
code(r"""
display(Image(plots.plot_model_comparison(results, metric="accuracy")))
best_name = results_df.index[0]
print("Best model:", best_name, "(accuracy = %.3f)" % results_df.loc[best_name, "accuracy"])
""")

# Importance
md("## 11. Feature importance (research question 3)")
code(r"""
importances = {}

# Model-specific importances (impurity decrease in the tree, stump weights in AdaBoost).
dt_full = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth).fit(X_train, y_train)
importances["Decision Tree"] = dt_full.feature_importances_
importances["AdaBoost"] = fitted["AdaBoost"][0].feature_importances_

display(Image(plots.plot_feature_importances(importances, feature_names)))

# Average the normalized importances across models to get a consensus.
normalized = {}
for model_name, importance in importances.items():
    importance = np.asarray(importance)
    normalized[model_name] = importance / (np.sum(np.abs(importance)) or 1)

imp_df = pd.DataFrame(normalized, index=feature_names)
imp_df["mean"] = imp_df.mean(axis=1)
imp_df.sort_values("mean", ascending=False)
""")

# Answers
md(r"""
## 12. Answers to the research questions

**1. Predicting heart disease.** Yes — after removing the duplicates, all models achieve accuracy
significantly above 50% (the random baseline), so heart disease can be predicted from the patient's
data at a useful level of accuracy.

**2. Comparing models.** See the results table and chart in section 10. The leading model is
identified automatically (`best_name`) by test accuracy.

**3. Feature importance.** See section 11. The features that recur as important for both models
(e.g. `cp`, `ca`, `slope`, `sex`) point to partial agreement between the methods, even though each
model weighs them slightly differently.

## 13. Challenges and lessons
- **Duplicate leakage** was the main challenge; we handled it by removing duplicates and evaluating honestly.
- **Standardization** was critical for k-NN and the neural network, and irrelevant for trees/AdaBoost.
- **Overfitting** in trees and in the number of AdaBoost rounds was controlled with cross-validation.
- **Gradient checking** in the neural network verified the correctness of the back-propagation.

The full conclusions are in the report files in the `reports/` folder.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", OUT, "with", len(cells), "cells")
