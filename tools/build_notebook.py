"""
Build the project notebook (notebooks/heart_disease.ipynb) with nbformat.

Keeping the notebook generation in a script guarantees valid JSON and lets us
regenerate the notebook deterministically. Run:  python3 tools/build_notebook.py
"""

from __future__ import annotations

import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "heart_disease.ipynb")

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# חיזוי מחלות לב — פרויקט למידת מכונה

מחברת זו מיישמת **מאפס (NumPy בלבד)** את כל שלבי הפרויקט על מאגר
*Heart Disease* (Kaggle: `johnsmith88/heart-disease-dataset`).
המטרה היא **סיווג בינארי**: האם למטופל יש מחלת לב (`target=1`) או לא (`target=0`).

## שאלות המחקר
1. **חיזוי מחלת לב** — האם ניתן לחזות את קיום המחלה על סמך נתוני המטופל?
2. **השוואת מודלים** — איזה מודל משיג את הביצועים הטובים ביותר?
3. **חשיבות מאפיינים** — אילו תכונות משפיעות ביותר על התחזית, והאם יש הסכמה בין המודלים?

## המודלים (מומשו מאפס)
1. **k-Nearest Neighbors** (הרצאה 10)
2. **עץ החלטה / Decision Tree** (הרצאה 11)
3. **AdaBoost** (הרצאה 8)
4. **רשת נוירונים / Neural Network** — MLP, מבוסס על תפיסת ה-Perceptron (הרצאה 6) והרגרסיה הלוגיסטית (הרצאה 14)
""")

# ---------------------------------------------------------------- setup
md("## 1. הכנת הסביבה (Setup)")
code(r"""
import sys, os
sys.path.append(os.path.abspath(".."))  # make `src` importable

import numpy as np
import pandas as pd
from IPython.display import Image, display

from src import data_loader as dl
from src import metrics, evaluation as ev, plots
from src.preprocessing import train_test_split, StandardScaler, kfold_indices
from src.models import (
    KNNClassifier, DecisionTreeClassifier, AdaBoostClassifier, NeuralNetwork,
)

SEED = 42
np.random.seed(SEED)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print("Environment ready.")
""")

# ---------------------------------------------------------------- data
md(r"""
## 2. טעינת הנתונים ותיאור המאגר

המאגר כולל 13 מאפיינים רפואיים (גיל, מין, סוג כאב בחזה, לחץ דם, כולסטרול, דופק מרבי ועוד)
ועמודת מטרה בינארית. נטען את הגרסה הגולמית ואת הגרסה ללא כפילויות, ונשווה ביניהן.
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

# ---------------------------------------------------------------- EDA
md("## 3. ניתוח חוקר (EDA)")
code("display(Image(plots.plot_class_balance(y)))")
code("display(Image(plots.plot_feature_histograms(df, feature_names)))")
code("display(Image(plots.plot_correlation_heatmap(df, feature_names + ['target'])))")
md(r"""
### 3.1 הפחתת ממד לצורך ויזואליזציה (הרצאה 15)

נשליך את הנתונים לדו-ממד בשתי דרכים: **PCA** (מהאלגברה הלינארית, באמצעות SVD)
ו-**היטל אקראי של Johnson–Lindenstrauss**. שתי השיטות מראות הפרדה חלקית בין החולים לבריאים.
""")
code(r"""
display(Image(plots.plot_2d_projection(X, y, method="pca")))
display(Image(plots.plot_2d_projection(X, y, method="jl")))
""")

# ---------------------------------------------------------------- leakage
md(r"""
## 4. אתגר מרכזי: כפילויות ודליפת מידע (Data Leakage)

גרסת ה-Kaggle מכילה **723 שורות כפולות** (1025 → 302 ייחודיות). פיצול אקראי מציב
עותקים זהים גם ב-train וגם ב-test, כך שמודל "משנן" מקבל דיוק מנופח. נדגים זאת:
מודלים בעלי יכולת שינון (1-NN, עץ מלא) מגיעים ל-100% על הנתונים עם הכפילויות,
ולדיוק ריאליסטי בלבד לאחר הסרתן.
""")
code(r"""
Xr, yr, _ = dl.get_Xy(raw_df)   # raw (with duplicates)
Xu, yu, _ = dl.get_Xy(df)       # unique

experiments = {
    "1-NN":                 (lambda: KNNClassifier(k=1), True),
    "5-NN":                 (lambda: KNNClassifier(k=5), True),
    "Decision Tree (full)": (lambda: DecisionTreeClassifier(max_depth=None), False),
    "Decision Tree (d=4)":  (lambda: DecisionTreeClassifier(max_depth=4), False),
}

rows = []
for name, (factory, scale) in experiments.items():
    e = ev.duplicate_leakage_experiment(factory, Xr, yr, Xu, yu, seed=SEED, scale=scale)
    rows.append({
        "model": name,
        "acc_with_duplicates": e["accuracy_with_duplicates"],
        "acc_deduplicated": e["accuracy_deduplicated"],
        "gap": e["accuracy_with_duplicates"] - e["accuracy_deduplicated"],
    })

leakage_df = pd.DataFrame(rows)
leakage_df
""")
md(r"""
**מסקנה:** הפער (`gap`) מדגים את הדליפה. לכן **כל ההערכה בהמשך מתבצעת על הנתונים
ללא כפילויות בלבד** — זו הדרך הישרה למדוד הכללה אמיתית.
""")

# ---------------------------------------------------------------- preprocessing
md(r"""
## 5. עיבוד מקדים ופיצול

נפצל ל-train/test בשמירה על יחס המחלקות (stratified), ונתקנן (z-score) על בסיס ה-train
בלבד. מודלים מבוססי מרחק/גרדיאנט (k-NN, רשת נוירונים) משתמשים בנתונים המתוקננים;
עצים ו-AdaBoost משתמשים בערכים המקוריים (חיתוכי סף אינם רגישים לסקלה).
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

# ---------------------------------------------------------------- kNN
md("## 6. מודל 1 — k-Nearest Neighbors (הרצאה 10)")
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

# ---------------------------------------------------------------- tree
md("## 7. מודל 2 — עץ החלטה (הרצאה 11)")
code(r"""
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
train_acc, val_acc = [], []
for d in depths:
    tr_scores, va_scores = [], []
    for tr_idx, va_idx in kfold_indices(y_train, n_splits=5, seed=SEED):
        m = DecisionTreeClassifier(criterion="entropy", max_depth=d)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        tr_scores.append(metrics.accuracy(y_train[tr_idx], m.predict(X_train[tr_idx])))
        va_scores.append(metrics.accuracy(y_train[va_idx], m.predict(X_train[va_idx])))
    train_acc.append(np.mean(tr_scores)); val_acc.append(np.mean(va_scores))

labels = [str(d) if d is not None else "None" for d in depths]
best_depth = depths[int(np.argmax(val_acc))]
print("Best max_depth =", best_depth, " (CV acc=%.3f)" % max(val_acc))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(labels, train_acc, "o-", label="train")
ax.plot(labels, val_acc, "s-", label="validation")
ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy")
ax.set_title("Decision tree: bias-variance vs depth"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("../figures/tree_depth_curve.png", dpi=130); plt.close(fig)
display(Image("../figures/tree_depth_curve.png"))
""")
md(r"""
עקומת ה-overfitting ממחישה את עקרון ההכללה מהרצאות **PAC ו-VC-dimension**: עץ עמוק יותר
מתאים בצורה כמעט מושלמת ל-train (דיוק→1) אך ביצועי ה-validation מתחילים לרדת — מורכבות
גבוהה מדי פוגעת בהכללה.
""")

# ---------------------------------------------------------------- adaboost
md("## 8. מודל 3 — AdaBoost (הרצאה 8)")
code(r"""
ada = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
display(Image(plots.plot_curve(range(1, len(ada.train_errors_) + 1),
            ada.train_errors_, "boosting round", "training error",
            "AdaBoost: training error vs rounds", name="adaboost_rounds.png",
            marker="")))

# Choose the number of rounds by 5-fold CV.
round_grid = [1, 5, 10, 20, 30, 50, 75, 100]
cv_round = []
for T in round_grid:
    s = ev.cross_val_score(lambda T=T: AdaBoostClassifier(n_estimators=T),
                           X_train, y_train, n_splits=5, seed=SEED, scale=False)
    cv_round.append(s.mean())
best_T = round_grid[int(np.argmax(cv_round))]
print("CV accuracy per T:", {T: round(a, 3) for T, a in zip(round_grid, cv_round)})
print("Best n_estimators =", best_T)
""")
md(r"""
חסם שגיאת האימון של AdaBoost מהרצאה 8 הוא $\prod_t 2\sqrt{\epsilon_t(1-\epsilon_t)}$,
ולכן שגיאת האימון יורדת (כמעט) מונוטונית עם מספר הסבבים — כפי שנראה בעקומה.
""")

# ---------------------------------------------------------------- NN
md("## 9. מודל 4 — רשת נוירונים (MLP, הרצאות 6 ו-14)")
code(r"""
# Verify back-propagation with a numerical gradient check.
gc = NeuralNetwork(hidden_layers=(8,), seed=0).gradient_check(X_train_s[:20], y_train[:20])
print(f"Numerical gradient check (relative diff) = {gc:.2e}  -> {'PASS' if gc < 1e-5 else 'FAIL'}")

nn = NeuralNetwork(hidden_layers=(16,), activation="relu", lr=0.1,
                   epochs=300, batch_size=32, l2=1e-4, seed=SEED)
nn.fit(X_train_s, y_train)
print("Final training loss = %.4f" % nn.loss_history_[-1])
print("Test accuracy = %.3f" % metrics.accuracy(y_test, nn.predict(X_test_s)))
display(Image(plots.plot_nn_loss(nn.loss_history_)))
""")

# ---------------------------------------------------------------- comparison
md("## 10. השוואת מודלים (שאלת מחקר 2)")
code(r"""
# Final tuned models.
final_models = {
    "k-NN":          (lambda: KNNClassifier(k=best_k), True),
    "Decision Tree": (lambda: DecisionTreeClassifier(criterion="entropy", max_depth=best_depth), False),
    "AdaBoost":      (lambda: AdaBoostClassifier(n_estimators=best_T), False),
    "Neural Net":    (lambda: NeuralNetwork(hidden_layers=(16,), lr=0.1, epochs=300,
                                            batch_size=32, l2=1e-4, seed=SEED), True),
}

results, fitted, roc_data = {}, {}, {}
for name, (factory, scale) in final_models.items():
    Xtr, Xte = (X_train_s, X_test_s) if scale else (X_train, X_test)
    model = factory(); model.fit(Xtr, y_train)
    fitted[name] = (model, scale)
    results[name] = ev.evaluate_model(model, Xtr, y_train, Xte, y_test)
    proba = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    roc_data[name] = (fpr, tpr, metrics.auc(fpr, tpr))

results_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "auc"]]
results_df = results_df.sort_values("accuracy", ascending=False)
results_df
""")
code(r"""
display(Image(plots.plot_model_comparison(results, metric="accuracy")))
display(Image(plots.plot_model_comparison(results, metric="f1", name="model_comparison_f1.png")))
display(Image(plots.plot_roc_curves(roc_data)))
""")
code(r"""
# Confusion matrix of the best model.
best_name = results_df.index[0]
best_model, best_scale = fitted[best_name]
Xte = X_test_s if best_scale else X_test
cm = metrics.confusion_matrix(y_test, best_model.predict(Xte))
display(Image(plots.plot_confusion_matrix(cm, name="confusion_matrix_best.png",
            title=f"Confusion matrix — {best_name}")))
print("Best model:", best_name)
""")

# ---------------------------------------------------------------- importance
md("## 11. חשיבות מאפיינים (שאלת מחקר 3)")
code(r"""
importances = {}

# Model-specific importances.
dt_full = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth).fit(X_train, y_train)
importances["Decision Tree"] = dt_full.feature_importances_
importances["AdaBoost"] = fitted["AdaBoost"][0].feature_importances_

# Permutation importance (model-agnostic) for k-NN and the neural net.
knn_model = fitted["k-NN"][0]
importances["k-NN (perm)"] = ev.permutation_importance(knn_model, X_test_s, y_test, seed=SEED)
nn_model = fitted["Neural Net"][0]
importances["Neural Net (perm)"] = ev.permutation_importance(nn_model, X_test_s, y_test, seed=SEED)

display(Image(plots.plot_feature_importances(importances, feature_names)))

# Average rank across models -> consensus.
imp_df = pd.DataFrame({m: np.asarray(v) / (np.sum(np.abs(v)) or 1)
                       for m, v in importances.items()}, index=feature_names)
imp_df["mean"] = imp_df.mean(axis=1)
imp_df.sort_values("mean", ascending=False)
""")

# ---------------------------------------------------------------- answers
md(r"""
## 12. תשובות לשאלות המחקר

**1. חיזוי מחלת לב.** כן — לאחר הסרת הכפילויות, כל המודלים משיגים דיוק גבוה משמעותית
מ-50% (קו הבסיס האקראי), כך שניתן לחזות מחלת לב מנתוני המטופל ברמת דיוק שימושית.

**2. השוואת מודלים.** ראו טבלת התוצאות והגרפים בסעיף 10. המודל המוביל מזוהה אוטומטית
(`best_name`) לפי דיוק המבחן, עם השוואת precision/recall/F1/AUC.

**3. חשיבות מאפיינים.** ראו סעיף 11. המאפיינים החוזרים כחשובים על פני מספר מודלים
(למשל `cp`, `thalach`, `oldpeak`, `ca`, `thal`, `exang`) מצביעים על הסכמה חלקית בין
השיטות, אף שכל מודל שוקל אותם מעט אחרת.

## 13. אתגרים ולקחים
- **דליפת כפילויות** הייתה האתגר המרכזי; ההתמודדות הייתה הסרת כפילויות והערכה ישרה.
- **תקנון** היה קריטי ל-k-NN ולרשת הנוירונים, וחסר משמעות לעצים/AdaBoost.
- **Overfitting** בעצים ובמספר סבבי ה-AdaBoost נשלט באמצעות תיקוף צולב (cross-validation).
- **אימות נגזרות** ברשת הנוירונים (gradient check) ווידא את נכונות ה-back-propagation.

המסקנות המלאות מופיעות בקבצי הדוח בתיקיית `reports/`.
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
