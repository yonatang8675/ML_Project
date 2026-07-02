# Heart Disease — Machine Learning Project

Predicting heart disease (binary classification) on the Kaggle
[`johnsmith88/heart-disease-dataset`](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
**All models are implemented from scratch with NumPy** (no ML libraries).

> The written report (submission) is in Hebrew under [`reports/`](reports/00_overview.md).
> Code and comments are in English.

## Models (from scratch)

| Model | Lecture | File |
|-------|---------|------|
| k-Nearest Neighbors | 10 | `src/models/knn.py` |
| Decision Tree (CART) | 11 | `src/models/decision_tree.py` |
| AdaBoost (decision stumps) | 8 | `src/models/adaboost.py` |
| Neural Network (MLP) | 6, 14 | `src/models/neural_network.py` |

## Headline results (deduplicated data, test set)

| Model | Accuracy | F1 | AUC |
|-------|:--------:|:--:|:---:|
| **k-NN** (k=11) | **0.885** | **0.896** | 0.926 |
| Decision Tree (depth=5) | 0.787 | 0.800 | 0.861 |
| AdaBoost (T=10) | 0.787 | 0.787 | **0.929** |
| Neural Net | 0.754 | 0.769 | 0.867 |

Key finding: the dataset has **723 duplicate rows (1025 → 302 unique)**. Without
removing them, memorizing models reach a fake 100% accuracy (train/test leakage).
See [`reports/05_challenges.md`](reports/05_challenges.md).

## Project layout

```
src/                 from-scratch implementations
  data_loader.py     download (kagglehub) + clean + dedup
  preprocessing.py   split, StandardScaler, k-fold
  metrics.py         accuracy/precision/recall/F1/ROC-AUC
  evaluation.py      CV, model comparison, permutation importance, leakage experiment
  plots.py           figures (incl. from-scratch PCA + JL projection)
  models/            knn, decision_tree, adaboost, neural_network
notebooks/
  heart_disease.ipynb   runnable end-to-end
tools/
  build_notebook.py     regenerates the notebook
reports/             Hebrew submission (.md)
figures/             generated plots
```

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/heart_disease.ipynb
# or open notebooks/heart_disease.ipynb and "Restart & Run All"
```

The dataset downloads automatically via `kagglehub` (no authentication required).
All randomness is seeded (`seed=42`) for reproducibility.
