"""Classification accuracy, from scratch. Labels are assumed to be in {0, 1}."""

import numpy as np


def accuracy(y_true, y_pred):
    # Fraction of predictions that match the true labels.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))
