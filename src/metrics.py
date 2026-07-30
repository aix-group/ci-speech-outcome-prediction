from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def clip_percent(y):
    return np.clip(np.asarray(y, dtype=float), 0, 100)


def round_to_five_percent(y):
    return np.clip(np.round(np.asarray(y, dtype=float) / 5.0) * 5.0, 0, 100)


def rmse_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


WINKLER_CI = pd.DataFrame({
    "EV":   [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    "LOW":  [0, 0, 0, 0, 5, 10, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 80, 90, 100],
    "HIGH": [0, 10, 20, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 90, 95, 100, 100, 100, 100],
})


def winkler_ci_bounds_for_prediction(y_pred):
    ci = WINKLER_CI.set_index("EV")
    y_ref = round_to_five_percent(y_pred).astype(int)
    lows = np.array([ci.loc[v, "LOW"] if v in ci.index else np.nan for v in y_ref], dtype=float)
    highs = np.array([ci.loc[v, "HIGH"] if v in ci.index else np.nan for v in y_ref], dtype=float)
    return lows, highs, y_ref


def winkler_inside_ci(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    lows, highs, _ = winkler_ci_bounds_for_prediction(y_pred)
    return (y_true >= lows) & (y_true <= highs)


def regression_metrics(y_true, y_pred, round_predictions=True):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_percent(y_pred)
    if round_predictions:
        y_pred = round_to_five_percent(y_pred)
    abs_error = np.abs(y_true - y_pred)
    inside = winkler_inside_ci(y_true, y_pred)
    outside = ~inside
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "RMSE": rmse_score(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Winkler hit rate": float(np.nanmean(inside)),
        "MAE inside Winkler CI": float(np.nanmean(abs_error[inside])) if np.any(inside) else np.nan,
        "MAE outside Winkler CI": float(np.nanmean(abs_error[outside])) if np.any(outside) else np.nan,
    }


def classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["AUROC"] = roc_auc_score(y_true, y_prob)
        out["AUPRC"] = average_precision_score(y_true, y_prob)
    else:
        out["AUROC"] = np.nan
        out["AUPRC"] = np.nan
    return out
