from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from data import prepare_analysis_frame, prepare_classification_frame
from metrics import regression_metrics, classification_metrics, round_to_five_percent, winkler_ci_bounds_for_prediction, winkler_inside_ci


def summarize_cv_rows(rows: list[dict], sort_by: str, ascending: bool = True) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    metric_cols = [c for c in table.columns if c not in ["model", "fold", "n", "analysis", "endpoint", "non_responder_n", "responder_n"]]
    means = table.groupby([c for c in ["endpoint", "analysis", "model"] if c in table.columns], as_index=False)[metric_cols].mean()
    sds = table.groupby([c for c in ["endpoint", "analysis", "model"] if c in table.columns], as_index=False)[metric_cols].std()
    sds = sds.rename(columns={c: f"{c}_sd" for c in metric_cols})
    group_cols = [c for c in ["endpoint", "analysis", "model"] if c in table.columns]
    meta_cols = [c for c in ["n", "non_responder_n", "responder_n"] if c in table.columns]
    if meta_cols:
        meta = table.groupby(group_cols, as_index=False)[meta_cols].first()
        out = means.merge(sds, on=group_cols, how="left").merge(meta, on=group_cols, how="left")
    else:
        out = means.merge(sds, on=group_cols, how="left")
    return out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)


def regression_cv(df, predictors, target, models, n_splits=5, random_seed=42, impute_predictors=True, analysis=None, endpoint=None):
    data, predictors = prepare_analysis_frame(df, predictors, target, impute_predictors=impute_predictors)
    X = data[predictors].copy()
    y = data[target].astype(float).values
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_seed).split(X, y))
    rows, predictions = [], []

    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            estimator = clone(model)
            estimator.fit(X.iloc[train_idx], y[train_idx])
            pred = round_to_five_percent(estimator.predict(X.iloc[test_idx]))
            metrics = regression_metrics(y[test_idx], pred)
            metrics.update({"model": model_name, "fold": fold, "n": len(y)})
            if analysis is not None:
                metrics["analysis"] = analysis
            if endpoint is not None:
                metrics["endpoint"] = endpoint
            rows.append(metrics)
            ci_low, ci_high, y_ref = winkler_ci_bounds_for_prediction(pred)
            inside = winkler_inside_ci(y[test_idx], pred)
            predictions.append(pd.DataFrame({
                "endpoint": endpoint,
                "analysis": analysis,
                "model": model_name,
                "fold": fold,
                "index": X.iloc[test_idx].index,
                "y_true": y[test_idx],
                "y_pred": pred,
                "y_pred_reference": y_ref,
                "winkler_low": ci_low,
                "winkler_high": ci_high,
                "within_winkler_ci": inside,
                "abs_error": np.abs(y[test_idx] - pred),
            }))
    return summarize_cv_rows(rows, sort_by="MAE", ascending=True), pd.concat(predictions, ignore_index=True), data, predictors


def classification_cv(df, predictors, target, cfg, models, n_splits=5, random_seed=42, impute_predictors=True):
    data, predictors, y_series = prepare_classification_frame(df, predictors, target, cfg, impute_predictors=impute_predictors)
    X = data[predictors].copy()
    y = y_series.astype(int).values
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed).split(X, y))
    rows, predictions = [], []

    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            estimator = clone(model)
            estimator.fit(X.iloc[train_idx], y[train_idx])
            pred = estimator.predict(X.iloc[test_idx])
            if hasattr(estimator, "predict_proba"):
                prob = estimator.predict_proba(X.iloc[test_idx])[:, 1]
            elif hasattr(estimator, "decision_function"):
                score = estimator.decision_function(X.iloc[test_idx])
                prob = 1 / (1 + np.exp(-score))
            else:
                prob = pred.astype(float)
            metrics = classification_metrics(y[test_idx], pred, prob)
            metrics.update({
                "model": model_name,
                "fold": fold,
                "n": len(y),
                "non_responder_n": int((y == 0).sum()),
                "responder_n": int((y == 1).sum()),
            })
            rows.append(metrics)
            predictions.append(pd.DataFrame({
                "model": model_name,
                "fold": fold,
                "index": X.iloc[test_idx].index,
                "y_true": y[test_idx],
                "y_pred": pred,
                "y_prob": prob,
            }))
    return summarize_cv_rows(rows, sort_by="Balanced Accuracy", ascending=False), pd.concat(predictions, ignore_index=True), data, predictors, y_series
