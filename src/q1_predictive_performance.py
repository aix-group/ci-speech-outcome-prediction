from __future__ import annotations

from pathlib import Path
import pandas as pd

from models import build_regression_models, build_classification_models
from evaluation import regression_cv, classification_cv


def run_q1(df: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    tables_dir = out_dir / "tables"
    preds_dir = out_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    predictors = cfg["primary_predictors"]
    target_reg = cfg["regression_target"]
    target_clf = cfg["classification_target"]
    seed = cfg.get("random_seed", 42)
    n_splits = cfg.get("n_splits", 5)

    reg_models = build_regression_models(df, predictors, random_seed=seed)
    reg_table, reg_pred, reg_data, reg_predictors = regression_cv(
        df, predictors, target_reg, reg_models,
        n_splits=n_splits, random_seed=seed, impute_predictors=True,
        analysis="Q1 regression", endpoint=target_reg,
    )

    clf_models = build_classification_models(df, predictors, random_seed=seed)
    clf_table, clf_pred, clf_data, clf_predictors, y_clf = classification_cv(
        df, predictors, target_clf, cfg, clf_models,
        n_splits=n_splits, random_seed=seed, impute_predictors=True,
    )

    reg_table.to_csv(tables_dir / "table_2_q1_regression.csv", index=False)
    clf_table.to_csv(tables_dir / "table_3_q1_classification.csv", index=False)
    reg_pred.to_csv(preds_dir / "q1_regression_predictions.csv", index=False)
    clf_pred.to_csv(preds_dir / "q1_classification_predictions.csv", index=False)

    return {
        "q1_regression": reg_table,
        "q1_classification": clf_table,
        "q1_regression_predictions": reg_pred,
        "q1_classification_predictions": clf_pred,
        "regression_analysis_data": reg_data,
        "regression_predictors": reg_predictors,
    }
