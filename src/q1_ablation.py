from __future__ import annotations

from pathlib import Path
import pandas as pd

from models import build_single_regression_model
from evaluation import regression_cv
from data import prepare_analysis_frame
from preprocessing import available_columns, unique_list


def run_q1_ablation(df: pd.DataFrame, cfg: dict, q1_results: dict, out_dir: Path) -> dict:
    tables_dir = out_dir / "tables"
    preds_dir = out_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    q1_reg = q1_results["q1_regression"]
    best_model = q1_reg.sort_values("MAE", ascending=True).iloc[0]["model"]
    model_names = [best_model]
    if best_model != "ElasticNet":
        model_names.append("ElasticNet")

    predictors = cfg["primary_predictors"]
    target = cfg["regression_target"]
    seed = cfg.get("random_seed", 42)
    n_splits = cfg.get("n_splits", 5)

    wearing = cfg.get("wearing_time_predictor")
    common_predictors = unique_list(predictors + [wearing])

    complete_data, _ = prepare_analysis_frame(
        df,
        common_predictors,
        target,
        impute_predictors=False,
    )

    imputed_data, _ = prepare_analysis_frame(
        df,
        predictors,
        target,
        impute_predictors=True,
    )


    missing = [col for col in common_predictors if col not in df.columns]
    if missing:
        raise KeyError(f"Missing ablation predictors: {missing}")

    specs = []
    for model_name in model_names:
        specs.append(("Primary predictors, common complete-case cohort", model_name, predictors, complete_data, False))
        wearing = cfg.get("wearing_time_predictor")
        if wearing and wearing in df.columns:
            specs.append(("Postoperative: + Tragezeit, common complete-case cohort", model_name, common_predictors, complete_data, False))
        specs.append(("Primary predictors, imputed cohort", model_name, predictors, imputed_data, True))

    rows, preds = [], []
    for analysis, model_name, pred_set, analysis_data, impute in specs:
        pred_set = available_columns(analysis_data, pred_set)
        model = build_single_regression_model(model_name, analysis_data, pred_set, random_seed=seed)
        table, pred, _, _ = regression_cv(
            analysis_data, pred_set, target, {model_name: model},
            n_splits=n_splits, random_seed=seed, impute_predictors=impute,
            analysis=analysis, endpoint=target,
        )
        rows.append(table)
        preds.append(pred)

    ablation_table = pd.concat(rows, ignore_index=True)
    ablation_pred = pd.concat(preds, ignore_index=True)
    ablation_table.to_csv(tables_dir / "table_4_q1_ablation.csv", index=False)
    ablation_pred.to_csv(preds_dir / "q1_ablation_predictions.csv", index=False)
    return {"q1_ablation": ablation_table, "q1_ablation_predictions": ablation_pred}
