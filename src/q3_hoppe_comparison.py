from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from data import apply_hoppe_exclusions, prepare_analysis_frame
from preprocessing import coerce_numeric
from metrics import regression_metrics, round_to_five_percent, winkler_ci_bounds_for_prediction, winkler_inside_ci
from models import published_hoppe_formula, build_hoppe_refit_model, build_single_regression_model
from evaluation import regression_cv


def _fixed_hoppe_row(df, cfg, endpoint: str) -> tuple[dict, pd.DataFrame]:
    y_true = coerce_numeric(df[endpoint]).astype(float).values
    y_pred = round_to_five_percent(published_hoppe_formula(df, cfg))
    row = regression_metrics(y_true, y_pred)
    row.update({"endpoint": endpoint, "analysis": "Hoppe comparison", "model": "Published Hoppe formula", "n": len(y_true)})
    ci_low, ci_high, y_ref = winkler_ci_bounds_for_prediction(y_pred)
    inside = winkler_inside_ci(y_true, y_pred)
    pred = pd.DataFrame({
        "endpoint": endpoint,
        "analysis": "Hoppe comparison",
        "model": "Published Hoppe formula",
        "fold": np.nan,
        "index": df.index,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_reference": y_ref,
        "winkler_low": ci_low,
        "winkler_high": ci_high,
        "within_winkler_ci": inside,
        "abs_error": np.abs(y_true - y_pred),
    })
    return row, pred


def run_q3(df: pd.DataFrame, cfg: dict, q1_results: dict, out_dir: Path) -> dict:
    tables_dir = out_dir / "tables"
    preds_dir = out_dir / "predictions"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("random_seed", 42)
    n_splits = cfg.get("n_splits", 5)
    hoppe_predictors = cfg["hoppe_predictors"]
    endpoints = [cfg.get("hoppe_6m_target"), cfg.get("regression_target")]
    endpoints = [e for e in endpoints if e and e in df.columns]

    best_q1_model = q1_results["q1_regression"].sort_values("MAE", ascending=True).iloc[0]["model"]
    if best_q1_model == "Naive baseline":
        best_q1_model = "ElasticNet"

    all_tables, all_predictions = [], []

    hoppe_base = apply_hoppe_exclusions(df, cfg)
    for endpoint in endpoints:
        needed = hoppe_predictors + [endpoint]
        dfx = hoppe_base.dropna(subset=[c for c in needed if c in hoppe_base.columns]).copy()
        for col in needed:
            if col in dfx.columns:
                dfx[col] = coerce_numeric(dfx[col])
        dfx = dfx.dropna(subset=needed).copy()
        if len(dfx) < n_splits:
            continue

        fixed_row, fixed_pred = _fixed_hoppe_row(dfx, cfg, endpoint)
        all_predictions.append(fixed_pred)

        refit_table, refit_pred, _, _ = regression_cv(
            dfx, hoppe_predictors, endpoint,
            {"Hoppe variables refit": build_hoppe_refit_model(), "Naive baseline": build_single_regression_model("Naive baseline", dfx, hoppe_predictors, seed)},
            n_splits=n_splits, random_seed=seed, impute_predictors=True,
            analysis="Hoppe comparison", endpoint=endpoint,
        )
        all_predictions.append(refit_pred)

        primary_predictors = [p for p in cfg["primary_predictors"] if p in dfx.columns]
        if primary_predictors:
            q1_model = build_single_regression_model(best_q1_model, dfx, primary_predictors, seed)
            main_table, main_pred, _, _ = regression_cv(
                dfx, primary_predictors, endpoint,
                {f"Best Q1 model ({best_q1_model})": q1_model},
                n_splits=n_splits, random_seed=seed, impute_predictors=True,
                analysis="Hoppe comparison", endpoint=endpoint,
            )
            all_predictions.append(main_pred)
            table = pd.concat([pd.DataFrame([fixed_row]), refit_table, main_table], ignore_index=True, sort=False)
        else:
            table = pd.concat([pd.DataFrame([fixed_row]), refit_table], ignore_index=True, sort=False)
        all_tables.append(table)

    hoppe_table = pd.concat(all_tables, ignore_index=True, sort=False) if all_tables else pd.DataFrame()
    hoppe_pred = pd.concat(all_predictions, ignore_index=True, sort=False) if all_predictions else pd.DataFrame()

    hoppe_table.to_csv(tables_dir / "table_5_q3_hoppe_comparison.csv", index=False)
    hoppe_pred.to_csv(preds_dir / "q3_hoppe_predictions.csv", index=False)

    if not hoppe_table.empty:
        import matplotlib.pyplot as plt
        plot_df = hoppe_table.dropna(subset=["MAE"]).copy()
        plot_df["label"] = plot_df["endpoint"].astype(str) + " | " + plot_df["model"].astype(str)
        plot_df = plot_df.sort_values("MAE", ascending=True)
        plt.figure(figsize=(9, max(4, 0.35 * len(plot_df))))
        plt.barh(plot_df["label"], plot_df["MAE"])
        plt.xlabel("MAE (percentage points)")
        plt.ylabel("")
        plt.title("Hoppe comparison")
        plt.tight_layout()
        plt.savefig(figures_dir / "figure_4_q3_hoppe_comparison.png", dpi=300)
        plt.close()

    return {"q3_hoppe": hoppe_table, "q3_hoppe_predictions": hoppe_pred}
