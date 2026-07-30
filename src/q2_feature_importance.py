from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from data import prepare_analysis_frame
from models import build_single_regression_model, build_regression_models
from preprocessing import get_feature_names_from_column_transformer


def _collapse_feature_to_variable(feature: str, predictors: list[str]) -> str:
    for predictor in sorted(predictors, key=len, reverse=True):
        if feature == predictor or feature.startswith(predictor + "_"):
            return predictor
    return feature


def _elasticnet_direction_table(model, predictors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = model.named_steps["preprocess"]
    est = model.named_steps["model"]
    feature_names = get_feature_names_from_column_transformer(pre)
    coefs = np.ravel(est.coef_)
    feature_table = pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)})
    feature_table["variable"] = feature_table["feature"].map(lambda x: _collapse_feature_to_variable(x, predictors))
    variable_table = (
        feature_table.groupby("variable", as_index=False)
        .agg(abs_coef_sum=("abs_coef", "sum"), signed_coef_sum=("coef", "sum"))
        .sort_values("abs_coef_sum", ascending=False)
    )
    variable_table["direction_hint"] = np.where(variable_table["signed_coef_sum"] > 0, "positive", np.where(variable_table["signed_coef_sum"] < 0, "negative", "zero"))
    return variable_table, feature_table


def run_q2(df: pd.DataFrame, cfg: dict, q1_results: dict, out_dir: Path) -> dict:
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictors = cfg["primary_predictors"]
    target = cfg["regression_target"]
    seed = cfg.get("random_seed", 42)

    best_model_name = q1_results["q1_regression"].sort_values("MAE", ascending=True).iloc[0]["model"]
    if best_model_name == "Naive baseline":
        best_model_name = "ElasticNet"

    data, predictors = prepare_analysis_frame(df, predictors, target, impute_predictors=True)
    X = data[predictors].copy()
    y = data[target].astype(float).values

    best_model = build_single_regression_model(best_model_name, data, predictors, random_seed=seed)
    best_model.fit(X, y)

    perm = permutation_importance(
        best_model, X, y,
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame({
        "variable": predictors,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_sd": perm.importances_std,
        "model": best_model_name,
    }).sort_values("permutation_importance_mean", ascending=False)

    elastic = build_regression_models(data, predictors, random_seed=seed)["ElasticNet"]
    elastic.fit(X, y)
    direction_variable, direction_feature = _elasticnet_direction_table(elastic, predictors)

    merged = importance.merge(direction_variable, on="variable", how="left")
    merged.to_csv(tables_dir / "table_s3_feature_importance.csv", index=False)
    direction_feature.to_csv(tables_dir / "table_s4_elasticnet_feature_coefficients.csv", index=False)

    # Figure
    import matplotlib.pyplot as plt
    top = merged.head(15).sort_values("permutation_importance_mean")
    plt.figure(figsize=(7, 5))
    plt.barh(top["variable"], top["permutation_importance_mean"])
    plt.xlabel("Permutation importance (increase in MAE)")
    plt.ylabel("")
    plt.title(f"Variable importance: {best_model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_3_q2_feature_importance.png", dpi=300)
    plt.close()

    return {"q2_feature_importance": merged, "q2_elasticnet_feature_coefficients": direction_feature, "q2_best_model": best_model_name}
