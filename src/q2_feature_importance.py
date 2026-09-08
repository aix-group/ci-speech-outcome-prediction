from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from data import prepare_analysis_frame
from models import build_single_regression_model, build_regression_models
from preprocessing import get_feature_names_from_column_transformer, unique_list


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
        .agg(elasticnet_abs_coef_sum=("abs_coef", "sum"), elasticnet_signed_coef_sum=("coef", "sum"))
        .sort_values("elasticnet_abs_coef_sum", ascending=False)
    )
    variable_table["elasticnet_direction_hint"] = np.where(variable_table["elasticnet_signed_coef_sum"] > 0, "positive", np.where(variable_table["elasticnet_signed_coef_sum"] < 0, "negative", "zero"))
    return variable_table, feature_table


def _calculate_feature_importance(
    data: pd.DataFrame,
    predictors: list[str],
    target: str,
    model_names: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data[predictors].copy()
    y = data[target].astype(float).values

    elastic = build_regression_models(data, predictors, random_seed=seed)["ElasticNet"]
    elastic.fit(X, y)
    direction_variable, direction_feature = _elasticnet_direction_table(elastic, predictors)

    importance_tables = []
    for model_name in model_names:
        model = build_single_regression_model(model_name, data, predictors, random_seed=seed)
        model.fit(X, y)

        perm = permutation_importance(
            model,
            X,
            y,
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=seed,
            n_jobs=-1,
        )
        importance = pd.DataFrame({
            "variable": predictors,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_sd": perm.importances_std,
            "model": model_name,
            "n": len(data),
        }).sort_values("permutation_importance_mean", ascending=False)
        importance_tables.append(
            importance.merge(direction_variable, on="variable", how="left")
        )

    return pd.concat(importance_tables, ignore_index=True), direction_feature


def run_q2(df: pd.DataFrame, cfg: dict, q1_results: dict, out_dir: Path) -> dict:
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    primary_predictors = cfg["primary_predictors"]
    target = cfg["regression_target"]
    seed = cfg.get("random_seed", 42)
    wearing = cfg.get("wearing_time_predictor")
    if not wearing or wearing not in df.columns:
        raise KeyError(f"Missing wearing-time predictor: {wearing!r}")

    ranked = (
        q1_results["q1_regression"]
        .query("model != 'Naive baseline'")
        .sort_values("MAE")
        .drop_duplicates("model")
    )

    best_model_name = ranked.iloc[0]["model"]
    model_names = [best_model_name]

    if best_model_name == "ElasticNet":
        second_best = ranked.loc[
            ranked["model"] != "ElasticNet", "model"
        ].iloc[0]
        model_names.append(second_best)
    primary_data, primary_predictors = prepare_analysis_frame(
        df,
        primary_predictors,
        target,
        impute_predictors=False,
    )

    wearing_predictors = unique_list(primary_predictors + [wearing])
    wearing_data, wearing_predictors = prepare_analysis_frame(
        df,
        wearing_predictors,
        target,
        impute_predictors=False,
    )

    primary_importance, direction_feature = _calculate_feature_importance(
        primary_data,
        primary_predictors,
        target,
        model_names,
        seed,
    )
    wearing_importance, _ = _calculate_feature_importance(
        wearing_data,
        wearing_predictors,
        target,
        model_names,
        seed,
    )

    primary_importance.to_csv(tables_dir / "table_s3_feature_importance.csv", index=False)
    direction_feature.to_csv(tables_dir / "table_s4_elasticnet_feature_coefficients.csv", index=False)
    wearing_importance.to_csv(tables_dir / "table_s5_feature_importance_with_wearing.csv", index=False)

    # Figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(7 * len(model_names), 5),
        squeeze=False,
    )
    for ax, model_name in zip(axes.ravel(), model_names):
        top = (
            primary_importance.loc[primary_importance["model"] == model_name]
            .nlargest(15, "permutation_importance_mean")
            .sort_values("permutation_importance_mean")
        )
        ax.barh(top["variable"], top["permutation_importance_mean"])
        ax.set_xlabel("Permutation importance (increase in MAE)")
        ax.set_ylabel("")
        ax.set_title(f"Variable importance: {model_name}")

    fig.tight_layout()
    fig.savefig(figures_dir / "figure_3_q2_feature_importance.png", dpi=300)
    plt.close(fig)

    return {
        "q2_feature_importance": primary_importance,
        "q2_feature_importance_with_wearing": wearing_importance,
        "q2_elasticnet_feature_coefficients": direction_feature,
        "q2_best_models": model_names,
    }
