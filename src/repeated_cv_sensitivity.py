from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import load_config
from data import prepare_analysis_frame, prepare_study_data
from evaluation import regression_cv
from models import build_single_regression_model
from preprocessing import unique_list
from reporting import copy_config, write_environment, write_manifest


DEFAULT_MODELS = ["ElasticNet", "MLP"]
DEFAULT_SEEDS = list(range(42, 62))


def parse_seed_spec(value: str) -> list[int]:
    """Parse comma-separated seeds and inclusive ranges, e.g. ``42:61,100``."""
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_text, end_text = item.split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(item))
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one split seed is required.")
    return seeds


def build_analysis_specs(df: pd.DataFrame, cfg: dict) -> list[dict]:
    predictors = cfg["primary_predictors"]
    target = cfg["regression_target"]
    wearing = cfg.get("wearing_time_predictor")
    if not wearing:
        raise KeyError("The config does not define wearing_time_predictor.")
    common_predictors = unique_list(predictors + [wearing])

    missing = [column for column in common_predictors if column not in df.columns]
    if missing:
        raise KeyError(f"Missing repeated-CV predictors: {missing}")

    primary_complete, _ = prepare_analysis_frame(
        df, predictors, target, impute_predictors=False
    )
    common_complete, _ = prepare_analysis_frame(
        df, common_predictors, target, impute_predictors=False
    )
    primary_imputed, _ = prepare_analysis_frame(
        df, predictors, target, impute_predictors=True
    )
    wearing_imputed, _ = prepare_analysis_frame(
        df, common_predictors, target, impute_predictors=True
    )

    return [
        {
            "analysis": "Main primary complete-case",
            "predictors": predictors,
            "data": primary_complete,
            "impute_predictors": False,
        },
        {
            "analysis": "Ablation primary common complete-case",
            "predictors": predictors,
            "data": common_complete,
            "impute_predictors": False,
        },
        {
            "analysis": "Ablation + wearing common complete-case",
            "predictors": common_predictors,
            "data": common_complete,
            "impute_predictors": False,
        },
        {
            "analysis": "Ablation primary imputed",
            "predictors": predictors,
            "data": primary_imputed,
            "impute_predictors": True,
        },
        {
            "analysis": "Ablation + wearing imputed",
            "predictors": common_predictors,
            "data": wearing_imputed,
            "impute_predictors": True,
        },
    ]


def summarize_repetitions(by_seed: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["analysis", "model", "n"]
    summary = (
        by_seed.groupby(group_columns, as_index=False)
        .agg(
            repetitions=("split_seed", "nunique"),
            mean_MAE_across_repeats=("MAE", "mean"),
            sd_of_mean_MAE=("MAE", "std"),
            min_mean_MAE=("MAE", "min"),
            max_mean_MAE=("MAE", "max"),
            mean_fold_MAE_SD=("MAE_sd", "mean"),
            median_fold_MAE_SD=("MAE_sd", "median"),
            min_fold_MAE_SD=("MAE_sd", "min"),
            max_fold_MAE_SD=("MAE_sd", "max"),
        )
        .sort_values(["analysis", "model"])
        .reset_index(drop=True)
    )
    return summary


def write_results_summary(
    out_dir: Path,
    summary: pd.DataFrame,
    split_seeds: list[int],
    model_names: list[str],
    model_seed: int,
    n_splits: int,
) -> None:
    display = summary.copy()
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)

    lines = [
        "# Repeated-CV sensitivity analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Method",
        "",
        f"- Repetitions: {len(split_seeds)}",
        f"- Outer CV: {n_splits} folds with shuffling",
        f"- Split seeds: {', '.join(str(seed) for seed in split_seeds)}",
        f"- Fixed model seed: {model_seed}",
        f"- Models: {', '.join(model_names)}",
        "",
        "Only the outer fold allocation changes between repetitions; the model seed is held fixed.",
        "",
        "## Results",
        "",
        display.to_markdown(index=False),
        "",
        "## Interpretation of the SD columns",
        "",
        "- `sd_of_mean_MAE`: standard deviation of the mean five-fold MAE across repetitions.",
        "- `mean_fold_MAE_SD`: mean, across repetitions, of the within-run standard deviation among the five fold MAEs.",
        "- `median_fold_MAE_SD`, `min_fold_MAE_SD`, and `max_fold_MAE_SD`: distribution of that within-run fold SD across repetitions.",
        "",
    ]
    (out_dir / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_repeated_cv(
    df: pd.DataFrame,
    cfg: dict,
    split_seeds: list[int],
    model_names: list[str],
    model_seed: int,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = cfg["regression_target"]
    analysis_specs = build_analysis_specs(df, cfg)
    seed_rows: list[pd.DataFrame] = []
    fold_rows: list[pd.DataFrame] = []

    for split_seed in split_seeds:
        print(f"Split seed {split_seed}")
        for spec in analysis_specs:
            for model_name in model_names:
                model = build_single_regression_model(
                    model_name,
                    spec["data"],
                    spec["predictors"],
                    random_seed=model_seed,
                )
                table, predictions, _, _ = regression_cv(
                    spec["data"],
                    spec["predictors"],
                    target,
                    {model_name: model},
                    n_splits=n_splits,
                    random_seed=split_seed,
                    impute_predictors=spec["impute_predictors"],
                    analysis=spec["analysis"],
                    endpoint=target,
                )
                table.insert(0, "split_seed", split_seed)
                table.insert(1, "model_seed", model_seed)
                seed_rows.append(table)

                fold_mae = (
                    predictions.groupby(
                        ["endpoint", "analysis", "model", "fold"], as_index=False
                    )
                    .agg(fold_n=("abs_error", "size"), MAE=("abs_error", "mean"))
                )
                fold_mae.insert(0, "split_seed", split_seed)
                fold_mae.insert(1, "model_seed", model_seed)
                fold_mae["n"] = len(spec["data"])
                fold_rows.append(fold_mae)

    by_seed = pd.concat(seed_rows, ignore_index=True)
    fold_results = pd.concat(fold_rows, ignore_index=True)
    summary = summarize_repetitions(by_seed)
    return by_seed, fold_results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the Q1 regression and ablation five-fold CV across split seeds."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument(
        "--seeds",
        type=parse_seed_spec,
        default=DEFAULT_SEEDS,
        help="Split seeds as comma-separated values/ranges (default: 42:61).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Regression models to evaluate (default: ElasticNet MLP).",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Fixed model seed (default: random_seed from config).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Number of CV folds (default: n_splits from config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split_seeds = args.seeds
    model_seed = args.model_seed if args.model_seed is not None else cfg.get("random_seed", 42)
    n_splits = args.n_splits if args.n_splits is not None else cfg.get("n_splits", 5)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    out_dir = Path(args.out or f"runs/{cfg.get('run_id', 'essen_primary')}_repeated_cv")
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df, meta = prepare_study_data(cfg)
    analysis_specs = build_analysis_specs(df, cfg)
    by_seed, fold_results, summary = run_repeated_cv(
        df,
        cfg,
        split_seeds=split_seeds,
        model_names=args.models,
        model_seed=model_seed,
        n_splits=n_splits,
    )

    by_seed.to_csv(tables_dir / "repeated_cv_by_seed.csv", index=False)
    fold_results.to_csv(tables_dir / "repeated_cv_fold_mae.csv", index=False)
    summary.to_csv(tables_dir / "repeated_cv_summary.csv", index=False)
    write_results_summary(
        out_dir, summary, split_seeds, args.models, model_seed, n_splits
    )

    manifest = {
        "analysis": "Repeated five-fold CV sensitivity analysis",
        "config_run_id": cfg.get("run_id"),
        "regression_target": cfg.get("regression_target"),
        "split_seeds": split_seeds,
        "model_seed": model_seed,
        "n_splits": n_splits,
        "models": args.models,
        "cohorts": [
            {
                "analysis": spec["analysis"],
                "n": len(spec["data"]),
                "predictors": spec["predictors"],
                "impute_predictors": spec["impute_predictors"],
            }
            for spec in analysis_specs
        ],
        "raw_shape": meta.get("raw_shape"),
        "essen_shape_before_exclusions": meta.get("essen_shape_before_exclusions"),
        "essen_shape_after_exclusions": meta.get("essen_shape_after_exclusions"),
        "outputs": {
            "results_summary": "results_summary.md",
            "by_seed": "tables/repeated_cv_by_seed.csv",
            "fold_mae": "tables/repeated_cv_fold_mae.csv",
            "summary": "tables/repeated_cv_summary.csv",
        },
        "notes": [
            "Split seeds vary only the shuffled outer KFold allocation.",
            "The model seed is held fixed across repetitions.",
            "MAE_sd is the sample standard deviation of the fold-specific MAEs within one CV repetition.",
        ],
    }
    write_manifest(out_dir, manifest)
    copy_config(args.config, out_dir)
    write_environment(out_dir)

    print("\nRepeated CV completed.")
    print(summary.to_string(index=False))
    print(f"\nResults written to: {out_dir}")


if __name__ == "__main__":
    main()
