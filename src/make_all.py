from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from data import prepare_study_data
from q1_predictive_performance import run_q1
from q1_ablation import run_q1_ablation
from q2_feature_importance import run_q2
from q3_hoppe_comparison import run_q3
from reporting import (
    write_cohort_table,
    write_supplement_tables,
    write_q1_figures,
    write_results_summary,
    write_environment,
    write_manifest,
    copy_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Essen CI outcome prediction analyses.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--out", default=None, help="Output run directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out or f"runs/{cfg.get('run_id', 'essen_primary')}")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    df, meta = prepare_study_data(cfg)
    cohort_table = write_cohort_table(df, meta, cfg, out_dir)
    write_supplement_tables(df, cfg, out_dir)

    q1 = run_q1(df, cfg, out_dir)
    write_q1_figures(out_dir, q1)

    q1_ablation = run_q1_ablation(df, cfg, q1, out_dir)
    q2 = run_q2(df, cfg, q1, out_dir)
    q3 = run_q3(df, cfg, q1, out_dir)

    tables = {
        "cohort": cohort_table,
        "q1_regression": q1["q1_regression"],
        "q1_classification": q1["q1_classification"],
        "q1_ablation": q1_ablation["q1_ablation"],
        "q2_feature_importance": q2["q2_feature_importance"],
        "q3_hoppe": q3["q3_hoppe"],
    }

    manifest = {
        "run_id": cfg.get("run_id"),
        "random_seed": cfg.get("random_seed", 42),
        "n_splits": cfg.get("n_splits", 5),
        "raw_shape": meta.get("raw_shape"),
        "essen_shape_before_exclusions": meta.get("essen_shape_before_exclusions"),
        "essen_shape_after_exclusions": meta.get("essen_shape_after_exclusions"),
        "regression_target": cfg.get("regression_target"),
        "classification_target": cfg.get("classification_target"),
        "hoppe_6m_target": cfg.get("hoppe_6m_target"),
        "models": ["Naive baseline", "ElasticNet", "Random Forest", "XGBoost", "MLP"],
        "outputs": {
            "cohort_table": "tables/table_1_cohort.csv",
            "q1_regression_table": "tables/table_2_q1_regression.csv",
            "q1_classification_table": "tables/table_3_q1_classification.csv",
            "q1_ablation_table": "tables/table_4_q1_ablation.csv",
            "q2_feature_importance": "tables/table_s3_feature_importance.csv", 
            "q2_elasticnet_feature_coefficients": "tables/table_s4_elasticnet_feature_coefficients.csv",
            "q3_hoppe_table": "tables/table_5_q3_hoppe_comparison.csv",
            "q1_observed_vs_predicted_figure": "figures/figure_1_q1_observed_vs_predicted_winkler.png",
            "q1_error_distribution_figure": "figures/figure_2_q1_error_distribution.png",
            "q2_feature_importance_figure": "figures/figure_3_q2_feature_importance.png",
            "q3_hoppe_figure": "figures/figure_4_q3_hoppe_comparison.png",
        },
    }

    write_manifest(out_dir, manifest)
    copy_config(args.config, out_dir)
    write_environment(out_dir)
    write_results_summary(out_dir, manifest, tables)

    print(f"Analysis completed. Results written to: {out_dir}")
    print(f"Summary: {out_dir / 'results_summary.md'}")


if __name__ == "__main__":
    main()
