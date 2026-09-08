from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from metrics import WINKLER_CI


def df_to_md(df: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(digits)
    return out.to_markdown(index=False)


def write_environment(out_dir: Path) -> None:
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        (out_dir / "environment.txt").write_text(result.stdout, encoding="utf-8")
    except Exception as exc:
        (out_dir / "environment.txt").write_text(f"Could not collect environment: {exc}\n", encoding="utf-8")


def write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_config(config_path: str | Path, out_dir: Path) -> None:
    shutil.copy2(config_path, out_dir / "config_used.yaml")


def write_results_summary(out_dir: Path, manifest: dict, tables: dict[str, pd.DataFrame]) -> None:
    out_dir = Path(out_dir)
    lines = []
    lines.append("# Results summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## Run metadata")
    lines.append("")
    for key in ["run_id", "random_seed", "n_splits", "regression_target", "classification_target", "hoppe_6m_target"]:
        lines.append(f"- {key}: `{manifest.get(key)}`")
    lines.append("")

    if "cohort" in tables:
        lines.append("## Cohort")
        lines.append("")
        lines.append(df_to_md(tables["cohort"]))
        lines.append("")

    if "q1_regression" in tables:
        q1 = tables["q1_regression"]
        lines.append("## Q1: Predictive performance — regression")
        lines.append("")
        lines.append(df_to_md(q1))
        lines.append("")
        if not q1.empty and "MAE" in q1.columns:
            best = q1.sort_values("MAE", ascending=True).iloc[0]
            lines.append(f"Best regression model by MAE: **{best['model']}** (MAE {best['MAE']:.2f} pp).")
            lines.append("")

    if "q1_classification" in tables:
        q1c = tables["q1_classification"]
        lines.append("## Q1: Predictive performance — classification")
        lines.append("")
        lines.append(df_to_md(q1c))
        lines.append("")
        if not q1c.empty and "Balanced Accuracy" in q1c.columns:
            best = q1c.sort_values("Balanced Accuracy", ascending=False).iloc[0]
            lines.append(f"Best classification model by balanced accuracy: **{best['model']}** (balanced accuracy {best['Balanced Accuracy']:.3f}).")
            lines.append("")

    if "q1_ablation" in tables:
        lines.append("## Q1 ablations")
        lines.append("")
        lines.append(df_to_md(tables["q1_ablation"]))
        lines.append("")

    if "q2_feature_importance" in tables:
        lines.append("## Q2: Feature importance")
        lines.append("")
        lines.append(df_to_md(tables["q2_feature_importance"]))
        lines.append("")

    if "q3_hoppe" in tables:
        lines.append("## Q3: Hoppe comparison")
        lines.append("")
        lines.append(df_to_md(tables["q3_hoppe"]))
        lines.append("")

    lines.append("## Generated files")
    lines.append("")
    for label, path in manifest.get("outputs", {}).items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    (out_dir / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_cohort_table(df: pd.DataFrame, meta: dict, cfg: dict, out_dir: Path) -> pd.DataFrame:
    target = cfg["regression_target"]
    clf = cfg["classification_target"]
    hoppe6 = cfg.get("hoppe_6m_target")
    rows = [
        {"criterion": "Raw dataset", "n": meta["raw_shape"][0]},
        {"criterion": "Essen before exclusions", "n": meta["essen_shape_before_exclusions"][0]},
        {"criterion": "Essen after general exclusions", "n": meta["essen_shape_after_exclusions"][0]},
        {"criterion": f"Primary endpoint available ({target})", "n": int(df[target].notna().sum()) if target in df.columns else 0},
        {"criterion": f"Classification target available ({clf})", "n": int(df[clf].notna().sum()) if clf in df.columns else 0},
        {"criterion": f"Hoppe 6m endpoint available ({hoppe6})", "n": int(df[hoppe6].notna().sum()) if hoppe6 in df.columns else 0},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "tables" / "table_1_cohort.csv", index=False)
    return table


def write_supplement_tables(df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    predictors = cfg.get("primary_predictors", [])
    cols = [c for c in predictors + [cfg.get("regression_target"), cfg.get("classification_target"), cfg.get("hoppe_6m_target")] if c and c in df.columns]
    missing = df[cols].isna().agg(["sum", "mean"]).T.rename(columns={"sum": "missing_n", "mean": "missing_fraction"})
    missing["available_n"] = len(df) - missing["missing_n"]
    missing.to_csv(out_dir / "tables" / "table_s1_missingness.csv")
    WINKLER_CI.to_csv(out_dir / "tables" / "table_s2_winkler_holube_ci.csv", index=False)


def write_q1_figures(out_dir: Path, q1_results: dict) -> None:
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    preds = q1_results["q1_regression_predictions"].copy()
    q1 = q1_results["q1_regression"]
    if preds.empty or q1.empty:
        return
    best_model = q1.sort_values("MAE", ascending=True).iloc[0]["model"]
    plot_df = preds[preds["model"] == best_model].copy()
    if plot_df.empty:
        return

    inside = plot_df[plot_df["within_winkler_ci"].astype(bool)]
    outside = plot_df[~plot_df["within_winkler_ci"].astype(bool)]

    plt.figure(figsize=(6.8, 6.2))
    plt.scatter(inside["y_pred_reference"], inside["y_true"], alpha=0.75, label="within Winkler/Holube 95% CI")
    plt.scatter(outside["y_pred_reference"], outside["y_true"], alpha=0.75, label="outside Winkler/Holube 95% CI")
    plt.plot(WINKLER_CI["EV"], WINKLER_CI["EV"], linestyle="--", label="identity")
    plt.plot(WINKLER_CI["EV"], WINKLER_CI["LOW"], linestyle=":", label="95% CI lower/upper")
    plt.plot(WINKLER_CI["EV"], WINKLER_CI["HIGH"], linestyle=":")
    plt.xlabel("Predicted EV65CI reference value (%)")
    plt.ylabel("Observed EV65CI (%)")
    plt.title(f"Observed vs predicted EV65CI: {best_model}")
    plt.xlim(-2, 102)
    plt.ylim(-2, 102)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "figure_1_q1_observed_vs_predicted_winkler.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist([inside["abs_error"], outside["abs_error"]], bins=range(0, 105, 5), label=["within CI", "outside CI"], edgecolor="black")
    plt.xlabel("Absolute prediction error (percentage points)")
    plt.ylabel("Number of patients")
    plt.title(f"Absolute errors by Winkler/Holube CI status: {best_model}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "figure_2_q1_error_distribution.png", dpi=300)
    plt.close()
