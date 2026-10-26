from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from metrics import WINKLER_CI


# Manuscript terminology for predictor columns; unmapped columns fall back to their raw name.
VARIABLE_LABELS = {
    "Alter_OP": "Alter",
    "Seite": "Seite",
    "Zeitpunkt_HV": "Zeitpunkt der Hörminderung",
    "Beginn_HV": "Dauer der Hörminderung",
    "Beginn_HHV": "Dauer der hochgradigen Hörminderung oder Taubheit",
    "Ursache_Transformiert": "Ursache",
    "SSD": "Einseitige Hörminderung (SSD)",
    "Versorgung_Gegenohr": "Versorgung Gegenohr",
    "HG_Nutzung": "Hörgerätenutzung",
    "Elektrodenform": "Elektrodenform",
    "PTA4_LL_prä": "4PTA LL",
    "EVmaxLL_prä": "EVmaxLL",
    "EV65HG_prä": "EV65HG",
}


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

    if "q2_feature_importance_with_wearing" in tables:
        lines.append("## Q2: Feature importance with wearing time")
        lines.append("")
        lines.append(df_to_md(tables["q2_feature_importance_with_wearing"]))
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


def save_figure(fig, path: Path) -> None:
    """Save a figure as PNG (300 dpi) and SVG; SVG text stays editable (real text, not outlines)."""
    fig.savefig(path.with_suffix(".png"), dpi=300)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(path.with_suffix(".svg"))


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
    save_figure(plt.gcf(), figures / "figure_1_q1_observed_vs_predicted_winkler")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist([inside["abs_error"], outside["abs_error"]], bins=range(0, 105, 5), label=["within CI", "outside CI"], edgecolor="black")
    plt.xlabel("Absolute prediction error (percentage points)")
    plt.ylabel("Number of patients")
    plt.title(f"Absolute errors by Winkler/Holube CI status: {best_model}")
    plt.legend()
    plt.tight_layout()
    save_figure(plt.gcf(), figures / "figure_2_q1_error_distribution")
    plt.close()


def _observed_vs_predicted_panel(ax, observed, predicted, r2: float, rng, xlabel: str, title: str) -> None:
    """Shared panel for Cochlea Figures 1 and 3: jittered scatter on fixed 0-100 axes with identity line and R²."""
    x_jittered = np.clip(predicted + rng.uniform(-1, 1, len(predicted)), 0, 100)
    y_jittered = np.clip(observed + rng.uniform(-1, 1, len(observed)), 0, 100)

    # clip_on=False keeps markers at 0 and 100 fully visible with the axes fixed to 0-100
    ax.scatter(x_jittered, y_jittered, s=12, alpha=0.5, clip_on=False)
    ax.plot([0, 100], [0, 100], "k--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Observed EV65CI (%)", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.text(0.04, 0.96, f"R² = {r2:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=14)
    ax.tick_params(labelsize=12)


def write_q1_top_models_scatter(out_dir: Path, q1_results: dict, random_seed: int = 42) -> None:
    """Cochlea Figure 1: observed vs predicted EV65CI for the two best Q1 models.

    Observed and predicted values both lie on 5-pp steps, so both get a small
    jitter for display only; R² is taken from the Q1 regression table (mean across CV folds).
    """
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    preds = q1_results["q1_regression_predictions"]
    ranked = (
        q1_results["q1_regression"]
        .query("model != 'Naive baseline'")
        .sort_values("MAE", ascending=True)
        .drop_duplicates("model")
        .head(2)
    )
    if preds.empty or ranked.empty:
        return

    rng = np.random.default_rng(random_seed)
    fig, axes = plt.subplots(1, len(ranked), figsize=(6 * len(ranked), 6), squeeze=False)
    for ax, (rank, row) in zip(axes.ravel(), enumerate(ranked.itertuples(index=False), start=1)):
        model_df = preds[preds["model"] == row.model]
        _observed_vs_predicted_panel(
            ax,
            observed=model_df["y_true"].to_numpy(dtype=float),
            predicted=model_df["y_pred"].to_numpy(dtype=float),
            r2=row.R2,
            rng=rng,
            xlabel="Predicted EV65CI (%)",
            title=f"{'Best' if rank == 1 else 'Second-best'} model: {row.model}",
        )

    fig.tight_layout()
    save_figure(fig, figures / "cochlea_figure_1_q1_top_models_observed_vs_predicted")
    plt.close(fig)


def write_q3_reference_models_scatter(out_dir: Path, hoppe_table: pd.DataFrame, hoppe_predictions: pd.DataFrame, random_seed: int = 42) -> None:
    """Cochlea Figure 3: observed vs predicted EV65CI for the published and the refitted Hoppe model.

    One figure per Q3 endpoint. R² comes from the Q3 table: a single value over the whole
    cohort for the published formula (no CV), the mean across CV folds for the refit.
    """
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if hoppe_table.empty or hoppe_predictions.empty:
        return
    panels = [
        ("Published Hoppe formula", "Original reference model", "Predicted EV65CI – Reference model (%)"),
        ("Hoppe variables refit", "Adapted reference model", "Predicted EV65CI – Adapted reference model (%)"),
    ]

    for endpoint in hoppe_table["endpoint"].dropna().unique():
        rng = np.random.default_rng(random_seed)
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.4), squeeze=False)
        for ax, (model, title, xlabel) in zip(axes.ravel(), panels):
            row = hoppe_table[(hoppe_table["endpoint"] == endpoint) & (hoppe_table["model"] == model)]
            model_df = hoppe_predictions[(hoppe_predictions["endpoint"] == endpoint) & (hoppe_predictions["model"] == model)]
            if row.empty or model_df.empty:
                ax.set_axis_off()
                continue
            _observed_vs_predicted_panel(
                ax,
                observed=model_df["y_true"].to_numpy(dtype=float),
                predicted=model_df["y_pred"].to_numpy(dtype=float),
                r2=row.iloc[0]["R2"],
                rng=rng,
                xlabel=xlabel,
                title=title,
            )
        fig.suptitle(f"Endpoint: {endpoint}", fontsize=15)
        fig.tight_layout()
        save_figure(fig, figures / f"cochlea_figure_3_q3_reference_models_{endpoint}")
        plt.close(fig)


def write_q2_best_model_importance(out_dir: Path, q1_regression: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Cochlea Figure 2: permutation importance (mean over repeats) of the best Q1 model.

    The best model is ranked as in Cochlea Figure 1 (lowest MAE, naive baseline excluded).
    """
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ranked = q1_regression.query("model != 'Naive baseline'").sort_values("MAE", ascending=True)
    if ranked.empty or importance.empty:
        return
    best_model = ranked.iloc[0]["model"]
    # barh draws the first row at the bottom, so ascending order puts the most important on top
    plot_df = importance[importance["model"] == best_model].sort_values("permutation_importance_mean")
    if plot_df.empty:
        return
    labels = plot_df["variable"].map(lambda v: VARIABLE_LABELS.get(v, v))

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(plot_df) + 1.8))
    ax.barh(labels, plot_df["permutation_importance_mean"])
    if (plot_df["permutation_importance_mean"] < 0).any():
        ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Permutation importance\n(increase in MAE, percentage points)", fontsize=14)
    ax.set_ylabel("")
    ax.set_title(f"Best model: {best_model}", fontsize=15)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    save_figure(fig, figures / "cochlea_figure_2_q2_best_model_importance")
    plt.close(fig)
