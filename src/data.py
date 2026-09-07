from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from preprocessing import clean_for_sklearn, coerce_numeric, harmonize_values, available_columns, unique_list, convert_numeric_like_columns


def first_existing_column(df: pd.DataFrame, candidates: list[str], required: bool = False, label: str = "column") -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"No {label} found. Tried: {candidates}")
    return None


def load_raw_data(cfg: dict) -> pd.DataFrame:
    data_path = Path(cfg["data_path"])
    sheet_name = cfg.get("sheet_name", 0)
    return pd.read_excel(data_path, sheet_name=sheet_name)


def filter_essen(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    center_col = cfg.get("center_column", "Klinik")
    center_value = str(cfg.get("center_value", "Essen")).strip().casefold()
    if center_col not in df_raw.columns:
        raise KeyError(f"Expected center column {center_col!r} not found.")
    mask = df_raw[center_col].astype(str).str.strip().str.casefold().eq(center_value)
    return df_raw.loc[mask].copy()


def make_primary_endpoint(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    target = cfg["regression_target"]
    if target in out.columns:
        out[target] = coerce_numeric(out[target])
        return out
    col24 = first_existing_column(out, cfg.get("ev65ci_24_candidates", []), required=True, label="24-month EV65CI")
    col12 = first_existing_column(out, cfg.get("ev65ci_12_candidates", []), required=True, label="12-month EV65CI")
    out[col24] = coerce_numeric(out[col24])
    out[col12] = coerce_numeric(out[col12])
    out[target] = out[col24].combine_first(out[col12])
    return out


def apply_general_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    if "Alter_OP" in out.columns:
        age = pd.to_numeric(out["Alter_OP"], errors="coerce")
        mask &= age.isna() | (age >= 18)
    language_cols = [c for c in out.columns if "sprach" in c.lower()]
    for col in language_cols:
        s = out[col].astype(str).str.strip().str.casefold()
        mask &= ~s.isin(["ja", "yes", "true", "1", "sprachbarriere", "vorhanden"])
    return out.loc[mask].copy()


def apply_hoppe_exclusions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    exclusions = cfg.get("hoppe_exclusions", {})

    if exclusions.get("exclude_prelingual", True) and "Zeitpunkt_HV" in out.columns:
        z = out["Zeitpunkt_HV"].astype(str).str.strip().str.casefold()
        mask &= ~z.isin(["prälingual", "praelingual", "prelingual"])

    if exclusions.get("exclude_evmax_zero", True) and "EVmaxLL_prä" in out.columns:
        evmax = coerce_numeric(out["EVmaxLL_prä"])
        mask &= evmax.ne(0) | evmax.isna()

    return out.loc[mask].copy()


def prepare_study_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw = load_raw_data(cfg)
    essen = filter_essen(raw, cfg)
    df = harmonize_values(essen)
    df = make_primary_endpoint(df, cfg)
    df = apply_general_exclusions(df)
    df = clean_for_sklearn(df)

    numeric_like = unique_list(
        cfg.get("primary_predictors", [])
        + cfg.get("hoppe_predictors", [])
        + [cfg.get("regression_target"), cfg.get("hoppe_6m_target"), cfg.get("wearing_time_predictor")]
    )
    for col in available_columns(df, numeric_like):
        if col in ["Geschlecht", "Seite", "Tinnitus", "Schwindel", "OAE", "Klickbera", "Bera_4kHz",
                   "Zeitpunkt_HV", "Beginn_HHV", "Ursache_Transformiert", "SSD", "Versorgung_Gegenohr",
                   "HG_Nutzung", "Elektrodenform", "Beginn_HV"]:
            continue
        df[col] = coerce_numeric(df[col])

    meta = {
        "raw_shape": list(raw.shape),
        "essen_shape_before_exclusions": list(essen.shape),
        "essen_shape_after_exclusions": list(df.shape),
    }
    return df, meta


def prepare_analysis_frame(df: pd.DataFrame, predictors: list[str], target: str, impute_predictors: bool = True) -> tuple[pd.DataFrame, list[str]]:
    predictors = available_columns(df, predictors)
    cols = unique_list(predictors + [target])
    data = clean_for_sklearn(df[cols].copy())
    data[target] = coerce_numeric(data[target])
    data = convert_numeric_like_columns(data, predictors)
    data = data[data[target].notna()].copy()
    if not impute_predictors:
        data = data.dropna(subset=predictors)
    return clean_for_sklearn(data), predictors


def prepare_classification_frame(df: pd.DataFrame, predictors: list[str], target: str, cfg: dict, impute_predictors: bool = True) -> tuple[pd.DataFrame, list[str], pd.Series]:
    predictors = available_columns(df, predictors)
    cols = unique_list(predictors + [target])
    data = clean_for_sklearn(df[cols].copy())
    data = convert_numeric_like_columns(data, predictors)
    data = data[data[target].notna()].copy()

    y_raw = data[target]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce").astype(float)
        y = y.map(lambda v: 1 if v == 1 else (0 if v == 0 else np.nan))
    else:
        positives = {str(v).strip().casefold() for v in cfg.get("classification_positive_values", [])}
        s = y_raw.astype(str).str.strip().str.casefold()
        y = s.map(lambda v: 1 if v in positives else 0)
    data = data.loc[y.notna()].copy()
    y = y.loc[data.index].astype(int)

    if not impute_predictors:
        keep = data[predictors].notna().all(axis=1)
        data = data.loc[keep].copy()
        y = y.loc[data.index].copy()
    return clean_for_sklearn(data), predictors, y
