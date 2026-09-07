from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def unique_list(items):
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in unique_list(columns) if c in df.columns]


def missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in unique_list(columns) if c not in df.columns]


def clean_for_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({pd.NA: np.nan})
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith(("string", "category")):
            s = out[col].astype(object)
            s = s.where(pd.notna(s), np.nan)
            s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
            s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
            out[col] = s.astype(object)
    return out


def coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object or str(s.dtype).startswith(("string", "category")):
        s = (
            s.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        )
    return pd.to_numeric(s, errors="coerce")


def convert_numeric_like_columns(df: pd.DataFrame, columns: list[str], min_numeric_fraction: float = 0.80) -> pd.DataFrame:
    out = df.copy()
    for col in available_columns(out, columns):
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        converted = pd.to_numeric(out[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        if converted.notna().mean() >= min_numeric_fraction:
            out[col] = converted
    return out


def harmonize_values(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_for_sklearn(df)
    replacements = {
        "Schwindel": {"Leicht": "Vorhanden", "leicht": "Vorhanden", "Leicht (1)": "Vorhanden"},
        "OAE": {"Nicht Nachweisbar": "Nicht nachweisbar", "Nicht erhoben": np.nan},
        "Klickbera": {"Nicht erhoben": np.nan},
        "Bera_4kHz": {"Nicht erhoben": np.nan},
        "Zeitpunkt_HV": {"Postlingual": "postlingual", "Prälingual": "prälingual"},
        "Beginn_HV": {"Unbekannt/kA": np.nan, "Unbekannt": np.nan, "kA": np.nan, "ka": np.nan, "KA": np.nan},
        "Beginn_HHV": {
            "Unbekannt/kA": np.nan, "Unbekannt": np.nan, "kA": np.nan, "ka": np.nan, "KA": np.nan,
            "<1 y": "< 1 y", "> 20y": "> 20 y", "5 -10 y": "5-10 y", "5 - 10 y": "5-10 y"
        },
        "Ursache_Transformiert": {"Infektös": "Infektiös"},
        "Versorgung_Gegenohr": {"Nicht erhoben": np.nan},
        "Implantattyp": {
            "Flex28": "FLEX28", "FlexSoft": "FLEXSOFT", "Flex20": "FLEX20",
            "WB HiRes Ultra 3D Mid-Scala": "HiRes Ultra 3D Mid-Scala",
            "WB HiRes Ultra 3D SlimJ": "HiRes Ultra 3D SlimJ",
            "WB HiRes Ultra 3D Slim J": "HiRes Ultra 3D SlimJ",
        },
    }
    for col, mapping in replacements.items():
        if col in out.columns:
            out[col] = out[col].replace(mapping)
    return clean_for_sklearn(out)


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_numeric_categorical(df: pd.DataFrame, predictors: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols, categorical_cols = [], []
    for col in available_columns(df, predictors):
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def make_preprocessor(df: pd.DataFrame, predictors: list[str], impute: bool = True, scale_numeric: bool = True) -> ColumnTransformer:
    predictors = available_columns(df, predictors)
    numeric_cols, categorical_cols = split_numeric_categorical(df, predictors)
    transformers = []

    numeric_steps = []
    if impute:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if numeric_cols:
        numeric_transformer = Pipeline(numeric_steps) if numeric_steps else "passthrough"
        transformers.append(("num", numeric_transformer, numeric_cols))

    categorical_steps = []
    if impute:
        categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    categorical_steps.append(("onehot", make_onehot_encoder()))
    if categorical_cols:
        transformers.append(("cat", Pipeline(categorical_steps), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names_from_column_transformer(ct: ColumnTransformer) -> list[str]:
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        if transformer == "passthrough":
            names.extend(cols)
        elif hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                names.extend(last_step.get_feature_names_out(cols))
            else:
                names.extend(cols)
        else:
            names.extend(cols)
    return list(names)
