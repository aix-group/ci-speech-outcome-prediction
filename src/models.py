from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from preprocessing import make_preprocessor

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


class HoppeLogisticRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_iter=2000):
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        age = X[:, 0]
        evmax = X[:, 1]
        ev65ha = X[:, 2]

        def predict_from_beta(beta):
            eta = beta[0] + beta[1] * evmax + beta[2] * age + beta[3] * ev65ha
            return 100.0 / (1.0 + np.exp(-eta))

        def objective(beta):
            pred = predict_from_beta(beta)
            return np.mean((y - pred) ** 2)

        beta0 = np.array([0.84, 0.012, -0.0094, 0.0059], dtype=float)
        result = minimize(objective, beta0, method="Nelder-Mead", options={"maxiter": self.max_iter})
        if not result.success:
            result = minimize(objective, beta0, method="BFGS", options={"maxiter": self.max_iter})
        self.coef_ = np.asarray(result.x, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        age = X[:, 0]
        evmax = X[:, 1]
        ev65ha = X[:, 2]
        beta = self.coef_
        eta = beta[0] + beta[1] * evmax + beta[2] * age + beta[3] * ev65ha
        return 100.0 / (1.0 + np.exp(-eta))


def build_regression_models(df, predictors, random_seed=42):
    pre_scaled = make_preprocessor(df, predictors, impute=True, scale_numeric=True)
    pre_unscaled = make_preprocessor(df, predictors, impute=True, scale_numeric=False)

    models = {
        "Naive baseline": DummyRegressor(strategy="median"),
        "ElasticNet": Pipeline([
            ("preprocess", pre_scaled),
            ("model", ElasticNetCV(
                l1_ratio=[0.05, 0.1, 0.5, 0.9, 1.0],
                alphas=np.logspace(-3, 2, 50),
                cv=5,
                random_state=random_seed,
                max_iter=20000,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", pre_unscaled),
            ("model", RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=5,
                random_state=random_seed,
                n_jobs=-1,
            )),
        ]),
        "MLP": Pipeline([
            ("preprocess", pre_scaled),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                early_stopping=True,
                max_iter=1000,
                random_state=random_seed,
            )),
        ]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([
            ("preprocess", pre_unscaled),
            ("model", XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_seed,
                objective="reg:squarederror",
                n_jobs=-1,
            )),
        ])
    return models


def build_classification_models(df, predictors, random_seed=42):
    pre_scaled = make_preprocessor(df, predictors, impute=True, scale_numeric=True)
    pre_unscaled = make_preprocessor(df, predictors, impute=True, scale_numeric=False)

    models = {
        "Naive baseline": DummyClassifier(strategy="most_frequent"),
        "ElasticNet": Pipeline([
            ("preprocess", pre_scaled),
            ("model", LogisticRegression(
                solver="saga",
                l1_ratio=0.5,  # l1_ratio=0.5 already indicates elastic-net regularisation
                C=1.0,
                class_weight="balanced",
                max_iter=20000,
                random_state=random_seed,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", pre_unscaled),
            ("model", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            )),
        ]),
        "MLP": Pipeline([
            ("preprocess", pre_scaled),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                early_stopping=True,
                max_iter=1000,
                random_state=random_seed,
            )),
        ]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([
            ("preprocess", pre_unscaled),
            ("model", XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_seed,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ])
    return models


def build_single_regression_model(model_name, df, predictors, random_seed=42):
    models = build_regression_models(df, predictors, random_seed=random_seed)
    if model_name not in models:
        raise KeyError(f"Model {model_name!r} not available. Available: {list(models)}")
    return models[model_name]


def published_hoppe_formula(df, cfg):
    from preprocessing import coerce_numeric
    beta = cfg["hoppe_coefficients"]
    age = coerce_numeric(df["Alter_OP"])
    evmax = coerce_numeric(df["EVmaxLL_prä"])
    ev65ha = coerce_numeric(df["EV65HG_prä"])
    eta = beta["intercept"] + beta["wrsmax"] * evmax + beta["age"] * age + beta["wrs65ha"] * ev65ha
    return 100.0 / (1.0 + np.exp(-eta))


def build_hoppe_refit_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HoppeLogisticRegressor()),
    ])
