from pathlib import Path
import argparse
import os


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = Path("student_mental_health_burnout_1M.csv")
MODEL_PATH = Path("models/burnout_model.joblib")
RANDOM_STATE = 42

TARGET = "burnout_score"
FEATURES = [
    "age",
    "gender",
    "academic_year",
    "study_hours_per_day",
    "exam_pressure",
    "academic_performance",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "sleep_hours",
    "physical_activity",
    "social_support",
    "screen_time",
    "internet_usage",
    "financial_stress",
    "family_expectation",
]

NUMERIC_FEATURES = [feature for feature in FEATURES if feature != "gender"]
CATEGORICAL_FEATURES = ["gender"]
DEFAULT_MODELS = ["hist_gradient", "random_forest", "xgboost", "lightgbm"]


def burnout_level(score: float) -> str:
    if score < 3:
        return "Low"
    if score < 6:
        return "Medium"
    return "High"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_model(model_name: str, params: dict | None = None):
    params = params or {}

    if model_name == "hist_gradient":
        defaults = {
            "max_iter": 200,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.01,
            "random_state": RANDOM_STATE,
        }
        return HistGradientBoostingRegressor(**(defaults | params))

    if model_name == "random_forest":
        defaults = {
            "n_estimators": 200,
            "max_depth": 18,
            "min_samples_leaf": 5,
            "max_features": 0.8,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        return RandomForestRegressor(**(defaults | params))

    if model_name == "xgboost":
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        return XGBRegressor(**(defaults | params))

    if model_name == "lightgbm":
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
            "verbose": -1,
        }
        return LGBMRegressor(**(defaults | params))

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str = "hist_gradient", params: dict | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model(model_name, params)),
        ]
    )


def suggest_params(model_name: str, trial: optuna.Trial) -> dict:
    if model_name == "hist_gradient":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 8, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }

    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_pipeline(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = np.clip(pipeline.predict(x_test), 0, 10)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def objective(
    trial: optuna.Trial,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> float:
    params = suggest_params(model_name, trial)
    pipeline = build_pipeline(model_name, params)
    pipeline.fit(x_train, y_train)
    metrics = evaluate_pipeline(pipeline, x_valid, y_valid)
    return metrics["mae"]


def tune_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int,
) -> tuple[dict, float]:
    study = optuna.create_study(
        direction="minimize",
        study_name=f"burnout_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, model_name, x_train, y_train, x_valid, y_valid),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study.best_params, float(study.best_value)


def load_data(sample_size: int | None) -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, usecols=FEATURES + [TARGET])
    df = df.dropna(subset=FEATURES + [TARGET])

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    return df[FEATURES], df[TARGET].clip(0, 10)


def save_artifact(
    pipeline: Pipeline,
    model_name: str,
    params: dict,
    metrics: dict,
    sample_size: int,
    tuned: bool,
) -> None:
    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "model_name": model_name,
        "best_params": params,
        "tuned": tuned,
        "metrics": metrics | {"sample_size": int(sample_size)},
        "level_thresholds": {
            "Low": "score < 3",
            "Medium": "3 <= score < 6",
            "High": "score >= 6",
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a burnout prediction model.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hist_gradient"],
        choices=DEFAULT_MODELS,
        help="Models to train or tune.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Use Optuna to tune hyperparameters for each selected model.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials per model.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Number of rows to sample. Use 0 for all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = None if args.sample_size == 0 else args.sample_size
    x, y = load_data(sample_size)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    candidates = []

    for model_name in args.models:
        if args.tune:
            print(f"\nTuning {model_name} with {args.trials} Optuna trials...")
            params, valid_mae = tune_model(
                model_name,
                x_train,
                y_train,
                x_valid,
                y_valid,
                args.trials,
            )
        else:
            print(f"\nTraining {model_name} with default parameters...")
            params = {}
            pipeline = build_pipeline(model_name, params)
            pipeline.fit(x_train, y_train)
            valid_mae = evaluate_pipeline(pipeline, x_valid, y_valid)["mae"]

        print(f"{model_name} validation MAE: {valid_mae:.4f}")
        print(f"{model_name} params: {params}")
        candidates.append(
            {
                "model_name": model_name,
                "params": params,
                "valid_mae": valid_mae,
            }
        )

    best_candidate = min(candidates, key=lambda candidate: candidate["valid_mae"])
    best_model_name = best_candidate["model_name"]
    best_params = best_candidate["params"]

    print(f"\nBest model on validation set: {best_model_name}")
    print(f"Best validation MAE: {best_candidate['valid_mae']:.4f}")

    final_pipeline = build_pipeline(best_model_name, best_params)
    final_pipeline.fit(x_temp, y_temp)
    test_metrics = evaluate_pipeline(final_pipeline, x_test, y_test)

    save_artifact(
        pipeline=final_pipeline,
        model_name=best_model_name,
        params=best_params,
        metrics=test_metrics,
        sample_size=len(x),
        tuned=args.tune,
    )

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Final model: {best_model_name}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
