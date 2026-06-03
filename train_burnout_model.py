from pathlib import Path
import os


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("student_mental_health_burnout_1M.csv")
MODEL_PATH = Path("models/burnout_model.joblib")

RANDOM_STATE = 42
TARGET_SCORE = "burnout_score"
TARGET = "burnout_level"
RISK_LEVEL_ORDER = ["Low", "Medium", "High"]
RISK_LEVEL_TO_INT = {label: index for index, label in enumerate(RISK_LEVEL_ORDER)}
INT_TO_RISK_LEVEL = {index: label for label, index in RISK_LEVEL_TO_INT.items()}

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


def build_model(params: dict | None = None):
    params = params or {}
    defaults = {
        "max_iter": 150,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "random_state": RANDOM_STATE,
    }
    return HistGradientBoostingClassifier(**(defaults | params))


def build_pipeline(params: dict | None = None) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", build_model(params))])


def burnout_level(score: float) -> int:
    if score < 3:
        return RISK_LEVEL_TO_INT["Low"]
    if score < 6:
        return RISK_LEVEL_TO_INT["Medium"]
    return RISK_LEVEL_TO_INT["High"]


def make_high_sample_weight(y: pd.Series, high_weight: float) -> np.ndarray:
    sample_weight = np.ones(len(y), dtype=float)
    high_label = RISK_LEVEL_TO_INT["High"]
    sample_weight[y.values == high_label] *= high_weight
    return sample_weight


def oversample_high(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    multiplier: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    high_label = RISK_LEVEL_TO_INT["High"]
    mask_high = y_train.values == high_label
    n_high = int(mask_high.sum())

    if n_high == 0 or multiplier <= 1:
        return x_train.reset_index(drop=True), y_train.reset_index(drop=True)

    x_high = x_train[mask_high]
    y_high = y_train[mask_high]
    x_balanced = pd.concat([x_train] + [x_high] * (multiplier - 1), ignore_index=True)
    y_balanced = pd.concat([y_train] + [y_high] * (multiplier - 1), ignore_index=True)
    print(f"-> Oversampling High: duplicated {n_high} rows x{multiplier - 1}.")
    return x_balanced, y_balanced


def evaluate_pipeline(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(x_test)
    labels = list(INT_TO_RISK_LEVEL.keys())
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=RISK_LEVEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }


def print_class_distribution(title: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    print(f"\n{title}")
    for label_id, count in counts.items():
        label = INT_TO_RISK_LEVEL[int(label_id)]
        percent = count / len(y) * 100
        print(f"- {label}: {count} ({percent:.2f}%)")


def print_evaluation(metrics: dict) -> None:
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Test F1 Macro: {metrics['f1_macro']:.4f}")

    print("\nPer-class report:")
    for label in RISK_LEVEL_ORDER:
        values = metrics["classification_report"][label]
        print(
            f"- {label}: precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, f1={values['f1-score']:.4f}, "
            f"support={int(values['support'])}"
        )

    print("\nConfusion matrix (rows=true, cols=pred; Low, Medium, High):")
    for row in metrics["confusion_matrix"]:
        print(row)


def objective(
    trial,
    x_train,
    y_train,
    x_valid,
    y_valid,
    high_weight: float,
) -> float:
    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
    }
    pipeline = build_pipeline(params)
    sample_weight = make_high_sample_weight(y_train, high_weight)
    pipeline.fit(x_train, y_train, model__sample_weight=sample_weight)
    return evaluate_pipeline(pipeline, x_valid, y_valid)["f1_macro"]


def tune_model(
    x_train,
    y_train,
    x_valid,
    y_valid,
    n_trials,
    high_weight,
) -> tuple[dict, float]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, x_valid, y_valid, high_weight),
        n_trials=n_trials,
    )
    return study.best_params, float(study.best_value)


def load_data(sample_size: int | None) -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Put the CSV in the same folder as this script."
        )

    df = pd.read_csv(DATA_PATH, usecols=FEATURES + [TARGET_SCORE])
    df = df.dropna(subset=FEATURES + [TARGET_SCORE])
    df[TARGET] = df[TARGET_SCORE].apply(burnout_level).astype(int)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    return df[FEATURES], df[TARGET]


def save_artifact(pipeline, params, metrics, sample_size) -> None:
    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "target_source": TARGET_SCORE,
        "level_thresholds": {
            "Low": "burnout_score < 3",
            "Medium": "3 <= burnout_score < 6",
            "High": "burnout_score >= 6",
        },
        "model_name": "hist_gradient",
        "best_params": params,
        "tuned": True,
        "metrics": metrics | {"sample_size": int(sample_size)},
        "label_mapping": INT_TO_RISK_LEVEL,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model artifact to {MODEL_PATH}")


def main() -> None:
    sample_size = 1_000_000
    high_weight = 5.0
    high_oversample_multiplier = 3
    trials = 10

    print(f"Loading up to {sample_size} rows from {DATA_PATH}...")
    x, y = load_data(sample_size)
    print_class_distribution("Original class distribution:", y)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train_raw, x_valid, y_train_raw, y_valid = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    x_train, y_train = oversample_high(x_train_raw, y_train_raw, high_oversample_multiplier)
    print_class_distribution("Training distribution after High oversampling:", y_train)

    print("\nTuning HistGradientBoosting with weighted High samples...")
    best_params, valid_f1 = tune_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        trials,
        high_weight,
    )
    print(f"Best validation macro F1: {valid_f1:.4f}")
    print(f"Best params: {best_params}")

    print("\nTraining final model on train+valid, with High oversampling and weights...")
    x_final, y_final = oversample_high(x_temp, y_temp, high_oversample_multiplier)
    sample_weight_final = make_high_sample_weight(y_final, high_weight)
    final_pipeline = build_pipeline(best_params)
    final_pipeline.fit(x_final, y_final, model__sample_weight=sample_weight_final)

    test_metrics = evaluate_pipeline(final_pipeline, x_test, y_test)
    save_artifact(final_pipeline, best_params, test_metrics, len(x))

    print("\nFinal test metrics:")
    print("Algorithm: HistGradientBoostingClassifier")
    print_evaluation(test_metrics)


if __name__ == "__main__":
    main()
