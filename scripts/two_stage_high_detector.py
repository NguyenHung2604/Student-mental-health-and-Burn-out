from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

# reuse helpers from train_burnout_model.py
import sys
from pathlib import Path as _Path
# ensure project root is on sys.path so we can import train_burnout_model
sys.path.append(str(_Path(__file__).resolve().parent.parent))

from train_burnout_model import (
    DATA_PATH,
    FEATURES,
    TARGET,
    RISK_LEVEL_TO_INT,
    INT_TO_RISK_LEVEL,
    build_preprocessor,
)
from xgboost import XGBClassifier


def load_df(sample_size: int | None):
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(DATA_PATH)
    df = pd.read_csv(DATA_PATH, usecols=FEATURES + [TARGET])
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].map(RISK_LEVEL_TO_INT).astype(int)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df


def build_stage_pipeline(model_name: str = "xgboost") -> Pipeline:
    if model_name == "xgboost":
        clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=1)
    else:
        raise ValueError("Unsupported model")
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument(
        "--stage1-oversample",
        action="store_true",
        help="Oversample minority 'High' class for stage1 training (simple duplication).",
    )
    parser.add_argument(
        "--stage1-oversample-multiplier",
        type=int,
        default=5,
        help="How many times to duplicate minority-class samples when --stage1-oversample is used.",
    )
    parser.add_argument(
        "--stage1-high-weight",
        type=float,
        default=1.0,
        help="Sample weight multiplier for 'High' in stage1 training.",
    )
    args = parser.parse_args()

    df = load_df(args.sample_size)

    X = df[FEATURES]
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Stage 1: binary classifier for High vs NotHigh
    high_label = RISK_LEVEL_TO_INT["High"]
    y_train_stage1 = (y_train == high_label).astype(int)
    y_valid_stage1 = (y_valid == high_label).astype(int)
    y_test_stage1 = (y_test == high_label).astype(int)
    X_train_stage1 = X_train
    # optionally oversample stage1 minority class
    if args.stage1_oversample:
        if args.stage1_oversample_multiplier < 1:
            raise ValueError("--stage1-oversample-multiplier must be >= 1")
        mask_high = y_train_stage1.values == 1
        n_high = mask_high.sum()
        if n_high == 0:
            print("No 'High' examples in stage1 training set to oversample.")
        else:
            reps = args.stage1_oversample_multiplier - 1
            if reps > 0:
                X_high = X_train[mask_high]
                y_high = y_train_stage1[mask_high]
                X_train_stage1 = pd.concat([X_train] + [X_high] * reps, ignore_index=True)
                y_train_stage1 = pd.concat([y_train_stage1] + [y_high] * reps, ignore_index=True)
                print(f"Stage1 oversampled: duplicated {n_high} 'High' rows x{reps}")

    stage1 = build_stage_pipeline("xgboost")
    # compute sample weights for stage1 if requested
    sample_weight_stage1 = None
    if args.stage1_high_weight and args.stage1_high_weight != 1.0:
        sample_weight_stage1 = np.ones(len(y_train_stage1), dtype=float)
        sample_weight_stage1[y_train_stage1.values == 1] *= args.stage1_high_weight
    if sample_weight_stage1 is not None:
        stage1.fit(X_train_stage1, y_train_stage1, model__sample_weight=sample_weight_stage1)
    else:
        stage1.fit(X_train_stage1, y_train_stage1)

    # Stage 2: multiclass classifier on all data (or could be trained on NotHigh only)
    stage2 = build_stage_pipeline("xgboost")
    stage2.fit(X_train, y_train)

    # Evaluate combined: if stage1 predicts High -> final = High else use stage2
    stage1_pred_test = stage1.predict(X_test)
    stage2_pred_test = stage2.predict(X_test)

    final_pred = []
    for s1, s2 in zip(stage1_pred_test, stage2_pred_test):
        if s1 == 1:
            final_pred.append(high_label)
        else:
            final_pred.append(int(s2))

    final_pred = np.array(final_pred)

    print("Two-stage classifier evaluation (sample-size=%d):" % len(df))
    print(f"Accuracy: {accuracy_score(y_test, final_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, final_pred):.4f}")
    print(f"F1 Macro: {f1_score(y_test, final_pred, average='macro'):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, final_pred, labels=[0,1,2], target_names=list(RISK_LEVEL_TO_INT.keys())))
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, final_pred))


if __name__ == '__main__':
    main()
