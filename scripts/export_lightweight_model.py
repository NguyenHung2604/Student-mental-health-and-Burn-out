from __future__ import annotations

import json
from pathlib import Path

import joblib


SOURCE_MODEL_PATH = Path("models/burnout_model.joblib")
OUTPUT_MODEL_PATH = Path("models/burnout_model_light.json")


def export_tree(tree) -> list[dict]:
    nodes = []
    for node in tree.nodes:
        nodes.append(
            {
                "value": float(node["value"]),
                "feature_idx": int(node["feature_idx"]),
                "threshold": float(node["num_threshold"]),
                "missing_go_to_left": bool(node["missing_go_to_left"]),
                "left": int(node["left"]),
                "right": int(node["right"]),
                "is_leaf": bool(node["is_leaf"]),
            }
        )
    return nodes


def main() -> None:
    artifact = joblib.load(SOURCE_MODEL_PATH)
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    categorical_encoder = preprocessor.named_transformers_["categorical"]
    gender_categories = list(categorical_encoder.categories_[0])
    numeric_features = list(preprocessor.transformers_[1][2])
    transformed_features = [f"gender={category}" for category in gender_categories] + numeric_features

    lightweight_artifact = {
        "model_type": "sklearn_hist_gradient_boosting_classifier_export",
        "features": list(artifact["features"]),
        "target": artifact.get("target"),
        "target_source": artifact.get("target_source"),
        "level_thresholds": artifact.get("level_thresholds", {}),
        "gender_categories": gender_categories,
        "numeric_features": numeric_features,
        "transformed_features": transformed_features,
        "classes": [int(value) for value in model.classes_],
        "baseline_prediction": [float(value) for value in model._baseline_prediction[0]],
        "label_mapping": {str(key): value for key, value in artifact.get("label_mapping", {}).items()},
        "predictors": [
            [export_tree(class_tree) for class_tree in iteration]
            for iteration in model._predictors
        ],
        "metrics": artifact.get("metrics", {}),
        "best_params": artifact.get("best_params", {}),
    }

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_PATH.write_text(
        json.dumps(lightweight_artifact, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Exported lightweight model to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
