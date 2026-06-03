from pathlib import Path
import json
import math
import os
from typing import Annotated

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

MODEL_PATH = Path(__file__).resolve().parent / "models" / "burnout_model_light.json"

app = FastAPI(title="Student Burnout Prediction")

def load_artifact():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run scripts/export_lightweight_model.py first."
        )
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


ARTIFACT = load_artifact()

LEGACY_DEFAULT_VALUES = {
    "dropout_risk": 1.0,
    "mental_health_index": 7.0,
    "burnout_score": 2.0,
}


def build_features(values: dict) -> list[float]:
    gender = values["gender"]
    encoded_gender = [
        1.0 if gender == category else 0.0
        for category in ARTIFACT["gender_categories"]
    ]
    numeric_values = [
        float(values.get(feature, LEGACY_DEFAULT_VALUES.get(feature, 0.0)))
        for feature in ARTIFACT["numeric_features"]
    ]
    return encoded_gender + numeric_values


def predict_tree(nodes: list[dict], features: list[float]) -> float:
    index = 0
    while True:
        node = nodes[index]
        if node["is_leaf"]:
            return node["value"]

        value = features[node["feature_idx"]]
        if math.isnan(value):
            go_left = node["missing_go_to_left"]
        else:
            go_left = value <= node["threshold"]
        index = node["left"] if go_left else node["right"]


def softmax(raw_scores: list[float]) -> list[float]:
    max_score = max(raw_scores)
    exp_scores = [math.exp(score - max_score) for score in raw_scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def predict_burnout(values: dict) -> tuple[int, str, list[float]]:
    features = build_features(values)
    raw_scores = list(ARTIFACT["baseline_prediction"])
    for iteration in ARTIFACT["predictors"]:
        for class_index, tree in enumerate(iteration):
            raw_scores[class_index] += predict_tree(tree, features)

    probabilities = softmax(raw_scores)
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    prediction = ARTIFACT["classes"][best_index]
    label = ARTIFACT["label_mapping"].get(str(prediction), str(prediction))
    return prediction, label, probabilities

DEFAULT_VALUES = {
    "age": 22,
    "gender": "Male",
    "academic_year": 2,
    "study_hours_per_day": 5,
    "exam_pressure": 5,
    "academic_performance": 70,
    "stress_level": 5,
    "anxiety_score": 4,
    "depression_score": 3,
    "sleep_hours": 7,
    "physical_activity": 2,
    "social_support": 5,
    "screen_time": 5,
    "internet_usage": 5,
    "financial_stress": 4,
    "family_expectation": 5,
}

def selected(value: str, current: str) -> str:
    return "selected" if value == current else ""

def number_field(
    label: str,
    name: str,
    value: object,
    *,
    min_value: object | None = None,
    max_value: object | None = None,
    step: str = "0.1",
) -> str:
    min_attr = "" if min_value is None else f' min="{min_value}"'
    max_attr = "" if max_value is None else f' max="{max_value}"'
    return f"""
      <label class="field">
        <span>{label}</span>
        <input name="{name}" type="number"{min_attr}{max_attr} step="{step}" value="{value}" required>
      </label>
    """

def select_field(label: str, name: str, current: str, options: list[str]) -> str:
    options_html = "\n".join(
        f'<option {selected(option, current)}>{option}</option>' for option in options
    )
    return f"""
      <label class="field">
        <span>{label}</span>
        <select name="{name}" required>
          {options_html}
        </select>
      </label>
    """

def render_form(result: str = "", values: dict | None = None) -> str:
    form_values = DEFAULT_VALUES | (values or {})
    core_fields = "\n".join(
        [
            number_field("Age (15-60)", "age", form_values["age"], min_value=15, max_value=60, step="1"),
            select_field("Gender", "gender", form_values["gender"], ["Male", "Female", "Other"]),
            number_field("Academic year (1-8)", "academic_year", form_values["academic_year"], min_value=1, max_value=8, step="1"),
            number_field("Study hours/day (0-16)", "study_hours_per_day", form_values["study_hours_per_day"], min_value=0, max_value=16),
            number_field("Exam pressure (0-10)", "exam_pressure", form_values["exam_pressure"], min_value=0, max_value=10),
            number_field("Academic performance (0-100)", "academic_performance", form_values["academic_performance"], min_value=0, max_value=100),
            number_field("Stress level (0-10)", "stress_level", form_values["stress_level"], min_value=0, max_value=10),
            number_field("Anxiety score (0-10)", "anxiety_score", form_values["anxiety_score"], min_value=0, max_value=10),
            number_field("Depression score (0-10)", "depression_score", form_values["depression_score"], min_value=0, max_value=10),
            number_field("Sleep hours/day (0-14)", "sleep_hours", form_values["sleep_hours"], min_value=0, max_value=14),
        ]
    )
    advanced_fields = "\n".join(
        [
            number_field("Physical activity (0-10)", "physical_activity", form_values["physical_activity"], min_value=0, max_value=10),
            number_field("Social support (0-10)", "social_support", form_values["social_support"], min_value=0, max_value=10),
            number_field("Screen time hour/day (0-20)", "screen_time", form_values["screen_time"], min_value=0, max_value=20),
            number_field("Internet usage hours/day (0-20)", "internet_usage", form_values["internet_usage"], min_value=0, max_value=20),
            number_field("Financial stress (0-10)", "financial_stress", form_values["financial_stress"], min_value=0, max_value=10),
            number_field("Family expectation (0-10)", "family_expectation", form_values["family_expectation"], min_value=0, max_value=10),
        ]
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Burnout Prediction</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --line: #dce4ee;
      --text: #172033;
      --muted: #5c6b82;
      --accent: #176b87;
      --accent-soft: #e9f7fb;
      --result: #eaf7f3;
    }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef4fb 0%, var(--bg) 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(26px, 3vw, 36px);
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    form {{
      display: grid;
      gap: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(17, 31, 53, 0.06);
    }}
    .section {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .section-title {{
      margin: 2px 0 0;
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: #fbfdff;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--accent);
      list-style: none;
    }}
    summary::-webkit-details-marker {{
      display: none;
    }}
    .details-grid {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .field {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 600;
    }}
    .field span {{
      color: var(--text);
    }}
    input, select {{
      height: 40px;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 0 12px;
      font-size: 14px;
      background: #fff;
    }}
    button {{
      height: 42px;
      border: 0;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent), #0f5d75);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(23, 107, 135, 0.22);
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .result {{
      margin: 18px 0 0;
      padding: 16px;
      background: var(--result);
      border: 1px solid #a8d8c8;
      border-radius: 12px;
      font-size: 17px;
      font-weight: 700;
    }}
    .note {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 720px) {{
      main {{
        padding: 18px 12px 30px;
      }}
      form {{
        padding: 14px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Student Burnout Prediction</h1>
    <p class="subtitle">Fill the key fields first. Open advanced inputs only if you want to refine the prediction.</p>
    <form method="get" action="/predict">
      <div class="section">
        {core_fields}
      </div>
      <details>
        <summary>Advanced inputs</summary>
        <div class="details-grid">
          {advanced_fields}
        </div>
      </details>
      <button class="full" type="submit">Predict burnout</button>
    </form>
    {result}
    <p class="note">This model is for a course project and screening support only, not a medical diagnosis.</p>
  </main>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return render_form()

@app.get("/predict", response_class=HTMLResponse)
def predict(
    age: Annotated[int, Query()],
    gender: Annotated[str, Query()],
    academic_year: Annotated[int, Query()],
    study_hours_per_day: Annotated[float, Query()],
    exam_pressure: Annotated[float, Query()],
    academic_performance: Annotated[float, Query()],
    stress_level: Annotated[float, Query()],
    anxiety_score: Annotated[float, Query()],
    depression_score: Annotated[float, Query()],
    sleep_hours: Annotated[float, Query()],
    physical_activity: Annotated[float, Query()],
    social_support: Annotated[float, Query()],
    screen_time: Annotated[float, Query()],
    internet_usage: Annotated[float, Query()],
    financial_stress: Annotated[float, Query()],
    family_expectation: Annotated[float, Query()],
):
    artifact = load_artifact()
    
    # Keep request values aligned with the feature list stored in the exported model.
    values = {
        "age": age,
        "gender": gender,
        "academic_year": academic_year,
        "study_hours_per_day": study_hours_per_day,
        "exam_pressure": exam_pressure,
        "academic_performance": academic_performance,
        "stress_level": stress_level,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "sleep_hours": sleep_hours,
        "physical_activity": physical_activity,
        "social_support": social_support,
        "screen_time": screen_time,
        "internet_usage": internet_usage,
        "financial_stress": financial_stress,
        "family_expectation": family_expectation,
    }

    _, level, probabilities = predict_burnout(values)
    label_map = ARTIFACT["label_mapping"]
    pairs = [
        f"{label_map.get(str(class_id), class_id)}: {probability * 100:.1f}%"
        for class_id, probability in zip(ARTIFACT["classes"], probabilities)
    ]
    proba_str = " | Probabilities: " + ", ".join(pairs)
    result = f'<div class="result">Burnout level: {level}{proba_str}</div>'
    return render_form(result, values)
