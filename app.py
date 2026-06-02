from pathlib import Path
import os
from typing import Annotated


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse


MODEL_PATH = Path("models/burnout_model.joblib")

app = FastAPI(title="Student Burnout Prediction")


def load_artifact():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. Run: python train_burnout_model.py"
        )
    return joblib.load(MODEL_PATH)


def burnout_level(score: float) -> str:
    if score < 3:
        return "Low"
    if score < 6:
        return "Medium"
    return "High"


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


def render_form(result: str = "", values: dict | None = None) -> str:
    form_values = DEFAULT_VALUES | (values or {})
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Burnout Prediction</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f7fb;
      color: #18202f;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px;
    }}
    h1 {{
      margin: 0 0 20px;
      font-size: 28px;
    }}
    form {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      background: #fff;
      border: 1px solid #dfe5ef;
      border-radius: 8px;
      padding: 20px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 14px;
      font-weight: 600;
    }}
    input, select {{
      height: 38px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 0 10px;
      font-size: 14px;
    }}
    button {{
      height: 42px;
      border: 0;
      border-radius: 6px;
      background: #176b87;
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .result {{
      margin: 18px 0 0;
      padding: 16px;
      background: #eaf7f3;
      border: 1px solid #a8d8c8;
      border-radius: 8px;
      font-size: 17px;
      font-weight: 700;
    }}
    .note {{
      margin-top: 14px;
      color: #596579;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Student Burnout Prediction</h1>
    <form method="get" action="/predict">
      <label>Age
        <input name="age" type="number" min="15" max="60" value="{form_values["age"]}" required>
      </label>
      <label>Gender
        <select name="gender" required>
          <option {selected("Male", form_values["gender"])}>Male</option>
          <option {selected("Female", form_values["gender"])}>Female</option>
          <option {selected("Other", form_values["gender"])}>Other</option>
        </select>
      </label>
      <label>Academic year
        <input name="academic_year" type="number" min="1" max="8" value="{form_values["academic_year"]}" required>
      </label>
      <label>Study hours per day
        <input name="study_hours_per_day" type="number" min="0" max="16" step="0.1" value="{form_values["study_hours_per_day"]}" required>
      </label>
      <label>Exam pressure (0-10)
        <input name="exam_pressure" type="number" min="0" max="10" step="0.1" value="{form_values["exam_pressure"]}" required>
      </label>
      <label>Academic performance (0-100)
        <input name="academic_performance" type="number" min="0" max="100" step="0.1" value="{form_values["academic_performance"]}" required>
      </label>
      <label>Stress level (0-10)
        <input name="stress_level" type="number" min="0" max="10" step="0.1" value="{form_values["stress_level"]}" required>
      </label>
      <label>Anxiety score (0-10)
        <input name="anxiety_score" type="number" min="0" max="10" step="0.1" value="{form_values["anxiety_score"]}" required>
      </label>
      <label>Depression score (0-10)
        <input name="depression_score" type="number" min="0" max="10" step="0.1" value="{form_values["depression_score"]}" required>
      </label>
      <label>Sleep hours
        <input name="sleep_hours" type="number" min="0" max="14" step="0.1" value="{form_values["sleep_hours"]}" required>
      </label>
      <label>Physical activity (hours/day)
        <input name="physical_activity" type="number" min="0" max="10" step="0.1" value="{form_values["physical_activity"]}" required>
      </label>
      <label>Social support (0-10)
        <input name="social_support" type="number" min="0" max="10" step="0.1" value="{form_values["social_support"]}" required>
      </label>
      <label>Screen time (hours/day)
        <input name="screen_time" type="number" min="0" max="20" step="0.1" value="{form_values["screen_time"]}" required>
      </label>
      <label>Internet usage (hours/day)
        <input name="internet_usage" type="number" min="0" max="20" step="0.1" value="{form_values["internet_usage"]}" required>
      </label>
      <label>Financial stress (0-10)
        <input name="financial_stress" type="number" min="0" max="10" step="0.1" value="{form_values["financial_stress"]}" required>
      </label>
      <label>Family expectation (0-10)
        <input name="family_expectation" type="number" min="0" max="10" step="0.1" value="{form_values["family_expectation"]}" required>
      </label>
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
    input_data = pd.DataFrame([values])

    score = float(artifact["pipeline"].predict(input_data)[0])
    score = max(0.0, min(10.0, score))
    level = burnout_level(score)
    result = (
        f'<div class="result">Predicted burnout score: {score:.2f}/10 '
        f'| Level: {level}</div>'
    )
    return render_form(result, values)
