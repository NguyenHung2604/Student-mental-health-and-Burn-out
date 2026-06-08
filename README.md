# Student-mental-health-and-Burn-out
A study based on students' data regarding GPA, sleeping duration, daily routine, .... to find out what are the main reason of burn out.

## Burnout prediction app

Train the model:

```bash
python train_burnout_model.py
```

Train and compare several models with default parameters:

```bash
python train_burnout_model.py --models hist_gradient random_forest xgboost lightgbm
```

Tune selected models with Optuna:

```bash
python train_burnout_model.py --tune --models xgboost lightgbm --trials 30 --sample-size 100000
```

Useful options:

- `--models`: choose one or more models from `hist_gradient`, `random_forest`, `xgboost`, `lightgbm`
- `--tune`: enable Optuna hyperparameter tuning
- `--trials`: number of Optuna trials per model
- `--sample-size`: number of rows used for training/tuning; use `0` for the full dataset

Run the web form:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`, fill in the student information, and submit the form to get a predicted burnout level:

- `Low`: score < 3
- `Medium`: 3 <= score < 6
- `High`: score >= 6

The model creates `burnout_level` from `burnout_score` as the target. `burnout_score`, `risk_level`, `mental_health_index`, and `dropout_risk` are not used as input features.
Link demo web:
https://student-mental-health-and-burn-out.vercel.app/
