import joblib
import pandas as pd
import numpy as np
import sys, os

# ensure project root is on sys.path so we can import train_burnout_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

artifact = joblib.load('models/burnout_model.joblib')
pipeline = artifact['pipeline']
FEATURES = artifact['features']
TARGET = artifact['target']

# Quick-run settings (reduced sample and repeats for speed)
SAMPLE_SIZE = 5000
N_REPEATS = 5

# load and prepare data
from train_burnout_model import apply_feature_engineering

df = pd.read_csv('student_mental_health_burnout_1M.csv', usecols=FEATURES + [TARGET])
df = apply_feature_engineering(df)
df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y = df[TARGET].clip(0, 10)

from sklearn.model_selection import train_test_split
if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
    X = X.sample(n=SAMPLE_SIZE, random_state=42)
    y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# permutation importance (MAE) — reduced repeats for speed
res = permutation_importance(pipeline, X_test, y_test, n_repeats=N_REPEATS, random_state=42, scoring='neg_mean_absolute_error', n_jobs=1)

# build feature names after preprocessing
pre = pipeline.named_steps['preprocessor']
cat_ohe = pre.named_transformers_['categorical']
cat_names = list(cat_ohe.get_feature_names_out(['gender']))
num_names = [f for f in FEATURES if f != 'gender']
feature_names = cat_names + num_names

print('\nPermutation importance (mean change in neg-MAE):')
for name, imp in sorted(zip(feature_names, res.importances_mean), key=lambda x: -abs(x[1])):
    print(f"{name:30s} {imp:+.6f}")

# built-in importance if available
model = pipeline.named_steps['model']
if hasattr(model, 'feature_importances_'):
    try:
        importances = model.feature_importances_
        print('\nModel.feature_importances_:')
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -abs(x[1])):
            print(f"{name:30s} {imp:.6f}")
    except Exception as e:
        print('Could not get model.feature_importances_', e)
else:
    print('\nModel has no feature_importances_ attribute')

# quick ablation: set feature to median and measure MAE change for top 5
base_preds = np.clip(pipeline.predict(X_test),0,10)
base_mae = mean_absolute_error(y_test, base_preds)
print(f"\nBase test MAE: {base_mae:.6f}")
for name, _ in sorted(zip(feature_names, res.importances_mean), key=lambda x: -abs(x[1]))[:5]:
    # try to ablate original column if one-hot expanded
    col = name
    if '_' in name and name.split('_')[0] in X_test.columns:
        col = name.split('_')[0]
    if col not in X_test.columns:
        # skip if can't ablate easily
        continue
    X_mod = X_test.copy()
    X_mod[col] = X_mod[col].median()
    preds = np.clip(pipeline.predict(X_mod),0,10)
    mae = mean_absolute_error(y_test, preds)
    print(f"Ablate {name:30s} -> MAE: {mae:.6f} (delta {mae-base_mae:+.6f})")
