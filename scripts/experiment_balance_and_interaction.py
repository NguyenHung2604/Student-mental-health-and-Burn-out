import joblib
import pandas as pd
import numpy as np
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, f1_score, accuracy_score

# ensure project root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train_burnout_model as m

MODEL_ARTIFACT = 'models/burnout_model.joblib'
SAMPLE_SIZE = 50000
WEIGHT_HIGH = 5.0

# load data
X, y = m.load_data(sample_size=None)  # load full then sample below
if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
    X = X.sample(n=SAMPLE_SIZE, random_state=m.RANDOM_STATE)
    y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=m.RANDOM_STATE)

# helper: evaluate
def eval_pipeline(pipeline, X_test, y_test):
    preds = np.clip(pipeline.predict(X_test), 0, 10)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    y_true_labels = pd.Series(y_test).apply(m.burnout_level)
    y_pred_labels = pd.Series(preds).apply(m.burnout_level)
    report = classification_report(y_true_labels, y_pred_labels, digits=4)
    f1_macro = f1_score(y_true_labels, y_pred_labels, average='macro')
    acc = accuracy_score(y_true_labels, y_pred_labels)
    return {
        'mae': mae,
        'r2': r2,
        'f1_macro': f1_macro,
        'accuracy': acc,
        'classification_report': report,
    }

# choose model to test — use saved best if available
best_model = 'xgboost'
if os.path.exists(MODEL_ARTIFACT):
    art = joblib.load(MODEL_ARTIFACT)
    best_model = art.get('model_name', best_model)

print('Running experiments for model:', best_model)

# Baseline
print('\n1) Baseline (no weighting, no interaction)')
pipe = m.build_pipeline(best_model, params={})
pipe.fit(X_train, y_train)
res_base = eval_pipeline(pipe, X_test, y_test)
print('Baseline:', res_base)

# 2) Sample-weighting to upweight High labels
print('\n2) Sample-weighting (High x {})'.format(WEIGHT_HIGH))
# compute sample weights on training set
labels_train = pd.Series(y_train).apply(m.burnout_level)
sample_weight = np.ones(len(y_train), dtype=float)
sample_weight[labels_train == 'High'] = WEIGHT_HIGH
pipe_w = m.build_pipeline(best_model, params={})
# pass sample_weight to final estimator via pipeline.fit kwargs
pipe_w.fit(X_train, y_train, model__sample_weight=sample_weight)
res_weight = eval_pipeline(pipe_w, X_test, y_test)
print('Weighted:', res_weight)

# 3) Add interaction feature (anxiety_score * sleep_hours)
print('\n3) Feature interaction: anxiety_score * sleep_hours')
# create copies to avoid mutating global lists
X2 = X.copy()
X2['anxiety_sleep'] = X2['anxiety_score'] * X2['sleep_hours']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=m.RANDOM_STATE)
# temporarily add feature name to module feature lists used by build_pipeline
if 'anxiety_sleep' not in m.FEATURES:
    m.FEATURES.append('anxiety_sleep')
if 'anxiety_sleep' not in m.NUMERIC_FEATURES:
    m.NUMERIC_FEATURES.append('anxiety_sleep')
pipe_i = m.build_pipeline(best_model, params={})
pipe_i.fit(X2_train, y2_train)
res_inter = eval_pipeline(pipe_i, X2_test, y2_test)
print('Interaction:', res_inter)

# 4) Combine weighting + interaction
print('\n4) Weighting + interaction')
labels_i_train = pd.Series(y2_train).apply(m.burnout_level)
sample_weight_i = np.ones(len(y2_train), dtype=float)
sample_weight_i[labels_i_train == 'High'] = WEIGHT_HIGH
pipe_wi = m.build_pipeline(best_model, params={})
pipe_wi.fit(X2_train, y2_train, model__sample_weight=sample_weight_i)
res_wi = eval_pipeline(pipe_wi, X2_test, y2_test)
print('Weight+Interaction:', res_wi)

# print concise table
print('\nSummary (MAE, F1_macro, Accuracy):')
print('Baseline         ', f"{res_base['mae']:.4f}", f"{res_base['f1_macro']:.4f}", f"{res_base['accuracy']:.4f}")
print('Weighted         ', f"{res_weight['mae']:.4f}", f"{res_weight['f1_macro']:.4f}", f"{res_weight['accuracy']:.4f}")
print('Interaction      ', f"{res_inter['mae']:.4f}", f"{res_inter['f1_macro']:.4f}", f"{res_inter['accuracy']:.4f}")
print('Weight+Interaction', f"{res_wi['mae']:.4f}", f"{res_wi['f1_macro']:.4f}", f"{res_wi['accuracy']:.4f}")
