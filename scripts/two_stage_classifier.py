import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

import train_burnout_model as m

SAMPLE_SIZE = 50000

# load data
X, y = m.load_data(sample_size=None)
if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
    X = X.sample(n=SAMPLE_SIZE, random_state=m.RANDOM_STATE)
    y = y.loc[X.index]

# binary label: High vs not
is_high = (y >= 7).astype(int)

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=m.RANDOM_STATE)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, is_high, test_size=0.2, random_state=m.RANDOM_STATE)

# build classifier pipeline
clf_pipeline = Pipeline(steps=[('preprocessor', m.build_preprocessor()), ('clf', RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=m.RANDOM_STATE))])

print('Training High vs Not classifier...')
clf_pipeline.fit(X_train_clf, y_train_clf)

preds_clf = clf_pipeline.predict(X_test_clf)
print('\nClassifier report (High=1):')
print(classification_report(y_test_clf, preds_clf, digits=4))
print('Confusion matrix:\n', confusion_matrix(y_test_clf, preds_clf))

# Evaluate combined: use regressor artifact if available
artifact_path = 'models/burnout_model.joblib'
if os.path.exists(artifact_path):
    art = joblib.load(artifact_path)
    reg_pipeline = art['pipeline']
    # predict regression on same test split as regressor test from earlier train/test split
    # We'll use X_test (from earlier y_test_reg)
    reg_preds = np.clip(reg_pipeline.predict(X_test), 0, 10)
    # classifier predictions on same X_test
    clf_preds_on_regsplit = clf_pipeline.predict(X_test)
    # combined: if classifier says High (1), force predicted score >=7 (e.g., max(reg_pred,7))
    combined_preds = reg_preds.copy()
    combined_preds[clf_preds_on_regsplit == 1] = np.maximum(combined_preds[clf_preds_on_regsplit == 1], 7.0)

    # regression metrics
    mae = mean_absolute_error(y_test_reg, combined_preds)
    print(f"\nCombined regressor+classifier MAE: {mae:.4f}")

    # evaluate binary detection of High using combined (i.e., threshold combined_preds >=7)
    y_true_high = (y_test_reg >= 7).astype(int)
    y_pred_high = (combined_preds >= 7).astype(int)
    print('\nCombined detection report:')
    print(classification_report(y_true_high, y_pred_high, digits=4))
    print('Confusion matrix:\n', confusion_matrix(y_true_high, y_pred_high))
else:
    print('No regressor artifact found at', artifact_path)
