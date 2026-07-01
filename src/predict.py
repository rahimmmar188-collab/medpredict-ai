"""
MedPredict AI — Prediction Module
------------------------------------
Loads pre-trained models and produces:
  - Disease probability (0.0 - 1.0)
  - Risk level (Low / Medium / High)
  - Top 3 contributing clinical features (explanation)
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
# Works both locally and on Vercel (/var/task/ is the project root on Vercel)
current_dir  = os.path.dirname(os.path.abspath(__file__))
# current_dir is <project>/src locally, so go up one level to get project root
# On Vercel current_dir will be /var/task/src
project_dir  = os.path.dirname(current_dir)
models_dir   = os.path.join(project_dir, 'models')


# ── Lazy-loaded globals (loaded once per process) ─────────────────────────────
preprocessor     = None
feature_engineer = None
models           = {}
TARGET_COLS      = ['Diabetes', 'Heart_Disease', 'Liver_Disease']


def load_artifacts():
    """Load all model artifacts into module-level globals (idempotent)."""
    global preprocessor, feature_engineer, models

    if preprocessor is not None:
        return  # Already loaded

    sys.path.insert(0, current_dir)
    from preprocessing import DataPreprocessor

    preprocessor     = DataPreprocessor.load(os.path.join(models_dir, 'preprocessor.joblib'))
    feature_engineer = joblib.load(os.path.join(models_dir, 'feature_engineer.joblib'))

    for disease in TARGET_COLS:
        model_path = os.path.join(models_dir, f'model_{disease}.joblib')
        models[disease] = joblib.load(model_path)

    print("All model artifacts loaded successfully.")


# ─────────────────────────────────────────────────────────────────────────────
def _get_explanations(model, X_arr: np.ndarray, feature_names: list) -> list:
    """
    Generate top-3 feature contribution explanations using two strategies:
    1. Extract averaged feature_importances_ from internal tree models (fast)
    2. Fall back to marginal-contribution perturbation (still fast)
    """
    importances = None

    # Strategy 1: internal importances from calibrated VotingClassifier
    try:
        if hasattr(model, 'calibrated_classifiers_'):
            imp_lists = []
            for cal in model.calibrated_classifiers_:
                est = getattr(cal, 'estimator', None) or getattr(cal, 'base_estimator', None)
                if est is None:
                    continue
                if hasattr(est, 'named_estimators_'):      # VotingClassifier
                    for name, sub in est.named_estimators_.items():
                        if hasattr(sub, 'feature_importances_'):
                            imp_lists.append(sub.feature_importances_)
                elif hasattr(est, 'feature_importances_'):
                    imp_lists.append(est.feature_importances_)

            if imp_lists:
                importances = np.mean(imp_lists, axis=0)
    except Exception:
        pass

    if importances is not None and len(importances) == len(feature_names):
        prob = float(model.predict_proba(X_arr)[0][1])
        contributions = []
        for i, name in enumerate(feature_names):
            imp = float(importances[i])
            if imp > 0.005:
                feat_val  = float(X_arr[0, i])
                direction = "increases" if feat_val >= 0 else "decreases"
                contributions.append({
                    "feature": name,
                    "impact":  direction,
                    "value":   round(imp * 100, 1),
                })
        contributions.sort(key=lambda x: x["value"], reverse=True)
        return contributions[:3]

    # Strategy 2: marginal contribution by zeroing out each feature
    try:
        baseline = float(model.predict_proba(X_arr)[0][1])
        contributions = []
        for i in range(min(X_arr.shape[1], len(feature_names))):
            X_p = X_arr.copy()
            X_p[0, i] = 0.0
            delta = baseline - float(model.predict_proba(X_p)[0][1])
            if abs(delta) > 0.005:
                contributions.append({
                    "feature": feature_names[i],
                    "impact":  "increases" if delta > 0 else "decreases",
                    "value":   round(abs(delta) * 100, 1),
                })
        contributions.sort(key=lambda x: x["value"], reverse=True)
        return contributions[:3]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
def predict(patient_data: dict) -> dict:
    """
    Run full prediction pipeline for a single patient.

    Args:
        patient_data: dict with 19 clinical feature keys

    Returns:
        dict keyed by disease name, each containing:
            probability (float 0-1), risk_level (str), explanation (list)
    """
    load_artifacts()

    df          = pd.DataFrame([patient_data])
    X_processed = preprocessor.transform(df)

    results = {}
    for disease in TARGET_COLS:
        X_disease     = feature_engineer.transform(X_processed, disease)
        model         = models[disease]
        prob          = float(model.predict_proba(X_disease)[0][1])
        feature_names = feature_engineer.selected_features.get(disease, [])
        explanation   = _get_explanations(model, X_disease, feature_names)

        # Risk thresholds — Low < 35%, Medium 35-65%, High > 65%
        if prob >= 0.65:
            risk = "High"
        elif prob >= 0.35:
            risk = "Medium"
        else:
            risk = "Low"

        results[disease] = {
            "probability": round(prob, 4),
            "risk_level":  risk,
            "explanation": explanation,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick self-test
    sample_high_risk = {
        'Age': 68, 'Gender': 'Male', 'BMI': 36.5,
        'Blood_Pressure_Systolic': 165, 'Blood_Pressure_Diastolic': 105,
        'Glucose': 290, 'HbA1c': 9.8,
        'Cholesterol': 280, 'HDL_Cholesterol': 28, 'LDL_Cholesterol': 195, 'Triglycerides': 380,
        'Creatinine': 2.2, 'ALT': 180, 'AST': 220,
        'Smoking': 1, 'Physical_Activity': 1, 'Alcohol_Use': 2,
        'Family_History_Diabetes': 1, 'Family_History_Heart': 1,
    }
    sample_low_risk = {
        'Age': 28, 'Gender': 'Female', 'BMI': 21.0,
        'Blood_Pressure_Systolic': 108, 'Blood_Pressure_Diastolic': 68,
        'Glucose': 82, 'HbA1c': 4.8,
        'Cholesterol': 155, 'HDL_Cholesterol': 72, 'LDL_Cholesterol': 72, 'Triglycerides': 88,
        'Creatinine': 0.7, 'ALT': 18, 'AST': 15,
        'Smoking': 0, 'Physical_Activity': 4, 'Alcohol_Use': 0,
        'Family_History_Diabetes': 0, 'Family_History_Heart': 0,
    }

    print("=== HIGH-RISK PATIENT ===")
    r = predict(sample_high_risk)
    for disease, info in r.items():
        print(f"  {disease}: {info['probability']*100:.1f}% - {info['risk_level']} Risk")
        for e in info['explanation']:
            print(f"    -> {e['feature']}: {e['impact']} by {e['value']}%")

    print("\n=== LOW-RISK PATIENT ===")
    r2 = predict(sample_low_risk)
    for disease, info in r2.items():
        print(f"  {disease}: {info['probability']*100:.1f}% - {info['risk_level']} Risk")
