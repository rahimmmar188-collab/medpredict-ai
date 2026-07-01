"""
MedPredict AI — Fast Training Pipeline v3
-------------------------------------------
Uses fixed, well-tuned hyperparameters instead of RandomizedSearchCV.
Training completes in ~2-3 minutes vs 10+ minutes for tuning.
Quality is equivalent since the defaults are already well-chosen for
tabular clinical data.
"""

import pandas as pd
import numpy as np
import os, sys, json, warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)
    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
def get_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    wmap  = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([wmap[yi] for yi in y])


# ─────────────────────────────────────────────────────────────────────────────
def build_and_train(X_train, y_train):
    """Build calibrated ensemble with fixed best-practice hyperparameters."""
    sw = get_sample_weights(y_train)

    # Random Forest — balanced, 400 trees, moderate depth
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # Gradient Boosting — 300 trees, learning rate 0.08
    gbc = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        min_samples_leaf=10,
        random_state=42,
    )

    # Logistic Regression — L2, balanced
    lr = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    print("      Fitting Random Forest...", flush=True)
    rf.fit(X_train, y_train)

    print("      Fitting Gradient Boosting...", flush=True)
    gbc.fit(X_train, y_train, sample_weight=sw)

    print("      Fitting Logistic Regression...", flush=True)
    lr.fit(X_train, y_train)

    # Soft-voting ensemble (RF and GBC get 2x weight)
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gbc', gbc), ('lr', lr)],
        voting='soft',
        weights=[2.0, 2.0, 1.0],
    )

    # Calibrate — wrap with isotonic so raw probs are well-calibrated
    print("      Calibrating ensemble (isotonic)...", flush=True)
    calibrated = CalibratedClassifierCV(estimator=ensemble, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)

    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
def train_pipeline():
    print("=" * 60)
    print("  MedPredict AI -- Training Pipeline v3 (Fast)")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(os.path.dirname(current_dir), "data", "patient_records.csv")

    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        print("  Run: python data/generate_mock_data.py")
        return

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data...")
    df = pd.read_csv(data_path)
    print(f"      Shape: {df.shape}")
    target_cols = ["Diabetes", "Heart_Disease", "Liver_Disease"]
    for col in target_cols:
        pct = df[col].mean() * 100
        print(f"      {col}: {df[col].sum()} positive ({pct:.1f}%)")

    X_raw  = df.drop(columns=target_cols)
    y_dict = {col: df[col].values for col in target_cols}

    # ── 2. Preprocessing ───────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing (RobustScaler + imputation)...")
    preprocessor     = DataPreprocessor()
    X_processed_full = preprocessor.fit_transform(X_raw)

    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    os.makedirs(models_dir, exist_ok=True)
    preprocessor.save(os.path.join(models_dir, "preprocessor.joblib"))
    print("      Preprocessor saved.")

    # ── 3. Feature Engineering ─────────────────────────────────────────────────
    print("\n[3/5] Feature engineering...")
    engineer          = FeatureEngineer(k="all")
    X_engineered_dict = engineer.fit_transform(X_processed_full, y_dict)
    joblib.dump(engineer, os.path.join(models_dir, "feature_engineer.joblib"))
    for disease, Xe in X_engineered_dict.items():
        print(f"      {disease}: {Xe.shape[1]} features")

    # ── 4. Train per disease ───────────────────────────────────────────────────
    print("\n[4/5] Training ensemble models...\n")
    metrics_report = {}

    for disease in target_cols:
        print(f"  --- {disease.upper()} ---")
        X = X_engineered_dict[disease]
        y = y_dict[disease]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        print(f"      Train: {len(X_train)}, Test: {len(X_test)}")

        calibrated = build_and_train(X_train, y_train)

        metrics = evaluate_model(calibrated, X_test, y_test)
        print(f"\n      Test Metrics:")
        for k, v in metrics.items():
            print(f"        {k:<12}: {v:.4f}")

        model_path = os.path.join(models_dir, f"model_{disease}.joblib")
        joblib.dump(calibrated, model_path)
        print(f"      Saved -> {model_path}\n")

        metrics_report[disease] = {
            "model": "Calibrated VotingClassifier (RF + GBC + LR)",
            **metrics,
        }

    # ── 5. Save metrics ────────────────────────────────────────────────────────
    metrics_path = os.path.join(models_dir, "metrics_report.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=4)

    print("=" * 60)
    print("  [5/5] Training Complete -- Summary")
    print("=" * 60)
    for disease, m in metrics_report.items():
        print(f"  {disease:<20} AUC={m['roc_auc']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")
    print("\n  All models saved to /models")
    print("=" * 60)


if __name__ == "__main__":
    train_pipeline()
