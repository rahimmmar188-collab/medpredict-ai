"""
Feature Engineering Module
---------------------------
1. Derives clinically meaningful interaction features
2. Applies ANOVA F-value SelectKBest per disease for dimensionality reduction
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


class FeatureEngineer:
    def __init__(self, k='all'):
        self.k = k
        self.selectors = {}
        self.selected_features = {}

    # ──────────────────────────────────────────────────────────────────────────
    def _add_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add clinically meaningful derived / interaction features."""
        df = X.copy()

        # Pulse Pressure (cardiovascular risk marker)
        if 'Blood_Pressure_Systolic' in df and 'Blood_Pressure_Diastolic' in df:
            df['Pulse_Pressure'] = df['Blood_Pressure_Systolic'] - df['Blood_Pressure_Diastolic']

        # Mean Arterial Pressure
        if 'Blood_Pressure_Systolic' in df and 'Blood_Pressure_Diastolic' in df:
            df['MAP'] = (df['Blood_Pressure_Systolic'] + 2 * df['Blood_Pressure_Diastolic']) / 3

        # Cholesterol Ratio (total / HDL) — strong CV risk predictor
        if 'Cholesterol' in df and 'HDL_Cholesterol' in df:
            df['Cholesterol_Ratio'] = df['Cholesterol'] / (df['HDL_Cholesterol'] + 1e-6)

        # Non-HDL Cholesterol
        if 'Cholesterol' in df and 'HDL_Cholesterol' in df:
            df['Non_HDL'] = df['Cholesterol'] - df['HDL_Cholesterol']

        # BMI × Age interaction (obesity risk compounded by age)
        if 'BMI' in df and 'Age' in df:
            df['BMI_Age'] = df['BMI'] * df['Age'] / 100

        # Glucose × HbA1c (both capture glycemic burden differently)
        if 'Glucose' in df and 'HbA1c' in df:
            df['Glucose_HbA1c'] = df['Glucose'] * df['HbA1c'] / 100

        # AST/ALT Ratio (de Ritis ratio — liver fibrosis marker)
        if 'AST' in df and 'ALT' in df:
            df['AST_ALT_Ratio'] = df['AST'] / (df['ALT'] + 1e-6)

        return df

    # ──────────────────────────────────────────────────────────────────────────
    def fit_transform(self, X: pd.DataFrame, y_dict: dict) -> dict:
        """
        Derive features, then fit SelectKBest selectors per disease.
        Returns a dict of {disease: numpy array}.
        """
        X_derived = self._add_derived_features(X)
        X_engineered_dict = {}

        for disease, y in y_dict.items():
            selector = SelectKBest(score_func=f_classif, k=self.k)
            X_sel = selector.fit_transform(X_derived, y)

            self.selectors[disease] = selector
            if isinstance(X_derived, pd.DataFrame):
                mask = selector.get_support()
                self.selected_features[disease] = X_derived.columns[mask].tolist()

            X_engineered_dict[disease] = X_sel

        return X_engineered_dict

    # ──────────────────────────────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame, disease: str) -> np.ndarray:
        if disease not in self.selectors:
            raise ValueError(f"No fitted selector for '{disease}'")
        X_derived = self._add_derived_features(X)
        return self.selectors[disease].transform(X_derived)
