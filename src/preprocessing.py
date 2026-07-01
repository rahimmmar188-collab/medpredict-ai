"""
Enhanced Data Preprocessor
---------------------------
Handles the full 19-feature clinical dataset:
- Robust imputation (median for numerics, mode for categoricals)
- RobustScaler (outlier-resistant) for numeric features
- Label encoding for Gender
- Persists all fitted state for inference
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib
import os


class DataPreprocessor:
    def __init__(self):
        self.num_imputer  = SimpleImputer(strategy='median')
        self.cat_imputer  = SimpleImputer(strategy='most_frequent')
        self.scaler       = RobustScaler()          # outlier-robust vs StandardScaler
        self.label_encoders = {}
        self.numeric_features    = []
        self.categorical_features = []

    # ──────────────────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit all preprocessing steps on training data and return transformed df."""
        df_processed = df.copy()

        target_cols = ['Diabetes', 'Heart_Disease', 'Liver_Disease']

        # Identify column types (exclude targets)
        self.numeric_features = [
            c for c in df_processed.select_dtypes(include=['int64', 'float64']).columns
            if c not in target_cols
        ]
        self.categorical_features = [
            c for c in df_processed.select_dtypes(include=['object', 'category']).columns
            if c not in target_cols
        ]

        # ── Numeric: impute → scale ──────────────────────────────────────────
        if self.numeric_features:
            df_processed[self.numeric_features] = self.num_imputer.fit_transform(
                df_processed[self.numeric_features]
            )
            df_processed[self.numeric_features] = self.scaler.fit_transform(
                df_processed[self.numeric_features]
            )

        # ── Categorical: impute → label-encode ──────────────────────────────
        if self.categorical_features:
            df_processed[self.categorical_features] = self.cat_imputer.fit_transform(
                df_processed[self.categorical_features]
            )
            for col in self.categorical_features:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le

        return df_processed

    # ──────────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new inference data using the already-fitted preprocessor."""
        df_processed = df.copy()

        # Ensure the dataframe has ALL expected columns (fill missing with NaN)
        for col in self.numeric_features + self.categorical_features:
            if col not in df_processed.columns:
                df_processed[col] = np.nan

        if self.numeric_features:
            df_processed[self.numeric_features] = self.num_imputer.transform(
                df_processed[self.numeric_features]
            )
            df_processed[self.numeric_features] = self.scaler.transform(
                df_processed[self.numeric_features]
            )

        if self.categorical_features:
            df_processed[self.categorical_features] = self.cat_imputer.transform(
                df_processed[self.categorical_features]
            )
            for col in self.categorical_features:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    col_data = df_processed[col].astype(str)
                    # Gracefully handle unseen categories
                    known_classes = set(le.classes_)
                    col_data = col_data.map(
                        lambda x: x if x in known_classes else le.classes_[0]
                    )
                    df_processed[col] = le.transform(col_data)

        return df_processed

    # ──────────────────────────────────────────────────────────────────────────
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
