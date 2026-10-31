"""
preprocess.py

Reusable preprocessing utilities for
Population Health Patient Segmentation
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)


class DataPreprocessor:

    def __init__(self):

        self.gender_encoder = LabelEncoder()

        self.scaler = StandardScaler()

    # =====================================
    # Feature Engineering
    # =====================================

    def create_features(self, df):

        df = df.copy()

        df["chronic_disease_count"] = (
            df["diabetes"]
            + df["hypertension"]
            + df["copd"]
            + df["mental_health"]
        )

        df["total_utilization"] = (
            df["visits_per_year"]
            + df["er_visits"]
            + df["admissions"]
        )

        df["cost_per_visit"] = np.where(
            df["visits_per_year"] > 0,
            df["total_cost"]
            / df["visits_per_year"],
            df["total_cost"]
        )

        return df

    # =====================================
    # Missing Values
    # =====================================

    def handle_missing_values(self, df):

        numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for col in numeric_cols:

            df[col] = df[col].fillna(
                df[col].median()
            )

        return df

    # =====================================
    # Outlier Treatment
    # =====================================

    def clip_outliers(self, df):

        numeric_cols = [
            col for col in df.columns
            if col not in ["patient_id", "gender"]
        ]

        for col in numeric_cols:

            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)

            df[col] = df[col].clip(
                q1,
                q99
            )

        return df


    # ==========================================
    # Preprocessing
    # ==========================================

    def fit_transform(self, df):

        df = df.copy()

        # Remove patient id

        if "patient_id" in df.columns:
            df.drop(
                columns=["patient_id"],inplace=True
            )
        
        df = self.handle_missing_values(df)

        df = self.clip_outliers(df)

        df = self.create_features(df)

        df["gender"] = (
            self.gender_encoder.fit_transform(
                df["gender"]
            )
        )

        # Scale Features

        X_scaled = self.scaler.fit_transform(
            df
        )

        #print(X_scaled.columns)

        print(X_scaled[:5])

        return X_scaled

        #return X_scaled

from app.config.settings import settings

if __name__ == "__main__":
    data_dir = settings.data_dir
    data_file = data_dir / "patient_population_health.csv"
    df = pd.read_csv(data_file)

    dp = DataPreprocessor()

    df_transformed = dp.fit_transform(df)
