"""
Patient Segmentation Implementation
"""

import os
import joblib
import pandas as pd
import numpy as np
from hdbscan import approximate_predict
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import os
from app.config.settings import settings
import time
from pathlib import Path

data_dir = settings.data_dir
data_file = data_dir / "patient_population_health.csv"

#f"saved_models/"
artifact_dir = Path(settings.artifact_dir)

class PatientSegmentationImpl:

    def __init__(self):

        self.data_path = (
            data_file
        )

        #os.makedirs("best_saved_models",exist_ok=True)

        self.model_path = (
            f"{artifact_dir}/best_hs_model.pkl" 
        )

        self.scaler_path = (
            f"{artifact_dir}/scaler.pkl"
        )

        self.encoder_path = (
            f"{artifact_dir}/encoder.pkl"
        )

        self.feature_columns_path = (
            f"{artifact_dir}/feature_columns.pkl"
        )

        

    
    # ==================================================
    # Load Dataset
    # ==================================================

    def load_data(self):

        return pd.read_csv(
            self.data_path
        )

    # ==================================================
    # Preprocess Data
    # ==================================================

    def preprocess_data(
        self,
        df
    ):

        df_model = df.copy()

        encoder = LabelEncoder()
        

        df_model["gender"] = (
            encoder.fit_transform(
                df_model["gender"]
            )
        )

        if "patient_id" in df_model.columns:

            df_model.drop(
                columns=["patient_id"],
                inplace=True
            )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            df_model
        )

        
        joblib.dump(
            encoder,
            self.encoder_path
        )

        joblib.dump(
            scaler,
            self.scaler_path
        )

        return X_scaled

    def _load_artifacts(self):

        required = {
            "model": self.model_path,
            "scaler": self.scaler_path,
            "encoder": self.encoder_path,
            "feature_columns": self.feature_columns_path
        }

        for name, path in required.items():

            path = Path(path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {name} artifact: {path}"
                )

        model = joblib.load(self.model_path)
        scaler = joblib.load(self.scaler_path)
        encoder = joblib.load(self.encoder_path)
        feature_columns = joblib.load(
            self.feature_columns_path
        )

        return model, scaler, encoder, feature_columns

    @staticmethod
    def _encode_and_scale(
        df,
        scaler,
        encoder,
        feature_columns
    ):
        """Encode and scale patient data using training artifacts."""

        df_model = df.drop(
            columns=["patient_id"],
            errors="ignore"
        ).copy()

        if "gender" not in df_model.columns:
            raise ValueError("Required feature 'gender' is missing.")

        # Validate gender categories
        gender_values = df_model["gender"].astype(str)

        known_values = set(encoder.classes_.astype(str))

        unseen = sorted(set(gender_values) - known_values)

        if unseen:
            raise ValueError(
                f"Unknown gender category values: {unseen}"
            )

        # Apply fitted encoder
        df_model["gender"] = encoder.transform(gender_values)

        # Validate feature schema
        missing_columns = [
            col for col in feature_columns
            if col not in df_model.columns
        ]

        extra_columns = [
            col for col in df_model.columns
            if col not in feature_columns
        ]

        if missing_columns or extra_columns:
            raise ValueError(
                f"Feature schema mismatch. "
                f"Missing={missing_columns}; "
                f"Unexpected={extra_columns}"
            )

        # Preserve training feature order
        df_model = df_model[feature_columns]

        # Apply fitted scaler
        return scaler.transform(df_model)

    # ==================================================
    # Predict Patient Segment with confidence
    # ==================================================

    def predict_cluster_with_confidence(self, model, X_scaled):

        model_name = type(model).__name__

        confidence = None
        membership_scores = None

        # --------------------------------
        # HDBSCAN
        # --------------------------------
        if model_name == "HDBSCAN":

            labels, strengths = approximate_predict(
                model,
                X_scaled
            )

            confidence = float(strengths[0])

        # --------------------------------
        # GMM
        # --------------------------------
        elif model_name == "GaussianMixture":

            membership_scores = model.predict_proba(
                X_scaled
            )

            labels = model.predict(X_scaled)

            confidence = float(
                np.max(membership_scores[0])
            )

        # --------------------------------
        # KMeans
        # --------------------------------
        elif model_name == "KMeans":

            labels = model.predict(X_scaled)

            distances = model.transform(X_scaled)

            assigned_distance = distances[
                np.arange(len(labels)), labels
            ]

            confidence = float(
                1 / (1 + assigned_distance[0])
            )

        # --------------------------------
        # Birch
        # --------------------------------
        elif model_name == "Birch":

            labels = model.predict(X_scaled)

            distances = pairwise_distances(
                X_scaled,
                model.subcluster_centers_
            )

            confidence = float(
                1 / (1 + np.min(distances[0]))
            )

        else:

            raise ValueError(
                f"Confidence calculation not implemented for {model_name}"
            )

        return {
            "cluster": int(labels[0]),
            "confidence": confidence,
            "membership_scores": (
                membership_scores[0].tolist()
                if membership_scores is not None
                else None
            ),
            "model_name": model_name
        }
    
    # ==================================================
    # Predict Patient Segment
    # ==================================================
    def predict(self, patient_data):

        start_time = time.perf_counter()

        model, scaler, encoder, feature_columns = (
            self._load_artifacts()
        )

        df = pd.DataFrame([patient_data])

        X_scaled = self._encode_and_scale(
            df,
            scaler,
            encoder,
            feature_columns
        )

        prediction = self.predict_cluster_with_confidence(
            model,
            X_scaled
        )

        cluster = prediction["cluster"]

        execution_time = time.perf_counter() - start_time

        

        return {
            "cluster": cluster,

            
            "confidence": prediction["confidence"],

            "membership_scores": prediction["membership_scores"],

            "model_name": prediction["model_name"],

            "execution_time_seconds": round(
                execution_time, 6
            )
        }

    # ==================================================
    # Evaluation
    # ==================================================

    def get_evaluation(self):

        df = self.load_data()

        model = joblib.load(
            self.model_path
        )

        encoder = joblib.load(
            self.encoder_path
        )

        scaler = joblib.load(
            self.scaler_path
        )

        df_model = df.copy()

        df_model["gender"] = (
            encoder.transform(
                df_model["gender"]
            )
        )

        df_model.drop(
            columns=["patient_id"],
            inplace=True
        )

        X_scaled = scaler.transform(
            df_model
        )

        #labels = model.labels_

        """ labels = model.predict(
            X_scaled
        ) """
        if model.__class__.__name__ == "HDBSCAN":
            labels, strengths = approximate_predict(model, X_scaled)
        else:
            labels = model.predict(X_scaled)

        return {

            "silhouette_score":
            round(
                silhouette_score(
                    X_scaled,
                    labels
                ),
                4
            ),

            "davies_bouldin_score":
            round(
                davies_bouldin_score(
                    X_scaled,
                    labels
                ),
                4
            ),

            "calinski_harabasz_score":
            round(
                calinski_harabasz_score(
                    X_scaled,
                    labels
                ),
                4
            )
        }

    # ==================================================
    # Cluster Distribution
    # ==================================================

    def cluster_distribution(self):

        df = self.load_data()

        model = joblib.load(
            self.model_path
        )

        encoder = joblib.load(
            self.encoder_path
        )

        scaler = joblib.load(
            self.scaler_path
        )

        df_model = df.copy()

        df_model["gender"] = (
            encoder.transform(
                df_model["gender"]
            )
        )

        df_model.drop(
            columns=["patient_id"],
            inplace=True
        )

        X_scaled = scaler.transform(
            df_model
        )

        #labels = model.labels_
        """ labels = model.predict(
            X_scaled
        ) """
        if model.__class__.__name__ == "HDBSCAN":
            labels, strengths = approximate_predict(model, X_scaled)
        else:
            labels = model.predict(X_scaled)

        distribution = (

            pd.Series(labels)

            .value_counts()

            .sort_index()

            .to_dict()
        )

        return distribution

    # ==================================================
    # Cluster Profile
    # ==================================================

    def cluster_profile(self):

        df = self.load_data()

        model = joblib.load(
            self.model_path
        )

        encoder = joblib.load(
            self.encoder_path
        )

        scaler = joblib.load(
            self.scaler_path
        )

        df_model = df.copy()

        df_model["gender"] = (
            encoder.transform(
                df_model["gender"]
            )
        )

        df_model.drop(
            columns=["patient_id"],
            inplace=True
        )

        X_scaled = scaler.transform(
            df_model
        )

        #labels = model.labels_

        """ labels = model.predict(
            X_scaled
        ) """
        if model.__class__.__name__ == "HDBSCAN":
            labels, strengths = approximate_predict(model, X_scaled)
        else:
            labels = model.predict(X_scaled)

        df["cluster"] = labels

        profile = (

            df.groupby("cluster")

            .agg({

                "age":"mean",

                "bmi":"mean",

                "diabetes":"mean",

                "hypertension":"mean",

                "copd":"mean",

                "mental_health":"mean",

                "visits_per_year":"mean",

                "er_visits":"mean",

                "admissions":"mean",

                "total_cost":"mean"

            })

            .round(2)

            .to_dict(
                orient="index"
            )
        )

        return profile