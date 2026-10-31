"""
Clustering Analytics Module
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


class ClusteringAnalytics:

    # =====================================
    # Evaluate Clustering Model
    # =====================================

    def evaluate_model(
        self,
        X_scaled,
        labels
    ):

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

    # =====================================
    # Cluster Distribution
    # =====================================

    def cluster_distribution(
        self,
        labels
    ):

        distribution = (

            pd.Series(labels)

            .value_counts()

            .sort_index()

            .to_dict()
        )

        return distribution

    # =====================================
    # Cluster Profile
    # =====================================

    def cluster_profile(
        self,
        df,
        labels
    ):

        df_copy = df.copy()

        df_copy["cluster"] = labels

        profile = (

            df_copy

            .groupby("cluster")

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
        )

        return profile.to_dict(
            orient="index"
        )

    # =====================================
    # Dashboard KPIs
    # =====================================

    def dashboard_kpis(
        self,
        df,
        labels
    ):

        df_copy = df.copy()

        df_copy["cluster"] = labels

        return {

            "total_patients":
            len(df_copy),

            "total_segments":
            len(
                np.unique(labels)
            ),

            "average_age":
            round(
                df_copy["age"].mean(),
                2
            ),

            "average_cost":
            round(
                df_copy["total_cost"].mean(),
                2
            ),

            "total_cost":
            round(
                df_copy["total_cost"].sum(),
                2
            )
        }

    # =====================================
    # Segment Summary
    # =====================================

    def segment_summary(
        self,
        df,
        labels,
        segment_mapping
    ):

        df_copy = df.copy()

        df_copy["cluster"] = labels

        summary = []

        grouped = (

            df_copy

            .groupby("cluster")

            .agg({

                "patient_id":"count",

                "age":"mean",

                "total_cost":"mean"

            })

            .reset_index()
        )

        for _, row in grouped.iterrows():

            summary.append({

                "cluster":
                int(
                    row["cluster"]
                ),

                "segment_name":
                segment_mapping.get(
                    int(
                        row["cluster"]
                    ),
                    "Unknown"
                ),

                "patient_count":
                int(
                    row["patient_id"]
                ),

                "average_age":
                round(
                    row["age"],
                    2
                ),

                "average_cost":
                round(
                    row["total_cost"],
                    2
                )
            })

        return summary