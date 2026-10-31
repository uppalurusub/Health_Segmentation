#Best Model Selection Strategy
"""
Best_Score = (
    silhouette_score
    +
    normalized_calinski_score
    -
    normalized_davies_score
)

Priority:

1. Highest Silhouette Score
2. Lowest Davies Bouldin Score
3. Highest Calinski Harabasz Score
4. Lowest Noise Percentage

"""
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from hdbscan import approximate_predict
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    SpectralClustering
)
#from app.config.settings import settings 
#logging = settings.get_logger()

import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

from sklearn.mixture import GaussianMixture

import hdbscan
import skfuzzy as fuzz

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from app.config.settings import settings
data_dir = settings.data_dir
data_file = data_dir / "patient_population_health.csv"

#f"saved_models/"
artifact_dir = settings.artifact_dir
results_dir = settings.results_dir
# ====================================================
# MLFLOW
# ====================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Health_Segmentation"
)

# ====================================================
# LOAD DATA
# ====================================================

df = pd.read_csv(
    data_file
)

df_model = df.copy()

encoder = LabelEncoder()

df_model["gender"] = encoder.fit_transform(
    df_model["gender"]
)

df_model.drop(
    columns=["patient_id"],
    inplace=True
)

# Preserve exact feature names and order
feature_columns = list(df_model.columns)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df_model
)

# ====================================================
# MODELS
# ====================================================

models = {

    "KMeans":
        KMeans(
            n_clusters=8,
            random_state=42,
            n_init=20
        ),

    "Agglomerative":
        AgglomerativeClustering(
            n_clusters=8
        ),

    "Birch":
        Birch(
            n_clusters=8
        ),

    "DBSCAN":
        DBSCAN(
            eps=1.2,
            min_samples=25
        ),

    "OPTICS":
        OPTICS(
            min_samples=50
        ),

    "Spectral":
        SpectralClustering(
            n_clusters=8,
            random_state=42
        ),

    "GMM":
        GaussianMixture(
            n_components=8,
            random_state=42
        ),
    "HDBSCAN":
        hdbscan.HDBSCAN(
            min_cluster_size=100,
            min_samples=25,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=-1
        )

    
    

}

# ====================================================
# RESULTS
# ====================================================

results = []

best_model = None
best_score = -999999

# ====================================================
# TRAIN
# ====================================================

for model_name, model in models.items():

    print(
        f"\nTraining {model_name}"
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        try:

            # ==================================
            # FIT
            # ==================================

            if model_name == "GMM":

                model.fit(X_scaled)

                labels = model.predict(
                    X_scaled
                )

            

            elif model_name == "HDBSCAN":

                #model.fit(X_scaled)
                labels, strengths = approximate_predict(
                    model,
                    X_scaled
                )

                print("\nHDBSCAN PREDICTION DIAGNOSTICS")
                print("--------------------------------")

                print("Predicted cluster:", labels[0])

                print("Prediction strength:", strengths[0])

                print("Number of training clusters:",
                    len(set(model.labels_) - {-1}))

                print("Training noise percentage:",
                    round(
                        np.mean(model.labels_ == -1) * 100,
                        2
                    ))

                print("Training cluster distribution:")

                print(
                    pd.Series(model.labels_)
                    .value_counts()
                    .sort_index()
                )

                # Explicitly generate prediction data
                model.generate_prediction_data()

                labels = model.labels_

                if model.prediction_data_ is None:
                    raise RuntimeError(
                        "HDBSCAN prediction data generation failed."
                    )

            

            else:

                labels = model.fit_predict(
                    X_scaled
                )

            # ==================================
            # REMOVE NOISE
            # ==================================
            unique_labels, counts = np.unique(
                labels,
                return_counts=True
            )

            noise_count = np.sum(labels == -1)

            print("\nCLUSTER DIAGNOSTICS")
            print("--------------------------")
            print("Total records:", len(labels))
            print("Unique labels:", unique_labels)
            print("Cluster distribution:", dict(zip(unique_labels, counts)))
            print("Number of clusters:", len(set(labels) - {-1}))
            print("Noise records:", noise_count)
            print(
                "Noise percentage:",
                round(noise_count / len(labels) * 100, 2)
            )


            mask = labels != -1

            X_eval = X_scaled[mask]
            y_eval = labels[mask]

            if len(
                np.unique(y_eval)
            ) < 2:

                print(
                    f"{model} : Skipped"
                )

                continue

            # ==================================
            # METRICS
            # ==================================

            silhouette = silhouette_score(
                X_eval,
                y_eval
            )

            davies = davies_bouldin_score(
                X_eval,
                y_eval
            )

            calinski = calinski_harabasz_score(
                X_eval,
                y_eval
            )

            final_score = (
                silhouette
                -
                davies
                +
                np.log1p(calinski)
            )

            mlflow.log_metric(
                "silhouette_score",
                silhouette
            )

            mlflow.log_metric(
                "davies_bouldin_score",
                davies
            )

            mlflow.log_metric(
                "calinski_harabasz_score",
                calinski
            )

            mlflow.log_metric(
                "final_score",
                final_score
            )

            print("final_score: ", final_score)

            # ==================================
            # SAVE MODEL
            # ==================================
            
            model_path = (
                f"{artifact_dir}/"
                f"{model_name}.pkl"
            )

            joblib.dump(
                model,
                model_path
            )

            mlflow.log_artifact(
                model_path
            )

            

            if final_score > best_score:

                best_score = final_score
                best_model = model
                best_model_name = model_name
                
                print(f"\nBest Model: {best_model_name}")
                print(f"Best Score: {best_score:.4f}")
                print("Selected estimator type:", type(best_model))

                
            results.append({

                "Model":
                    model_name,

                "Silhouette":
                    silhouette,

                "Davies":
                    davies,

                "Calinski":
                    calinski,

                "FinalScore":
                    final_score
            })

        except Exception as e:

            print(
                model_name,
                str(e)
            )

# ====================================================
# RESULTS
# ====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "FinalScore",
    ascending=False
)

print(
    "\nMODEL COMPARISON"
)

print(
    results_df
)

print(
    "\nBEST MODEL:",
    best_model
)

print(
    "BEST SCORE:",
    best_score
)

results_df.to_csv(
    f"{results_dir}/clustering_comparison.csv",
    index=False
)


#Best Model Registration
with mlflow.start_run(
    run_name="Best_Clustering_Model"
) as run:

    mlflow.log_param(
        "best_model_name",
        best_model_name
    )

    mlflow.log_metric(
        "best_score",
        best_score
    )

    mlflow.sklearn.log_model(

        sk_model=best_model,

        artifact_path="HealthSegmentModel",

        registered_model_name= "HealthSegmentModel",
        skops_trusted_types=['hdbscan.dist_metrics.EuclideanDistance', 
            'hdbscan.hdbscan_.HDBSCAN', 
            'hdbscan.prediction.PredictionData', 
            'joblib.memory.Memory', 
            'sklearn.metrics._dist_metrics.EuclideanDistance64', 
            'sklearn.neighbors._kd_tree.KDTree'
        ]
    )

    model_uri = (
        f"runs:/{run.info.run_id}/model"
    )

    print(
        "Best Model URI:",
        model_uri
    )

print("Saving Best Model: ")
joblib.dump(
    best_model,
    f"{artifact_dir}/best_hs_model.pkl"
)

# Verify the saved artifact
saved_model = joblib.load(f"{artifact_dir}/best_hs_model.pkl")

print("Loaded model type:", type(saved_model))

print(
    "Loaded model predict():",
    callable(getattr(saved_model, "predict", None))
)

""" if not callable(getattr(saved_model, "predict", None)):
    raise TypeError("Saved model does not support predict().") """

print("Best model saved and verified successfully.")

# ====================================================
# SAVE PREPROCESSOR
# ====================================================

joblib.dump(
    feature_columns,
    f"{artifact_dir}/feature_columns.pkl"
)

joblib.dump(
    scaler,
    f"{artifact_dir}/scaler.pkl"
)

joblib.dump(
    encoder,
    f"{artifact_dir}/encoder.pkl"
)