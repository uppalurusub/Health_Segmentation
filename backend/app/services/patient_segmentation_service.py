"""
Patient Segmentation Service
"""

from app.implementations.patient_segmentation_impl import (
    PatientSegmentationImpl
)

import time
import mlflow
from datetime import datetime, timezone
# ====================================================
# MLFLOW
# ====================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Health_Segmentation"
)

class PatientSegmentationService:

    def __init__(self):

        self.impl = PatientSegmentationImpl()

        

    
    # ==========================================
    # Predict Segment
    # ==========================================

    def predict_segment(
        self,
        patient_data: dict
    ):

        started = time.perf_counter()

        with mlflow.start_run(
                run_name=
                f"predict_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):

            prediction = self.impl.predict(
                patient_data
            )
            print("prediction: ",prediction)

            
            elapsed = time.perf_counter() - started

            cluster = int(
                prediction["cluster"]
            )

            print("cluster: ",cluster)
            

            prediction_response = {

                "cluster": cluster,
                "confidence":
                prediction.get(
                    "confidence",
                    None
                )
            }

            elapsed = time.perf_counter() - started

            mlflow.log_param(
                "input_text",
                patient_data
            )

            mlflow.log_param(
                "prediction",
                prediction
            )

            mlflow.log_metric(
                "execution_time",
                elapsed
            )

            mlflow.set_tag(
                "source",
                "FastAPI"
            )

            mlflow.set_tag(
                "endpoint",
                "/predict_segment"
            )


        

        return prediction_response

    # ==========================================
    # Evaluation
    # ==========================================

    def get_evaluation(self):

        

        prediction_response = self.impl.get_evaluation()
        
        return prediction_response
        #return self.impl.get_evaluation()

    # ==========================================
    # Cluster Distribution
    # ==========================================

    def cluster_distribution(self):

        return self.impl.cluster_distribution()

    # ==========================================
    # Cluster Profile
    # ==========================================

    def cluster_profile(self):

        return self.impl.cluster_profile()

    # ==========================================
    # Segment Summary
    # ==========================================

    def segment_summary(self):

        distribution = (
            self.impl.cluster_distribution()
        )

        summary = []

        for cluster, count in (
            distribution.items()
        ):

            summary.append({

                "cluster": cluster,

                

                "patient_count":
                count
            })

        return summary