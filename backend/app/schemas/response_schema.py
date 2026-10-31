"""
response_schema.py
"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


# =====================================================
# Generic API Response
# =====================================================

class BaseResponse(BaseModel):

    status: str = "success"

    message: str

    data: Optional[Any] = None


# =====================================================
# Training Response
# =====================================================

class TrainingMetricsResponse(BaseModel):

    algorithm: str

    n_clusters: int

    silhouette_score: float

    davies_bouldin_score: float

    calinski_harabasz_score: float


# =====================================================
# Prediction Response
# =====================================================

class PatientSegmentationResponse(BaseModel):

    cluster: int

    segment_name: str

    confidence: Optional[float] = None


# =====================================================
# Model Evaluation Response
# =====================================================

class EvaluationResponse(BaseModel):

    silhouette_score: float

    davies_bouldin_score: float

    calinski_harabasz_score: float


# =====================================================
# Cluster Distribution Response
# =====================================================

class ClusterDistributionItem(BaseModel):

    cluster: int

    patient_count: int


# =====================================================
# Cluster Profile Response
# =====================================================

class ClusterProfileResponse(BaseModel):

    cluster: int

    age: float

    bmi: float

    diabetes: float

    hypertension: float

    copd: float

    mental_health: float

    visits_per_year: float

    er_visits: float

    admissions: float

    total_cost: float


# =====================================================
# Segment Summary Response
# =====================================================

class SegmentSummaryResponse(BaseModel):

    cluster: int

    segment_name: str

    patient_count: int

    average_age: float

    average_cost: float


# =====================================================
# Dashboard KPI Response
# =====================================================

class DashboardKPIResponse(BaseModel):

    total_patients: int

    total_segments: int

    average_age: float

    average_cost: float

    total_cost: float


# =====================================================
# Health Check Response
# =====================================================

class HealthCheckResponse(BaseModel):

    status: str

    module: str


# =====================================================
# Cluster Analytics Response
# =====================================================

class ClusterAnalyticsResponse(BaseModel):

    distribution: Dict

    profile: Dict

    evaluation: Dict


# =====================================================
# Model Comparison Response
# =====================================================

class ModelComparisonResponse(BaseModel):

    algorithm: str

    silhouette_score: float

    davies_bouldin_score: float

    calinski_harabasz_score: float