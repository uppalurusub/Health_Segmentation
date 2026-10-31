"""
Patient Segmentation Router
"""

from fastapi import (
    APIRouter,
    HTTPException,
    status
)

from app.schemas.request_schema import (
    PatientRequest
)

from app.services.patient_segmentation_service import (
    PatientSegmentationService
)

router = APIRouter(
    prefix="/patient-segmentation",
    tags=["Patient Segmentation"]
)

service = PatientSegmentationService()





# =====================================================
# Predict Patient Segment
# =====================================================

@router.post(
    "/predict",
    status_code=status.HTTP_200_OK
)
def predict_patient_segment(
    request: PatientRequest
):

    try:

        result = service.predict_segment(
            request.model_dump()
        )

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# Model Evaluation
# =====================================================

@router.get(
    "/evaluation",
    status_code=status.HTTP_200_OK
)
def model_evaluation():

    try:

        return {
            "status": "success",
            "data": service.get_evaluation()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# Cluster Distribution
# =====================================================

@router.get(
    "/cluster-distribution",
    status_code=status.HTTP_200_OK
)
def cluster_distribution():

    try:

        return {
            "status": "success",
            "data": service.cluster_distribution()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# Cluster Profile
# =====================================================

@router.get(
    "/cluster-profile",
    status_code=status.HTTP_200_OK
)
def cluster_profile():

    try:

        return {
            "status": "success",
            "data": service.cluster_profile()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# Segment Summary
# =====================================================

@router.get(
    "/segment-summary",
    status_code=status.HTTP_200_OK
)
def segment_summary():

    try:

        return {
            "status": "success",
            "data": service.segment_summary()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# Health Check
# =====================================================

@router.get(
    "/health",
    status_code=status.HTTP_200_OK
)
def health():

    return {
        "status": "UP",
        "module": "Patient Segmentation"
    }