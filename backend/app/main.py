"""
main.py

Population Health Patient Segmentation API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.patient_segmentation_router import (
    router as patient_segmentation_router
)

# =====================================================
# Create FastAPI App
# =====================================================

app = FastAPI(

    title="Population Health Segmentation API",

    description="""
    Healthcare Population Health Management

    Features:
    - Patient Segmentation
    - Population Health Analytics
    - Clustering Models
    - Cluster Profiling
    - Risk Stratification
    - Healthcare Cost Analysis
    """,

    version="1.0.0"
)

# =====================================================
# CORS
# =====================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)

# =====================================================
# Register Routers
# =====================================================

app.include_router(

    patient_segmentation_router,

    prefix="/api/v1"
)

# =====================================================
# Root Endpoint
# =====================================================

@app.get("/")
def home():

    return {

        "application":
        "Population Health Segmentation API",

        "version":
        "1.0.0",

        "status":
        "Running"
    }


# =====================================================
# Health Check
# =====================================================

@app.get("/health")
def health():

    return {

        "status": "UP",

        "application":
        "Population Health Segmentation API"
    }