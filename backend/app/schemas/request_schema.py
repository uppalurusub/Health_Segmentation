"""
Request Schemas
"""

from pydantic import (
    BaseModel,
    Field
)

from typing import Optional


# =====================================================
# Patient Segmentation Prediction Request
# =====================================================

class PatientRequest(BaseModel):

    age: int = Field(
        ...,
        ge=0,
        le=120,
        description="Patient Age"
    )

    gender: str = Field(
        ...,
        description="M or F"
    )

    bmi: float = Field(
        ...,
        ge=10,
        le=80,
        description="Body Mass Index"
    )

    diabetes: int = Field(
        ...,
        ge=0,
        le=1
    )

    hypertension: int = Field(
        ...,
        ge=0,
        le=1
    )

    copd: int = Field(
        ...,
        ge=0,
        le=1
    )

    mental_health: int = Field(
        ...,
        ge=0,
        le=1
    )

    visits_per_year: int = Field(
        ...,
        ge=0
    )

    er_visits: int = Field(
        ...,
        ge=0
    )

    admissions: int = Field(
        ...,
        ge=0
    )

    total_cost: float = Field(
        ...,
        ge=0
    )

    class Config:

        json_schema_extra = {
            "example": {

                "age": 72,

                "gender": "F",

                "bmi": 34.5,

                "diabetes": 1,

                "hypertension": 1,

                "copd": 0,

                "mental_health": 0,

                "visits_per_year": 15,

                "er_visits": 4,

                "admissions": 2,

                "total_cost": 45000
            }
        }