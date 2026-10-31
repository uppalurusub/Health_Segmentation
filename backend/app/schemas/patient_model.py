"""
patient_model.py
"""

from dataclasses import dataclass


@dataclass
class Patient:

    patient_id: str

    age: int

    gender: str

    bmi: float

    diabetes: int

    hypertension: int

    copd: int

    mental_health: int

    visits_per_year: int

    er_visits: int

    admissions: int

    total_cost: float