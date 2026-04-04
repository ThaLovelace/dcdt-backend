"""
api/routes.py
-------------
FastAPI route definitions for the dCDT analysis pipeline.
All comments are in English.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import AnalysisRequest, AnalysisResponse
from core.inference import run_analysis

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_drawing(payload: AnalysisRequest):
    """
    Main endpoint for dCDT drawing analysis.
    
    This endpoint executes the full pipeline:
    1. Preprocessing (Smoothing & Signal Cleaning)
    2. Kinematic Feature Extraction (K1-K5 Biomarkers)
    3. Clinical Inference (Truth Table C0-C7 & Risk Mapping)
    4. Contextual Adjustments (Age Normalization & Education Warning)
    """
    try:
        # The run_analysis function is the top-level orchestrator that 
        # handles the entire workflow defined in the technical spec.
        result = run_analysis(
            strokes=payload.strokes,
            image_b64=payload.image_b64,
            age=payload.patient_age,
            education_years=payload.education_years,
            device_dpi=payload.device_dpi
        )

        return result

    except Exception as e:
        # Catch unexpected errors and return a 500 status code
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring."""
    return {"status": "healthy", "version": "2.0.0"}