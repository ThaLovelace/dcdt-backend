"""
schemas.py
----------
Pydantic request and response schemas for the dCDT analysis API.

All field definitions align with the Technical Specification v2.0.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class StrokePoint(BaseModel):
    """A single sampled point captured during the drawing test."""

    t:   float = Field(..., description="Timestamp in milliseconds (performance.now())")
    x:   float = Field(..., description="Canvas X coordinate in pixels")
    y:   float = Field(..., description="Canvas Y coordinate in pixels")
    p:   float = Field(..., description="Pen pressure, normalised 0.0–1.0")
    az:  float = Field(..., description="Azimuth angle in radians")
    alt: float = Field(..., description="Altitude angle in radians")
    id:  int   = Field(..., description="Stroke ID, monotonically increasing per session")


class AnalysisRequest(BaseModel):
    """
    Full payload sent by the frontend at the end of a drawing session.

    Fields
    ------
    strokes         : All captured stroke points for the session.
    image_b64       : Base-64 encoded PNG of the finished canvas (for ViT).
    patient_age     : Used to compute age-adjusted dynamic thresholds.
    education_years : Used to trigger the Education Bias Warning.
    device_dpi      : Screen DPI reported by the frontend; required to convert
                      pixel distances to centimetres for K1 and K2.
    """

    strokes:         list[StrokePoint] = Field(..., min_length=1)
    image_b64:       str               = Field(..., description="Base-64 PNG of canvas")
    patient_age:     int               = Field(..., ge=0, description="Patient age in years")
    education_years: int               = Field(..., ge=0, description="Years of formal education")
    device_dpi:      float             = Field(..., gt=0, description="Device screen DPI")


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class KinematicResult(BaseModel):
    """
    Computed kinematic biomarker values (K1–K5).

    A value of None indicates the metric could not be computed, either because
    the hardware does not support the required sensor, or because the stroke
    data did not meet the minimum eligibility criteria.
    """

    K1_rms_cm:             float | None = Field(None, description="RMS tremor deviation (cm)")
    K2_velocity_cms:       float | None = Field(None, description="Mean drawing velocity (cm/s)")
    K3_pressure_avg:       float | None = Field(None, description="Mean pen pressure (0–1)")
    K3_pressure_decrement: float | None = Field(None, description="P_last / P_first ratio")
    K4_pct_think_time:     float | None = Field(None, description="Percentage of think time (0–100)")
    K5_pfhl_ms:            float | None = Field(None, description="Pre-first-hand latency (ms)")
    flags:                 list[str]    = Field(default_factory=list,
                                                description="Processing flags, e.g. PRESSURE_NOT_SUPPORTED")


class DomainResult(BaseModel):
    """Binary abnormality flags for each clinical domain and individual K-rule."""

    motor_abnormal:     bool = False
    cognitive_abnormal: bool = False
    ai_abnormal:        bool = False
    k1_triggered:       bool = False
    k2_triggered:       bool = False
    k3_triggered:       bool = False
    k4_triggered:       bool = False
    k5_triggered:       bool = False


class AnalysisResponse(BaseModel):
    """
    Full analysis result returned to the frontend.

    Fields
    ------
    class_id    : Truth-table class C0–C7.
    risk_level  : Human-readable risk tier ('normal' | 'mild' | 'high').
    risk_color  : UI colour token ('green' | 'yellow' | 'red').
    kinematic   : Raw computed biomarker values.
    domain      : Boolean domain-level and rule-level flags.
    warnings    : List of contextual warnings, e.g. 'EDUCATION_BIAS_WARNING'.
    model_version: Version string of the inference engine used.
    """

    class_id:      str            = Field(..., description="Truth-table class, e.g. 'C2'")
    risk_level:    str            = Field(..., description="'normal' | 'mild' | 'high'")
    risk_color:    str            = Field(..., description="'green' | 'yellow' | 'red'")
    kinematic:     KinematicResult
    domain:        DomainResult
    warnings:      list[str]      = Field(default_factory=list,
                                          description="Contextual warnings for clinicians")
    model_version: str            = Field(default="mock-vit-v2.0")