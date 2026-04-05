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
    device_dpi      : Hardware screen resolution for physical normalisation.
    """

    strokes:         list[StrokePoint]
    image_b64:       str   = Field(..., description="Base-64 encoded PNG")
    patient_age:     int   = Field(..., ge=0, description="Age in years")
    education_years: int   = Field(..., ge=0, description="Years of formal education")
    device_dpi:      float = Field(..., gt=0, description="Device pixels per inch")


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class KinematicResult(BaseModel):
    """Raw, non-normalised biomarker values."""

    K1_rms_cm:             float | None = Field(None, description="Tremor: RMS deviation (cm)")
    K2_velocity_cms:       float | None = Field(None, description="Bradykinesia: Mean velocity (cm/s)")
    K3_pressure_avg:       float | None = Field(None, description="Micrographia: Mean pressure (0-1)")
    K3_pressure_decrement: float | None = Field(None, description="Micrographia: P_last / P_first ratio")
    K4_pct_think_time:     float | None = Field(None, description="Hesitation: % of time spent thinking")
    K5_pfhl_ms:            float | None = Field(None, description="Pre-first hand latency (ms)")
    flags:                 list[str]    = Field(default_factory=list, description="Data quality warnings")


class DomainResult(BaseModel):
    """Boolean flags for each clinical domain and individual K-rule."""

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
    velocity_profile: List of stroke durations in ms.
    """

    class_id:      str            = Field(..., description="Truth-table class, e.g. 'C2'")
    risk_level:    str            = Field(..., description="'normal' | 'mild' | 'high'")
    risk_color:    str            = Field(..., description="'green' | 'yellow' | 'red'")
    kinematic:     KinematicResult
    domain:        DomainResult
    warnings:      list[str]
    model_version: str
    velocity_profile: list[float] = Field(default_factory=list)