"""
inference.py
------------
Clinical decision logic for the dCDT pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
1. ``evaluate_k_series``  — compare computed features against thresholds.
2. ``apply_truth_table``  — map the three binary domain signals to C0–C7.
3. ``classify_risk``      — translate C0–C7 to 3-tier colour-coded risk level
                            and append the Education Bias Warning when required.
4. ``run_analysis``       — top-level orchestrator called by the FastAPI route.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass   # avoid circular imports at runtime

# ---------------------------------------------------------------------------
# Import all necessary modules for the orchestrator
# ---------------------------------------------------------------------------
from core.preprocessing import process_strokes
from core.kinematics import extract_all_features, detect_pressure_support
from core.normalization import get_dynamic_thresholds

# ---------------------------------------------------------------------------
# Risk level mapping (spec §3.5.5.2, Table 3.4)
# ---------------------------------------------------------------------------

_RISK_MAP: dict[str, tuple[str, str]] = {
    "C0": ("normal", "green"),
    "C1": ("mild",   "yellow"),
    "C2": ("mild",   "yellow"),
    "C3": ("mild",   "yellow"),
    "C4": ("mild",   "yellow"),
    "C5": ("high",   "red"),
    "C6": ("high",   "red"),
    "C7": ("high",   "red"),
}

# Classes that trigger the Education Bias Warning (spec §3.5.5.1, note 4)
_EDUCATION_WARNING_CLASSES: frozenset[str] = frozenset({"C4", "C5", "C6", "C7"})
EDUCATION_BIAS_WARNING = "EDUCATION_BIAS_WARNING"


# ---------------------------------------------------------------------------
# Step 1: Evaluate K-Series against thresholds
# ---------------------------------------------------------------------------

def evaluate_k_series(
    features:   dict,
    thresholds: dict,
) -> dict[str, bool]:
    """
    Compare each kinematic feature value against its clinical threshold.

    A feature that is ``None`` (hardware not supported, or insufficient
    data) is treated as **not triggered** — it does not contribute an
    abnormal signal.
    """
    def _val(key: str) -> "float | None":
        return features.get(key)

    # K1
    k1_val = _val("K1_rms_cm")
    k1 = (k1_val is not None) and (k1_val > thresholds["K1_rms_threshold_cm"])

    # K2
    k2_val = _val("K2_velocity_cms")
    k2 = (k2_val is not None) and (k2_val < thresholds["K2_velocity_cms"])

    # K3 — two conditions joined by OR
    k3_avg_val = _val("K3_pressure_avg")
    k3_dec_val = _val("K3_pressure_decrement")
    k3_avg_flag = (k3_avg_val is not None) and (k3_avg_val < thresholds["K3_pressure_avg"])
    k3_dec_flag = (
        (k3_dec_val is not None)
        and (k3_dec_val < thresholds["K3_decrement_ratio"])
    )
    k3 = k3_avg_flag or k3_dec_flag

    # K4
    k4_val = _val("K4_pct_think_time")
    k4 = (k4_val is not None) and (k4_val > thresholds["K4_pct_think_time"])

    # K5
    k5_val = _val("K5_pfhl_ms")
    k5 = (k5_val is not None) and (k5_val > thresholds["K5_pfhl_ms"])

    return {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5}


# ---------------------------------------------------------------------------
# Step 2: Truth Table (C0–C7)
# ---------------------------------------------------------------------------

def apply_truth_table(
    ai_abnormal:     bool,
    motor_abnormal:  bool,
    cog_abnormal:    bool,
) -> str:
    """
    Map the three binary domain signals to a truth-table class C0–C7.
    """
    index = (
        (4 if ai_abnormal    else 0)
        + (2 if motor_abnormal else 0)
        + (1 if cog_abnormal   else 0)
    )
    # Explicit mapping keeps the logic transparent for clinical audit
    _table = {
        0: "C0",   # N N N
        2: "C1",   # N Y N
        1: "C2",   # N N Y
        3: "C3",   # N Y Y
        4: "C4",   # Y N N
        5: "C5",   # Y N Y
        6: "C6",   # Y Y N
        7: "C7",   # Y Y Y
    }
    return _table[index]


# ---------------------------------------------------------------------------
# Step 3: Risk classification + Education Bias Warning
# ---------------------------------------------------------------------------

def classify_risk(
    class_id:        str,
    education_years: int,
    extra_warnings:  list[str] | None = None,
) -> dict:
    """
    Convert the truth-table class to a 3-tier risk level and collect warnings.
    """
    risk_level, risk_color = _RISK_MAP.get(class_id, ("unknown", "grey"))

    warnings: list[str] = list(extra_warnings or [])

    if education_years < 8 and class_id in _EDUCATION_WARNING_CLASSES:
        warnings.append(EDUCATION_BIAS_WARNING)

    return {
        "class_id":   class_id,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "warnings":   warnings,
    }


# ---------------------------------------------------------------------------
# Step 4: Mock ViT — structural domain signal
# ---------------------------------------------------------------------------

def _mock_ai_structural_result(features: dict) -> tuple[bool, float]:
    """
    Heuristic mock for the ViT structural analysis.
    """
    k4 = features.get("K4_pct_think_time") or 0.0
    k5 = features.get("K5_pfhl_ms")        or 0.0

    ai_abnormal   = (k4 > 60.0) or (k5 > 12_000.0)
    confidence    = 0.88 if ai_abnormal else 0.95
    return ai_abnormal, confidence


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_analysis(
    strokes: list,
    image_b64: str,
    age: int,
    education_years: int,
    device_dpi: float
) -> dict:
    """
    Full clinical decision pipeline orchestrator.
    Expects raw payload data, extracts features, and runs clinical logic.
    """
    # 1. Check if the device hardware provides meaningful pressure data
    pressure_supported = detect_pressure_support(strokes)

    # 2. Pre-processing: Apply Savitzky-Golay smoothing and eligibility filtering
    processed_summary = process_strokes(strokes)

    # EXTRACT TIMELINE DATA HERE
    timeline_data = []
    if "processed_strokes" in processed_summary:
        timeline_data = [
            stroke["duration_ms"] 
            for stroke in processed_summary["processed_strokes"] 
            if stroke["duration_ms"] > 0
        ]

    # 3. Kinematic Feature Extraction (K1-K5)
    features = extract_all_features(
        raw_strokes=strokes,
        processed_summary=processed_summary,
        device_dpi=device_dpi,
        pressure_supported=pressure_supported
    )
    
    # 4. Get Age-Adjusted Thresholds
    thresholds = get_dynamic_thresholds(age)

    # 5. Evaluate individual K-rules
    k_results = evaluate_k_series(features, thresholds)

    # 6. Aggregate into domain signals
    motor_abnormal = k_results["K1"] or k_results["K2"] or k_results["K3"]
    cog_abnormal   = k_results["K4"] or k_results["K5"]
    ai_abnormal, confidence = _mock_ai_structural_result(features)

    # 7. Map to Truth Table C0–C7
    class_id = apply_truth_table(ai_abnormal, motor_abnormal, cog_abnormal)

    # 8. Risk level + education warning
    upstream_warnings = features.get("flags", [])
    result = classify_risk(class_id, education_years, upstream_warnings)

    # 9. Format the exact response expected by schemas.AnalysisResponse
    return {
        "class_id": result["class_id"],
        "risk_level": result["risk_level"],
        "risk_color": result["risk_color"],
        "kinematic": {
            "K1_rms_cm": features.get("K1_rms_cm"),
            "K2_velocity_cms": features.get("K2_velocity_cms"),
            "K3_pressure_avg": features.get("K3_pressure_avg"),
            "K3_pressure_decrement": features.get("K3_pressure_decrement"),
            "K4_pct_think_time": features.get("K4_pct_think_time"),
            "K5_pfhl_ms": features.get("K5_pfhl_ms"),
            "flags": features.get("flags", [])
        },
        "domain": {
            "motor_abnormal":     motor_abnormal,
            "cognitive_abnormal": cog_abnormal,
            "ai_abnormal":        ai_abnormal,
            "k1_triggered":       k_results["K1"],
            "k2_triggered":       k_results["K2"],
            "k3_triggered":       k_results["K3"],
            "k4_triggered":       k_results["K4"],
            "k5_triggered":       k_results["K5"],
        },
        "warnings": result["warnings"],
        "model_version": "mock-vit-v2.0",
        "velocity_profile": timeline_data
    }