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

    Parameters
    ----------
    features:
        Output of ``kinematics.extract_all_features``.
    thresholds:
        Output of ``normalization.get_dynamic_thresholds``.

    Returns
    -------
    dict[str, bool]
        ``{"K1": bool, "K2": bool, "K3": bool, "K4": bool, "K5": bool}``

    Rules (spec §3.5.4.3, Table 3.2)
    ----------------------------------
    K1 : RMS > K1_rms_threshold_cm
    K2 : velocity < K2_velocity_cms
    K3 : (P_avg < K3_pressure_avg) OR (P_decrement < K3_decrement_ratio)
    K4 : %ThinkTime > K4_pct_think_time
    K5 : PFHL_ms > K5_pfhl_ms
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

    Domain definitions (spec §3.5.5.1)
    ------------------------------------
    * S_Motor : K1 OR K2 OR K3
    * S_Cog   : K4 OR K5
    * S_AI    : structural ViT result (mocked as a heuristic here)

    Truth table
    -----------
    AI  Motor  Cog  →  Class
    N    N      N   →  C0
    N    Y      N   →  C1
    N    N      Y   →  C2
    N    Y      Y   →  C3
    Y    N      N   →  C4
    Y    N      Y   →  C5
    Y    Y      N   →  C6
    Y    Y      Y   →  C7
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

    Education Bias Warning (spec §3.5.5.1, note 4)
    -----------------------------------------------
    If ``education_years < 8`` **and** ``class_id ∈ {C4, C5, C6, C7}``,
    append ``"EDUCATION_BIAS_WARNING"`` to the warnings list.

    The system does **not** auto-downgrade the risk level (No Auto-
    Downgrading policy) to prevent false negatives in low-education
    patients who may have genuine dementia (Rossetti et al., 2011).

    Parameters
    ----------
    class_id:
        Output of ``apply_truth_table``.
    education_years:
        Years of formal education from the request.
    extra_warnings:
        Any warnings already accumulated upstream (e.g. from kinematics).

    Returns
    -------
    dict
        ``{class_id, risk_level, risk_color, warnings}``
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

    In production this function is replaced by ONNX Runtime inference.
    The mock triggers an AI abnormal signal when cognitive biomarkers
    suggest severe impairment, matching the heuristic described in
    the original mock spec.

    Returns
    -------
    (ai_abnormal, confidence_score)
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
    features:        dict,
    thresholds:      dict,
    education_years: int,
) -> dict:
    """
    Full clinical decision pipeline: K-series → domains → truth table → risk.

    Parameters
    ----------
    features:
        Output of ``kinematics.extract_all_features``.
    thresholds:
        Output of ``normalization.get_dynamic_thresholds``.
    education_years:
        From the ``AnalysisRequest``.

    Returns
    -------
    dict
        ``{class_id, risk_level, risk_color, warnings, domain, confidence}``
    """
    # 1. Evaluate individual K-rules
    k_results = evaluate_k_series(features, thresholds)

    # 2. Aggregate into domain signals
    motor_abnormal = k_results["K1"] or k_results["K2"] or k_results["K3"]
    cog_abnormal   = k_results["K4"] or k_results["K5"]
    ai_abnormal, confidence = _mock_ai_structural_result(features)

    # 3. Map to C0–C7
    class_id = apply_truth_table(ai_abnormal, motor_abnormal, cog_abnormal)

    # 4. Risk level + education warning
    upstream_warnings = features.get("flags", [])
    result = classify_risk(class_id, education_years, upstream_warnings)

    # 5. Attach domain-level detail for the response schema
    result["domain"] = {
        "motor_abnormal":     motor_abnormal,
        "cognitive_abnormal": cog_abnormal,
        "ai_abnormal":        ai_abnormal,
        "k1_triggered":       k_results["K1"],
        "k2_triggered":       k_results["K2"],
        "k3_triggered":       k_results["K3"],
        "k4_triggered":       k_results["K4"],
        "k5_triggered":       k_results["K5"],
    }
    result["confidence"] = confidence
    result["model_version"] = "mock-vit-v2.0"

    return result