"""
core/inference.py
-----------------
Clinical decision logic for the dCDT pipeline.

Responsibilities
~~~~~~~~~~~~~~~~
1. ``evaluate_k_series``  - Compare computed features against thresholds.
2. ``apply_truth_table``  - Map the three binary domain signals to C0-C7.
3. ``classify_risk``      - Translate C0-C7 to a 3-tier color-coded risk level.
4. ``run_real_vit``       - Evaluates the structural domain using the trained ViT model.
5. ``run_analysis``       - Top-level orchestrator called by the FastAPI route.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    pass   # avoid circular imports at runtime

from core.preprocessing import process_strokes
from core.kinematics import extract_all_features, detect_pressure_support
from core.normalization import get_dynamic_thresholds

# Custom internal modules for ViT integration
from core.vision_bridge import prepare_image_for_vit
from core.xai_engine import load_vit_model, generate_xai_b64

# ===========================================================================
# Global Model Initialization
# ===========================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Direct path to local model file
VIT_CHECKPOINT_PATH = os.path.join("models", "best_vit_v3_final.pth")

try:
    print(f"⏳ [Local Mode] Loading model from: {VIT_CHECKPOINT_PATH}")
    
    if not os.path.exists(VIT_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model file not found at {VIT_CHECKPOINT_PATH}")

    # Load model into memory
    VIT_MODEL = load_vit_model(VIT_CHECKPOINT_PATH, device=DEVICE)
    print("✅ [Local Mode] ViT Model loaded successfully! 🚀")

except Exception as e:
    print(f"❌ Warning: Could not load local ViT model. Error: {e}")
    VIT_MODEL = None

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

_EDUCATION_WARNING_CLASSES: frozenset[str] = frozenset({"C4", "C5", "C6", "C7"})
EDUCATION_BIAS_WARNING = "EDUCATION_BIAS_WARNING"


# ---------------------------------------------------------------------------
# Step 1: Evaluate K-Series against thresholds
# ---------------------------------------------------------------------------
def evaluate_k_series(features: dict, thresholds: dict) -> dict[str, bool]:
    """
    Compare each kinematic feature value against its clinical threshold.
    """
    K1_RMS_THRESHOLD_CM: float = 0.03

    def _val(key: str) -> "float | None":
        return features.get(key)

    # K1
    k1_val = _val("K1_rms_cm")
    k1_threshold = min(thresholds.get("K1_rms_threshold_cm", K1_RMS_THRESHOLD_CM), K1_RMS_THRESHOLD_CM)
    k1 = (k1_val is not None) and (k1_val > k1_threshold)

    # K2
    k2_val = _val("K2_velocity_cms")
    k2 = (k2_val is not None) and (k2_val < thresholds["K2_velocity_cms"])

    # K3 (Joined by OR condition)
    k3_avg_val = _val("K3_pressure_avg")
    k3_dec_val = _val("K3_pressure_decrement")
    k3_avg_flag = (k3_avg_val is not None) and (k3_avg_val < thresholds["K3_pressure_avg"])
    k3_dec_flag = ((k3_dec_val is not None) and (k3_dec_val < thresholds["K3_decrement_ratio"]))
    k3 = k3_avg_flag or k3_dec_flag

    # K4
    k4_val = _val("K4_pct_think_time")
    k4 = (k4_val is not None) and (k4_val > thresholds["K4_pct_think_time"])

    # K5
    k5_val = _val("K5_pfhl_ms")
    k5 = (k5_val is not None) and (k5_val > thresholds["K5_pfhl_ms"])

    return {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5}


# ---------------------------------------------------------------------------
# Step 2: Truth Table (C0-C7)
# ---------------------------------------------------------------------------
def apply_truth_table(ai_abnormal: bool, motor_abnormal: bool, cog_abnormal: bool) -> str:
    """
    Map the three binary domain signals to a truth-table class C0-C7.
    """
    index = ((4 if ai_abnormal else 0) + (2 if motor_abnormal else 0) + (1 if cog_abnormal else 0))
    _table = {0: "C0", 2: "C1", 1: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7"}
    return _table[index]

# ---------------------------------------------------------------------------
# Step 3: Risk classification + Education Bias Warning
# ---------------------------------------------------------------------------
def classify_risk(class_id: str, education_years: int, extra_warnings: list[str] | None = None) -> dict:
    """
    Convert the truth-table class to a 3-tier risk level and collect warnings.
    """
    risk_level, risk_color = _RISK_MAP.get(class_id, ("unknown", "grey"))
    warnings: list[str] = list(extra_warnings or [])
    
    if education_years < 8 and class_id in _EDUCATION_WARNING_CLASSES:
        warnings.append(EDUCATION_BIAS_WARNING)
        
    return {
        "class_id": class_id, 
        "risk_level": risk_level, 
        "risk_color": risk_color, 
        "warnings": warnings
    }

# ---------------------------------------------------------------------------
# Step 4: REAL ViT INFERENCE
# ---------------------------------------------------------------------------
def run_real_vit_inference(image_b64: str) -> tuple[bool, float, str, str]:
    """
    Executes the trained ViT model and generates the EigenCAM heatmap.
    Returns: (ai_abnormal_flag, confidence_score, processed_b64, heatmap_b64)
    """
    if not VIT_MODEL:
        return False, 0.0, "", ""

    # 1. Digital Bridge Preprocessing (Centering and Padding logic happens here)
    # Updated to receive processed_b64 for side-by-side display
    input_tensor, img_pil, processed_b64 = prepare_image_for_vit(image_b64)
    input_tensor = input_tensor.to(DEVICE)

    # 2. Model Prediction
    with torch.no_grad():
        logits = VIT_MODEL(input_tensor)
        prob_risk = torch.sigmoid(logits)[0, 0].item()

    # 3. Clinical Threshold (Optimized via Youden's J-Index)
    pred_cls = 1 if prob_risk >= 0.2741 else 0
    ai_abnormal = (pred_cls == 1)
    
    # Calculate confidence as a percentage
    confidence = prob_risk * 100 if ai_abnormal else (1.0 - prob_risk) * 100

    # 4. Generate XAI Base64 (Using the centered input_tensor and img_pil)
    heatmap_b64 = generate_xai_b64(VIT_MODEL, input_tensor, img_pil, pred_cls, DEVICE)

    return ai_abnormal, confidence, processed_b64, heatmap_b64


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def run_analysis(strokes: list, image_b64: str, age: int, education_years: int, device_dpi: float) -> dict:
    """
    Full clinical decision pipeline orchestrator.
    Expects raw payload data, extracts features, and runs clinical logic.
    """
    # 1. Kinematics Pre-processing
    pressure_supported = detect_pressure_support(strokes)
    processed_summary = process_strokes(strokes)

    # 2. Build velocity profile
    px_per_cm = device_dpi / 2.54 if device_dpi > 0 else 37.8 
    velocity_profile: list[float] = []
    
    if "processed_strokes" in processed_summary:
        for stroke in processed_summary["processed_strokes"]:
            if not stroke.get("eligible_for_kinematics"):
                continue
            duration_s = stroke["duration_ms"] / 1000.0
            if duration_s <= 0:
                continue
            if "smoothed_x" in stroke and "smoothed_y" in stroke:
                from core.kinematics import _smoothed_arc_length_px
                arc_px = _smoothed_arc_length_px(stroke["smoothed_x"], stroke["smoothed_y"])
            else:
                arc_px = stroke["path_length_px"]
            length_cm = arc_px / px_per_cm
            velocity_profile.append(round(length_cm / duration_s, 4))

    # 3. Kinematic Feature Extraction (K1-K5)
    features = extract_all_features(
        raw_strokes=strokes,
        processed_summary=processed_summary,
        device_dpi=device_dpi,
        pressure_supported=pressure_supported
    )

    # 4. Dynamic Thresholds & Rules Evaluation
    thresholds = get_dynamic_thresholds(age)
    k_results = evaluate_k_series(features, thresholds)

    motor_abnormal = k_results["K1"] or k_results["K2"] or k_results["K3"]
    cog_abnormal   = k_results["K4"] or k_results["K5"]
    
    # 5. Structural Analysis (Real ViT with Centering & Confidence)
    ai_abnormal, confidence, processed_b64, heatmap_b64 = run_real_vit_inference(image_b64)

    # 6. Final Truth Table and Risk Classification
    class_id = apply_truth_table(ai_abnormal, motor_abnormal, cog_abnormal)
    upstream_warnings = features.get("flags", [])
    result = classify_risk(class_id, education_years, upstream_warnings)

    # 7. Payload Generation
    return {
        "class_id": result["class_id"],
        "risk_level": result["risk_level"],
        "risk_color": result["risk_color"],
        "kinematic": {
            "K1_rms_cm":             features.get("K1_rms_cm"),
            "K2_velocity_cms":       features.get("K2_velocity_cms"),
            "K3_pressure_avg":       features.get("K3_pressure_avg"),
            "K3_pressure_decrement": features.get("K3_pressure_decrement"),
            "K4_pct_think_time":     features.get("K4_pct_think_time"),
            "K5_pfhl_ms":            features.get("K5_pfhl_ms"),
            "flags":                 features.get("flags", [])
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
        "model_version": "vit-b16-chefer-v3.0",
        "velocity_profile": velocity_profile,
        "ai_confidence": confidence,
        "processed_image_b64": processed_b64,
        "xai_evidence_b64": heatmap_b64  
    }