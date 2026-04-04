import pytest
import numpy as np
from models.schemas import StrokePoint
from core.preprocessing import compute_adaptive_window, process_strokes
from core.kinematics import extract_k_features

# ---------------------------------------------------------------------------
# TEST 1: ADAPTIVE WINDOW LOGIC
# ---------------------------------------------------------------------------
def test_adaptive_window_edge_cases():
    """Checks if the window size adapts correctly to sampling rates."""
    # Case: Zero delta (fallback to default odd number)
    assert compute_adaptive_window([0, 0, 0, 0]) == 11
    # Case: High frequency (1ms) -> 50ms/1ms = 50 -> next odd is 51
    assert compute_adaptive_window([0, 1, 2, 3]) == 51
    # Case: Low frequency (50ms) -> 50ms/50ms = 1 -> floor to MIN_WINDOW (5)
    assert compute_adaptive_window([0, 50, 100, 150]) == 5

# ---------------------------------------------------------------------------
# TEST 2: KINEMATICS & ELIGIBILITY Tiers
# ---------------------------------------------------------------------------
def test_three_tier_and_k_features():
    """Verifies that strokes are filtered correctly and K4 is calculated."""
    def make_stroke(s_id, count, start_t, dt=10.0, dx=2.0, dy=2.0):
        return [
            StrokePoint(
                t=start_t + i*dt, 
                x=float(i*dx), 
                y=float(i*dy), 
                p=0.5, az=0.0, alt=0.0, id=s_id
            ) for i in range(count)
        ]

    # Valid stroke
    stroke_1 = make_stroke(s_id=1, count=10, start_t=0)
    # Too short for smoothing (Tier 2 fail)
    stroke_2 = make_stroke(s_id=2, count=3, start_t=1000)
    # No movement (Tier 3 fail)
    stroke_3 = make_stroke(s_id=3, count=10, start_t=2000, dx=0.0, dy=0.0)

    raw_strokes = stroke_1 + stroke_2 + stroke_3
    processed_res = process_strokes(raw_strokes)
    
    # Check K4 calculation: Gap1(910) + Gap2(980) = 1890
    k_feat = extract_k_features(raw_strokes, processed_res)
    assert k_feat["K4_hesitation_ms"] == 1890.0
    assert "K5_pre_first_hand_latency_ms" in k_feat

# ---------------------------------------------------------------------------
# TEST 3: K5 GEOMETRIC LATENCY (THE FIX)
# ---------------------------------------------------------------------------
def test_k5_geometric_latency():
    """
    Validates K5 calculation by simulating a Digit and a Clock Hand.
    The Digit is placed in the corner, and the Hand is placed in the center.
    """
    # 1. Digit Stroke: Located at the top-left (10, 10). 
    # Its centroid will be far from the center of a (0,0 to 100,100) canvas.
    digit_stroke = [
        StrokePoint(t=100, x=10, y=10, p=0.5, az=0, alt=0, id=1),
        StrokePoint(t=150, x=15, y=15, p=0.5, az=0, alt=0, id=1) # Ends at 150ms
    ]
    
    # 2. Clock Hand Stroke: Located near the center (50, 50).
    hand_stroke = [
        StrokePoint(t=3150, x=50, y=50, p=0.5, az=0, alt=0, id=2), # Starts at 3150ms
        StrokePoint(t=3200, x=52, y=52, p=0.5, az=0, alt=0, id=2)
    ]
    
    # 3. Reference Stroke: We add a point at (100, 100) to ensure the bounding box 
    # encompasses the whole area, putting (50, 50) as the true center.
    ref_stroke = [
        StrokePoint(t=4000, x=100, y=100, p=0.5, az=0, alt=0, id=3),
        StrokePoint(t=4010, x=0, y=0, p=0.5, az=0, alt=0, id=3)
    ]
    
    raw_strokes = digit_stroke + hand_stroke + ref_stroke
    
    # Mock processed summary as it's not used for K5 logic
    processed_res = {"processed_strokes": []}
    
    k_feat = extract_k_features(raw_strokes, processed_res)
    
    # Target: t_start_hand(3150) - t_end_digit(150) = 3000.0 ms
    latency = k_feat["K5_pre_first_hand_latency_ms"]
    
    assert latency == 3000.0, f"Expected 3000.0 but got {latency}"

    # ---------------------------------------------------------------------------
# ADDITIONAL IMPORTS (Place these at the top of tests/test_pipeline.py)
# ---------------------------------------------------------------------------
from core.inference import run_vit_inference_mock
from models.schemas import DrawingPayload, StrokePoint

# ---------------------------------------------------------------------------
# TEST 4: INFERENCE TRUTH TABLE (C0 - C7 LOGIC)
# ---------------------------------------------------------------------------
def test_inference_truth_table():
    """
    Validates that the Mock ViT and Clinical Truth Table correctly 
    map K-features to the exact C0-C7 classification indices.
    """
    # Case 1: Fully Normal (C0)
    # All features are within normal physiological thresholds
    res_c0 = run_vit_inference_mock({
        "K1_jerk": 100.0, 
        "K4_hesitation_ms": 1000.0
    })
    assert res_c0["result_index"] == 0
    assert res_c0["domain_summary"]["motor_abnormal"] is False

    # Case 2: Pure Physical Risk / Tremor (C1)
    # High jerk magnitude (>500,000) combined with normal hesitation
    res_c1 = run_vit_inference_mock({
        "K1_jerk": 600000.0, 
        "K4_hesitation_ms": 1000.0
    })
    assert res_c1["result_index"] == 1
    assert res_c1["domain_summary"]["motor_abnormal"] is True

    # Case 3: Typical Alzheimer's Pattern (C5)
    # Extreme hesitation triggering both cognitive and AI structural flags
    res_c5 = run_vit_inference_mock({
        "K1_jerk": 100.0, 
        "K4_hesitation_ms": 20000.0 
    })
    assert res_c5["result_index"] == 5
    assert res_c5["domain_summary"]["cognitive_abnormal"] is True
    assert res_c5["domain_summary"]["ai_abnormal"] is True

# ---------------------------------------------------------------------------
# TEST 5: SCHEMA VALIDATION (AGE & EDUCATION)
# ---------------------------------------------------------------------------
def test_payload_schema_with_demographics():
    """
    Validates that the DrawingPayload correctly accepts and parses 
    the clinical demographic fields ('age' and 'education').
    """
    # Initialize a mock stroke point
    point = StrokePoint(t=0, x=10, y=10, p=0.5, az=0, alt=0, id=1)
    
    # Instantiate the payload simulating the frontend request
    payload = DrawingPayload(
        strokes=[point],
        age=72,
        education="Bachelor Degree"
    )
    
    # Assert demographic fields are successfully parsed
    assert payload.age == 72
    assert payload.education == "Bachelor Degree"
    assert len(payload.strokes) == 1