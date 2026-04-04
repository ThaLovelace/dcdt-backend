"""
test_dcdt.py
------------
Pytest unit tests for the dCDT backend pipeline.

Coverage
~~~~~~~~
* K1  — array length mismatch guard, empty input, valid computation
* K2  — zero duration guard, unit conversion
* K3  — hardware detection (constant pressure), P_first/P_last averaging
* K4  — gap < 500 ms ignored, gap > 500 ms accumulated,
         T_total = 0 returns None, negative gaps ignored
* K5  — no hand stroke returns None, drawing order anomaly returns 0.0,
         happy-path PFHL value
* Normalization — dynamic threshold formulae + lower bound (age = 999)
* Education Bias Warning — triggered / not triggered matrix
* Truth table — all 8 C0–C7 combinations
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal StrokePoint stub (avoids importing FastAPI / Pydantic in tests)
# ---------------------------------------------------------------------------

class _StrokePoint:
    """Lightweight stand-in for schemas.StrokePoint."""

    def __init__(self, t: float, x: float, y: float, p: float = 0.5,
                 az: float = 0.0, alt: float = 1.57, id: int = 1):
        self.t   = t
        self.x   = x
        self.y   = y
        self.p   = p
        self.az  = az
        self.alt = alt
        self.id  = id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stroke(n: int = 20, x_start: float = 0.0, y_start: float = 0.0,
                 dt: float = 8.0, pressure: float = 0.5,
                 stroke_id: int = 1) -> list[_StrokePoint]:
    """Return a simple straight stroke with *n* uniformly-spaced points."""
    return [
        _StrokePoint(
            t  = i * dt,
            x  = x_start + i * 2.0,
            y  = y_start,
            p  = pressure,
            id = stroke_id,
        )
        for i in range(n)
    ]


# ===========================================================================
# K1 — compute_k1_rms
# ===========================================================================

class TestK1RMS:

    def _import(self):
        from core.kinematics import compute_k1_rms
        return compute_k1_rms

    def test_array_length_mismatch_raises(self):
        """Mismatched raw / smoothed lengths must raise ValueError (not silently continue)."""
        compute_k1_rms = self._import()
        with pytest.raises(ValueError, match="Array length mismatch"):
            compute_k1_rms(
                raw_x      = [1.0, 2.0, 3.0],
                raw_y      = [0.0, 0.0, 0.0],
                smoothed_x = [1.0, 2.0],          # shorter — mismatch
                smoothed_y = [0.0, 0.0],
                device_dpi = 96.0,
            )

    def test_empty_arrays_returns_none(self):
        compute_k1_rms = self._import()
        result = compute_k1_rms([], [], [], [], device_dpi=96.0)
        assert result is None

    def test_invalid_dpi_returns_none(self):
        compute_k1_rms = self._import()
        result = compute_k1_rms([1.0], [1.0], [1.0], [1.0], device_dpi=0.0)
        assert result is None

    def test_negative_dpi_returns_none(self):
        compute_k1_rms = self._import()
        result = compute_k1_rms([1.0], [1.0], [1.0], [1.0], device_dpi=-1.0)
        assert result is None

    def test_perfect_smooth_gives_zero_rms(self):
        """When raw == smoothed, RMS deviation must be 0."""
        compute_k1_rms = self._import()
        coords = [float(i) for i in range(10)]
        result = compute_k1_rms(coords, coords, coords, coords, device_dpi=96.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_known_deviation_converted_to_cm(self):
        """
        raw   = [0, 0] in both axes
        smooth= [1, 1] in both axes → per-point error = sqrt(2) px
        RMS   = sqrt(2) px
        at 96 DPI → px_per_cm = 96/2.54 ≈ 37.795
        expected  ≈ sqrt(2) / 37.795
        """
        compute_k1_rms = self._import()
        import math
        result = compute_k1_rms(
            raw_x=[0.0, 0.0], raw_y=[0.0, 0.0],
            smoothed_x=[1.0, 1.0], smoothed_y=[1.0, 1.0],
            device_dpi=96.0,
        )
        expected = math.sqrt(2) / (96.0 / 2.54)
        assert result == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# K2 — compute_k2_velocity
# ===========================================================================

class TestK2Velocity:

    def _import(self):
        from core.kinematics import compute_k2_velocity
        return compute_k2_velocity

    def _make_processed(self, path_px: float, duration_ms: float) -> list[dict]:
        return [{
            "eligible_for_kinematics": True,
            "path_length_px": path_px,
            "duration_ms": duration_ms,
        }]

    def test_zero_duration_returns_none(self):
        compute_k2_velocity = self._import()
        result = compute_k2_velocity(
            self._make_processed(path_px=100.0, duration_ms=0.0),
            device_dpi=96.0,
        )
        assert result is None

    def test_invalid_dpi_returns_none(self):
        compute_k2_velocity = self._import()
        result = compute_k2_velocity(
            self._make_processed(path_px=100.0, duration_ms=1000.0),
            device_dpi=0.0,
        )
        assert result is None

    def test_unit_conversion(self):
        """
        path_length_px = 96 px, duration = 1000 ms, DPI = 96
        px_per_cm  = 96 / 2.54 ≈ 37.795
        length_cm  = 96 / 37.795 ≈ 2.54 cm
        velocity   = 2.54 / 1.0 = 2.54 cm/s
        """
        compute_k2_velocity = self._import()
        result = compute_k2_velocity(
            self._make_processed(path_px=96.0, duration_ms=1000.0),
            device_dpi=96.0,
        )
        assert result == pytest.approx(2.54, rel=1e-3)

    def test_ineligible_strokes_excluded(self):
        compute_k2_velocity = self._import()
        strokes = [
            {"eligible_for_kinematics": False, "path_length_px": 999.0, "duration_ms": 500.0},
            {"eligible_for_kinematics": True,  "path_length_px": 96.0,  "duration_ms": 1000.0},
        ]
        result = compute_k2_velocity(strokes, device_dpi=96.0)
        assert result == pytest.approx(2.54, rel=1e-3)


# ===========================================================================
# K3 — Pressure
# ===========================================================================

class TestK3Pressure:

    def _import(self):
        from core.kinematics import detect_pressure_support, compute_k3_pressure
        return detect_pressure_support, compute_k3_pressure

    def test_constant_pressure_not_supported(self):
        detect_pressure_support, _ = self._import()
        strokes = [_StrokePoint(t=i, x=0, y=0, p=0.5) for i in range(20)]
        assert detect_pressure_support(strokes) is False

    def test_all_zero_pressure_not_supported(self):
        detect_pressure_support, _ = self._import()
        strokes = [_StrokePoint(t=i, x=0, y=0, p=0.0) for i in range(10)]
        assert detect_pressure_support(strokes) is False

    def test_varying_pressure_supported(self):
        detect_pressure_support, _ = self._import()
        pressures = [0.2, 0.5, 0.8, 0.3, 0.9]
        strokes = [_StrokePoint(t=i, x=0, y=0, p=p) for i, p in enumerate(pressures)]
        assert detect_pressure_support(strokes) is True

    def test_p_first_and_p_last_are_stroke_averages(self):
        """
        P_first must be the mean of all points in the first stroke,
        P_last must be the mean of all points in the last stroke.
        """
        _, compute_k3_pressure = self._import()

        stroke1 = [_StrokePoint(t=i, x=0, y=0, p=0.2, id=1) for i in range(5)]
        stroke2 = [_StrokePoint(t=10+i, x=0, y=0, p=0.8, id=2) for i in range(5)]

        strokes_dict = {1: stroke1, 2: stroke2}
        result = compute_k3_pressure(strokes_dict, [1, 2], pressure_supported=True)

        assert result["P_first"] == pytest.approx(0.2, rel=1e-6)
        assert result["P_last"]  == pytest.approx(0.8, rel=1e-6)

    def test_returns_none_when_not_supported(self):
        _, compute_k3_pressure = self._import()
        stroke = [_StrokePoint(t=i, x=0, y=0, p=0.5, id=1) for i in range(5)]
        result = compute_k3_pressure({1: stroke}, [1], pressure_supported=False)
        assert result["P_avg"]   is None
        assert result["P_first"] is None
        assert result["P_last"]  is None


# ===========================================================================
# K4 — %ThinkTime
# ===========================================================================

class TestK4ThinkTime:

    def _import(self):
        from core.kinematics import compute_k4_think_time
        return compute_k4_think_time

    def _build_dict(self, stroke_defs: list[tuple[float, float, int]]):
        """Build strokes_dict from (t_start, t_end, id) tuples."""
        strokes_dict = {}
        for t_start, t_end, sid in stroke_defs:
            strokes_dict[sid] = [
                _StrokePoint(t=t_start, x=0, y=0, id=sid),
                _StrokePoint(t=t_end,   x=1, y=1, id=sid),
            ]
        return strokes_dict, sorted(strokes_dict.keys())

    def test_gap_below_500ms_is_ignored(self):
        """A 300 ms gap must not contribute to T_think."""
        compute_k4_think_time = self._import()
        # Stroke 1: 0→1000, Stroke 2: 1300→2300  (gap = 300 ms → ignored)
        sdict, sids = self._build_dict([(0, 1000, 1), (1300, 2300, 2)])
        result = compute_k4_think_time(sdict, sids, t_noise_ms=500.0)
        assert result is not None
        assert result["T_think_ms"] == pytest.approx(0.0)

    def test_gap_exactly_500ms_is_ignored(self):
        """A gap equal to T_noise must also be ignored (strictly greater than)."""
        compute_k4_think_time = self._import()
        sdict, sids = self._build_dict([(0, 1000, 1), (1500, 2500, 2)])
        result = compute_k4_think_time(sdict, sids, t_noise_ms=500.0)
        assert result is not None
        assert result["T_think_ms"] == pytest.approx(0.0)

    def test_gap_above_500ms_is_accumulated(self):
        """A 600 ms gap must be counted."""
        compute_k4_think_time = self._import()
        sdict, sids = self._build_dict([(0, 1000, 1), (1600, 2500, 2)])
        result = compute_k4_think_time(sdict, sids, t_noise_ms=500.0)
        assert result is not None
        assert result["T_think_ms"] == pytest.approx(600.0)

    def test_multiple_gaps_partial_accumulation(self):
        """Only gaps > 500 ms accumulate; others are ignored."""
        compute_k4_think_time = self._import()
        # Gaps: 300 (ignored), 600 (counted), 200 (ignored), 800 (counted) → 1400 ms
        sdict, sids = self._build_dict([
            (0,    1000, 1),
            (1300, 2300, 2),   # gap 300 — ignored
            (2900, 3500, 3),   # gap 600 — counted
            (3700, 4500, 4),   # gap 200 — ignored
            (5300, 6000, 5),   # gap 800 — counted
        ])
        result = compute_k4_think_time(sdict, sids, t_noise_ms=500.0)
        assert result is not None
        assert result["T_think_ms"] == pytest.approx(1400.0)

    def test_negative_gap_is_ignored(self):
        """A negative inter-stroke gap (disordered timestamps) must be ignored."""
        compute_k4_think_time = self._import()
        sdict, sids = self._build_dict([(0, 2000, 1), (1500, 3000, 2)])
        result = compute_k4_think_time(sdict, sids, t_noise_ms=500.0)
        assert result is not None
        assert result["T_think_ms"] == pytest.approx(0.0)

    def test_t_total_zero_returns_none(self):
        """When T_total ≤ 0, the function must return None (division-by-zero guard)."""
        compute_k4_think_time = self._import()
        # Single-point strokes: t_start == t_end → duration = 0, no gaps
        strokes_dict = {
            1: [_StrokePoint(t=100, x=0, y=0, id=1), _StrokePoint(t=100, x=0, y=0, id=1)],
        }
        result = compute_k4_think_time(strokes_dict, [1], t_noise_ms=500.0)
        assert result is None

    def test_pct_think_time_calculation(self):
        """
        T_ink   = 1000 + 900 = 1900 ms
        T_think = 600 ms (one gap > 500 ms)
        T_total = 2500 ms
        %ThinkTime = 600 / 2500 × 100 = 24.0 %
        """
        compute_k4_think_time = self._import()
        sdict, sids = self._build_dict([(0, 1000, 1), (1600, 2500, 2)])
        result = compute_k4_think_time(sdict, sids, t_noise_ms=500.0)
        assert result is not None
        assert result["pct_think_time"] == pytest.approx(24.0, rel=1e-6)
        assert result["T_ink_ms"]       == pytest.approx(1900.0)
        assert result["T_think_ms"]     == pytest.approx(600.0)
        assert result["T_total_ms"]     == pytest.approx(2500.0)


# ===========================================================================
# K5 — Pre-First Hand Latency
# ===========================================================================

class TestK5PFHL:

    def _import(self):
        from core.kinematics import _compute_k5_pre_first_hand_latency
        return _compute_k5_pre_first_hand_latency

    def test_no_hand_stroke_returns_none(self):
        """When no stroke can be classified as a hand, return None."""
        fn = self._import()
        # All strokes are far from canvas centre → classified as digits
        strokes = _make_stroke(20, x_start=300, y_start=300, stroke_id=1)
        sdict  = {1: strokes}
        flags  = []
        result = fn(sdict, [1], flags)
        assert result is None
        assert "K5_SEGMENTATION_FAILED" in flags

    def test_empty_input_returns_none(self):
        fn = self._import()
        flags = []
        result = fn({}, [], flags)
        assert result is None
        assert "K5_SEGMENTATION_FAILED" in flags

    def test_drawing_order_anomaly_returns_zero(self):
        """
        Anomaly Test: Hand centroid is centered using symmetric digits.
        Hand (id=3) is drawn with earliest timestamps (0-40ms), 
        while digits (id=1,2) start later (100ms+).
        Resulting negative latency should be clamped to 0.0 with a flag.
        """
        _compute_k5_pre_first_hand_latency = self._import()

        # 1. Hand stroke placed exactly at (200, 200)
        hand = [_StrokePoint(t=i*10, x=200, y=200, id=3) for i in range(5)]
        
        # 2. Digit 1 placed far left at (50, 200)
        d1 = [_StrokePoint(t=100+i*10, x=50, y=200, id=1) for i in range(5)]
        
        # 3. Digit 2 placed far right at (350, 200)
        # Bounding Box will be 50 to 350, center = 200. Hand is perfectly centered.
        d2 = [_StrokePoint(t=200+i*10, x=350, y=200, id=2) for i in range(5)]

        strokes_dict = {1: d1, 2: d2, 3: hand}
        flags = []
        
        # Execute with sorted IDs [1, 2, 3] so hand is not the first in order
        result = _compute_k5_pre_first_hand_latency(strokes_dict, [1, 2, 3], flags)

        # Assertion: Should be clamped to 0.0 due to negative latency (0 - 240)
        assert result == pytest.approx(0.0)
        assert "DRAWING_ORDER_ANOMALY" in flags


# ===========================================================================
# Normalization — dynamic thresholds
# ===========================================================================

class TestNormalization:

    def _import(self):
        from core.normalization import get_dynamic_thresholds
        return get_dynamic_thresholds

    @pytest.mark.parametrize("age, expected_k2", [
        (60,  3.00),
        (70,  2.70),
        (80,  2.40),
        (90,  2.10),
        (100, 1.80),
    ])
    def test_k2_threshold_by_age(self, age, expected_k2):
        get_dynamic_thresholds = self._import()
        t = get_dynamic_thresholds(age)
        assert t["K2_velocity_cms"] == pytest.approx(expected_k2, abs=1e-6)

    def test_k2_lower_bound_age_999(self):
        """Data-entry errors (age = 999) must be clamped to the 0.5 cm/s floor."""
        get_dynamic_thresholds = self._import()
        t = get_dynamic_thresholds(999)
        assert t["K2_velocity_cms"] == pytest.approx(0.5)

    @pytest.mark.parametrize("age, expected_k4", [
        (60, 40.0),
        (69, 40.0),   # same decade as 60
        (70, 43.0),
        (80, 46.0),
        (90, 49.0),
    ])
    def test_k4_threshold_by_age(self, age, expected_k4):
        get_dynamic_thresholds = self._import()
        t = get_dynamic_thresholds(age)
        assert t["K4_pct_think_time"] == pytest.approx(expected_k4)

    @pytest.mark.parametrize("age, expected_k5_ms", [
        (60,  8000.0),
        (70,  9500.0),
        (80, 11000.0),
        (90, 12500.0),
    ])
    def test_k5_threshold_by_age(self, age, expected_k5_ms):
        get_dynamic_thresholds = self._import()
        t = get_dynamic_thresholds(age)
        assert t["K5_pfhl_ms"] == pytest.approx(expected_k5_ms)

    def test_invalid_age_zero_uses_default(self):
        """age ≤ 0 should not raise and should return sensible thresholds."""
        get_dynamic_thresholds = self._import()
        t = get_dynamic_thresholds(0)
        assert t["K2_velocity_cms"] == pytest.approx(3.0)

    def test_fixed_thresholds_unchanged(self):
        get_dynamic_thresholds = self._import()
        t = get_dynamic_thresholds(65)
        assert t["K1_rms_threshold_cm"] == pytest.approx(0.05)
        assert t["K3_pressure_avg"]     == pytest.approx(0.25)
        assert t["K3_decrement_ratio"]  == pytest.approx(0.70)
        assert t["K4_noise_filter_ms"]  == pytest.approx(500.0)


# ===========================================================================
# Truth Table — apply_truth_table
# ===========================================================================

class TestTruthTable:

    def _import(self):
        from core.inference import apply_truth_table
        return apply_truth_table

    @pytest.mark.parametrize("ai, motor, cog, expected", [
        (False, False, False, "C0"),
        (False, True,  False, "C1"),
        (False, False, True,  "C2"),
        (False, True,  True,  "C3"),
        (True,  False, False, "C4"),
        (True,  False, True,  "C5"),
        (True,  True,  False, "C6"),
        (True,  True,  True,  "C7"),
    ])
    def test_all_eight_combinations(self, ai, motor, cog, expected):
        apply_truth_table = self._import()
        assert apply_truth_table(ai, motor, cog) == expected


# ===========================================================================
# Education Bias Warning — classify_risk
# ===========================================================================

class TestEducationWarning:

    def _import(self):
        from core.inference import classify_risk, EDUCATION_BIAS_WARNING
        return classify_risk, EDUCATION_BIAS_WARNING

    @pytest.mark.parametrize("class_id, education, expect_warning", [
        # Warning triggered (education < 8, class in {C4,C5,C6,C7})
        ("C4", 6,  True),
        ("C5", 0,  True),
        ("C6", 7,  True),
        ("C7", 5,  True),
        # Warning NOT triggered (education >= 8)
        ("C4", 8,  False),
        ("C4", 12, False),
        # Warning NOT triggered (class not in warning set)
        ("C0", 6,  False),
        ("C1", 3,  False),
        ("C2", 0,  False),
        ("C3", 7,  False),
    ])
    def test_education_warning_matrix(self, class_id, education, expect_warning):
        classify_risk, EDUCATION_BIAS_WARNING = self._import()
        result = classify_risk(class_id, education_years=education)
        if expect_warning:
            assert EDUCATION_BIAS_WARNING in result["warnings"], (
                f"Expected warning for class={class_id}, education={education}"
            )
        else:
            assert EDUCATION_BIAS_WARNING not in result["warnings"], (
                f"Unexpected warning for class={class_id}, education={education}"
            )

    def test_no_auto_downgrade(self):
        """Risk level must NOT change even when the education warning fires."""
        classify_risk, _ = self._import()
        result = classify_risk("C4", education_years=4)
        assert result["risk_level"] == "mild"    # still mild, not downgraded to normal
        assert result["risk_color"] == "yellow"


# ===========================================================================
# K-Series evaluation — evaluate_k_series
# ===========================================================================

class TestEvaluateKSeries:

    def _import(self):
        from core.inference import evaluate_k_series
        from core.normalization import get_dynamic_thresholds
        return evaluate_k_series, get_dynamic_thresholds

    def _default_features(self, **overrides) -> dict:
        base = {
            "K1_rms_cm":             0.02,   # below 0.05 → normal
            "K2_velocity_cms":       5.0,    # above 3.0 → normal
            "K3_pressure_avg":       0.5,    # above 0.25 → normal
            "K3_pressure_decrement": 0.9,    # above 0.70 → normal
            "K4_pct_think_time":     25.0,   # below 40 → normal
            "K5_pfhl_ms":            3000.0, # below 8000 → normal
            "flags":                 [],
        }
        base.update(overrides)
        return base

    def test_all_normal_produces_no_triggers(self):
        evaluate_k_series, get_dynamic_thresholds = self._import()
        thresholds = get_dynamic_thresholds(60)
        result = evaluate_k_series(self._default_features(), thresholds)
        assert not any(result.values())

    def test_k1_triggers_above_threshold(self):
        evaluate_k_series, get_dynamic_thresholds = self._import()
        thresholds = get_dynamic_thresholds(60)
        features = self._default_features(K1_rms_cm=0.10)
        result = evaluate_k_series(features, thresholds)
        assert result["K1"] is True

    def test_k2_triggers_below_threshold(self):
        evaluate_k_series, get_dynamic_thresholds = self._import()
        thresholds = get_dynamic_thresholds(60)
        features = self._default_features(K2_velocity_cms=1.0)
        result = evaluate_k_series(features, thresholds)
        assert result["K2"] is True

    def test_none_feature_does_not_trigger(self):
        """None values (hardware not supported) must never cause a trigger."""
        evaluate_k_series, get_dynamic_thresholds = self._import()
        thresholds = get_dynamic_thresholds(60)
        features = self._default_features(
            K1_rms_cm=None,
            K3_pressure_avg=None,
            K3_pressure_decrement=None,
            K5_pfhl_ms=None,
        )
        result = evaluate_k_series(features, thresholds)
        assert result["K1"] is False
        assert result["K3"] is False
        assert result["K5"] is False

    def test_k3_triggers_via_decrement(self):
        """K3 must trigger when P_last / P_first < 0.70, even if P_avg is normal."""
        evaluate_k_series, get_dynamic_thresholds = self._import()
        thresholds = get_dynamic_thresholds(60)
        features = self._default_features(
            K3_pressure_avg=0.5,          # normal
            K3_pressure_decrement=0.50,   # < 0.70 → abnormal
        )
        result = evaluate_k_series(features, thresholds)
        assert result["K3"] is True

# ===========================================================================
# Reporting System: Save results to file
# ===========================================================================
import os

def pytest_sessionfinish(session, exitstatus):
    """
    ฟังก์ชันนี้จะทำงานอัตโนมัติเมื่อรัน Test เสร็จ
    จะบันทึกผลลงไฟล์ test_report.txt ในโฟลเดอร์ที่สั่งรันคำสั่ง
    """
    total = session.testscollected
    failed = session.testsfailed
    passed = total - failed
    
    report_path = os.path.abspath("test_report.txt")
    
    report_content = [
        "=== dCDT Backend Test Summary ===",
        f"Total Tests: {total}",
        f"Passed:      {passed}",
        f"Failed:      {failed}",
        f"Status:      {'PASS' if failed == 0 else 'FAIL'}",
        f"Report Location: {report_path}",
        "================================="
    ]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
    
    # พิมพ์บอกที่อยู่ไฟล์บน Terminal เพื่อให้ก๊อปปี้ไปเปิดได้ง่าย
    print(f"\n\n" + "="*50)
    print(f"✅ TEST REPORT CREATED!")
    print(f"📍 Location: {report_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # สำหรับการรันด้วยคำสั่ง: python tests/test_dcdt.py
    import pytest
    pytest.main([__file__])