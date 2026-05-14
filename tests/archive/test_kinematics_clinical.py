"""
backend/tests/test_kinematics_clinical.py
==========================================
Unit Test Suite — dCDT Clinical Feature Accuracy
==================================================

Purpose
-------
Verify that K1, K2, K4, and K5 kinematic features are computed correctly
using mathematically generated (synthetic) strokes.

Each test is annotated with a "Clinical Reason" block explaining *why*
the assertion matters for patient safety / clinical validity.

Dependencies
------------
    pytest
    numpy
    scipy

Run with:
    pytest backend/tests/test_kinematics_clinical.py -v

Architecture notes
------------------
The tests bypass FastAPI and call the feature-extraction modules directly:
    core.preprocessing.process_strokes
    core.kinematics.compute_k1_rms, compute_k2_velocity,
                    compute_k4_think_time, _compute_k5_pre_first_hand_latency
This isolates clinical logic from network / DB concerns and makes every
assertion deterministic.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LIGHTWEIGHT STUB for models.schemas.StrokePoint
# ---------------------------------------------------------------------------
# The production codebase imports StrokePoint from `models.schemas`
# (a Pydantic model that lives in the FastAPI app).  Rather than pulling in
# the entire web-app dependency tree, we inject a minimal dataclass stub
# into sys.modules before importing the kinematics modules.  The stub
# exposes exactly the attributes the kinematics code reads: id, t, x, y, p.
# ---------------------------------------------------------------------------

@dataclass
class _StrokePoint:
    """Minimal stand-in for models.schemas.StrokePoint."""
    id: int
    t:  float   # timestamp in milliseconds
    x:  float   # pixel coordinate
    y:  float   # pixel coordinate
    p:  float = 0.5   # pressure (0–1), defaults to mid-range


# Inject the stub so `from models.schemas import StrokePoint` resolves cleanly.
_schemas_mod = types.ModuleType("models.schemas")
_schemas_mod.StrokePoint = _StrokePoint          # type: ignore[attr-defined]
sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules["models.schemas"] = _schemas_mod

# ---------------------------------------------------------------------------
# Stub for core.normalization (used only by inference.run_analysis; not
# needed by the lower-level functions we test here, but imported at module
# level by inference.py so we must satisfy the import).
# ---------------------------------------------------------------------------
_norm_mod = types.ModuleType("core.normalization")

def _get_dynamic_thresholds(age: int) -> dict:
    return {
        "K1_rms_threshold_cm": 0.03,
        "K2_velocity_cms":      2.0,
        "K3_pressure_avg":      0.3,
        "K3_decrement_ratio":   0.7,
        "K4_pct_think_time":   40.0,
        "K5_pfhl_ms":        5000.0,
    }

_norm_mod.get_dynamic_thresholds = _get_dynamic_thresholds   # type: ignore
# NOTE: Do NOT register a bare "core" module here — that would shadow the
# real core package on sys.path and prevent core.preprocessing / core.kinematics
# from loading.  Register only the sub-module that is missing at test time.
sys.modules["core.normalization"] = _norm_mod

# ---------------------------------------------------------------------------
# Now it is safe to import the real production modules.
# ---------------------------------------------------------------------------
from core.preprocessing import process_strokes                          # noqa: E402
from core.kinematics import (                                           # noqa: E402
    compute_k1_rms,
    compute_k2_velocity,
    compute_k4_think_time,
    _compute_k5_pre_first_hand_latency,
    K5_SEGMENTATION_FAILED_FLAG,
    K5_FALLBACK_USED_FLAG,
    DRAWING_ORDER_ANOMALY_FLAG,
)

# ---------------------------------------------------------------------------
# Shared constants (match production inference.py)
# ---------------------------------------------------------------------------

DEVICE_DPI       = 96.0          # Standard 96 DPI screen
PX_PER_CM        = DEVICE_DPI / 2.54   # ≈ 37.795 px/cm
K1_THRESHOLD_CM  = 0.03          # Updated threshold for Trajectory-B residual
K4_NOISE_MS      = 500.0         # Minimum gap counted as "think time"


# ===========================================================================
# Helper: stroke-point factory
# ===========================================================================

def make_stroke(
    stroke_id:   int,
    x_values:    list[float],
    y_values:    list[float],
    t_start_ms:  float,
    duration_ms: float,
    pressure:    float = 0.5,
) -> list[_StrokePoint]:
    """
    Build a list of StrokePoint objects from explicit x/y arrays.

    Timestamps are distributed evenly across [t_start_ms, t_start_ms +
    duration_ms], simulating a constant-rate digitiser.
    """
    n = len(x_values)
    assert n == len(y_values), "x and y arrays must be the same length"
    assert n >= 2, "A stroke needs at least 2 points"

    if n == 1:
        timestamps = [t_start_ms]
    else:
        timestamps = list(
            np.linspace(t_start_ms, t_start_ms + duration_ms, n)
        )

    return [
        _StrokePoint(id=stroke_id, t=timestamps[i],
                     x=x_values[i], y=y_values[i], p=pressure)
        for i in range(n)
    ]


# ===========================================================================
# ===========================================================================
#  TEST K1 — TREMOR RMS  (Trajectory B, stiff-reference residual)
# ===========================================================================
# ===========================================================================

class TestK1Tremor:
    """
    K1 measures the RMS deviation of raw pen coordinates from a stiff
    Savitzky-Golay reference (Trajectory B).  A straight, steady stroke
    should produce near-zero residual; a stroke corrupted by high-frequency
    tremor must exceed the clinical threshold of 0.03 cm.

    The dual-trajectory design (stiff reference for K1, balanced for K2)
    ensures the two biomarkers are decoupled: tremor suppression in K2 does
    not hide tremor in K1.
    """

    # -----------------------------------------------------------------------
    # Case A — Perfectly Straight Stroke
    # -----------------------------------------------------------------------

    def test_k1_straight_stroke_near_zero(self):
        """
        Clinical Reason
        ---------------
        A healthy patient draws clock hands in a smooth, straight motion.
        The stiff SG reference should fit such a stroke perfectly, leaving
        virtually zero residual.  If K1 falsely fires on a straight line,
        every healthy patient would be flagged for tremor — a serious
        false-positive risk.

        Synthetic Data
        --------------
        100 samples along a perfect horizontal line (y = 200 px, x: 0→500 px).
        The line is mathematically straight, so the stiff reference will
        follow it exactly; raw − reference ≈ 0 at every point.
        """
        N = 100
        x_vals = list(np.linspace(0.0, 500.0, N))
        y_vals = [200.0] * N
        # Simulate 60 Hz: one sample every ≈16.67 ms → 21-sample window = 350 ms
        t_vals = list(np.linspace(0.0, (N - 1) * (1000.0 / 60.0), N))

        rms_cm = compute_k1_rms(x_vals, y_vals, device_dpi=DEVICE_DPI, t_arr=t_vals)

        assert rms_cm is not None, "compute_k1_rms should return a float for valid input"
        assert rms_cm < K1_THRESHOLD_CM, (
            f"K1 for a perfectly straight stroke must be < {K1_THRESHOLD_CM} cm "
            f"(got {rms_cm:.6f} cm).  A false positive here would incorrectly "
            "classify healthy patients as showing tremor."
        )

    # -----------------------------------------------------------------------
    # Case B — Pathological Tremor Stroke (6 Hz sine wave overlay)
    # -----------------------------------------------------------------------

    def test_k1_tremor_stroke_exceeds_threshold(self):
        """
        Clinical Reason
        ---------------
        Parkinson's disease and essential tremor manifest as involuntary
        oscillations in the 4–8 Hz range.  The stiff reference is designed
        to span at least 2 full tremor cycles (K1_STIFF_MIN_WINDOW = 21
        samples) so it averages the oscillation out — it cannot track it —
        leaving a large residual that correctly captures the tremor.

        Why sample rate matters
        -----------------------
        The stiff window is capped at 21 samples regardless of device rate.
        At 200 Hz, 21 samples = 105 ms < one 6 Hz cycle (167 ms), so a
        quadratic fit can still partially follow the sinusoid and the
        residual is too small.  At 60 Hz (typical iOS/Android rate), 21
        samples = 350 ms ≈ 2.1 tremor cycles — the filter averages multiple
        cycles and produces a large residual.  This test uses 60 Hz to match
        the clinically targeted operating environment.

        Synthetic Data
        --------------
        60 samples at 60 Hz (1 second total). Straight horizontal baseline
        with a 6 Hz sine wave (amplitude 0.08 cm ≈ 3 px) overlaid on Y —
        mild tremor range from Souillard-Mandar et al. (2016), Table 4.
        """
        N            = 60            # 60 Hz: 21-sample window = 2+ tremor cycles
        duration_s   = 1.0
        tremor_hz    = 6.0
        amplitude_cm = 0.08
        amplitude_px = amplitude_cm * PX_PER_CM

        t_s    = np.linspace(0.0, duration_s, N)
        x_vals = list(np.linspace(0.0, 500.0, N))
        # Superimpose high-frequency tremor on a straight horizontal path
        y_vals = list(200.0 + amplitude_px * np.sin(2.0 * math.pi * tremor_hz * t_s))
        # Convert timestamps to milliseconds for compute_k1_rms
        t_vals = list(t_s * 1000.0)

        rms_cm = compute_k1_rms(x_vals, y_vals, device_dpi=DEVICE_DPI, t_arr=t_vals)

        assert rms_cm is not None, "compute_k1_rms should not return None for valid data"
        assert rms_cm > K1_THRESHOLD_CM, (
            f"K1 for a 6 Hz, {amplitude_cm} cm tremor must exceed the clinical "
            f"threshold of {K1_THRESHOLD_CM} cm (got {rms_cm:.6f} cm).  "
            "Failure here means the pipeline would MISS a pathological tremor "
            "and incorrectly classify an impaired patient as healthy."
        )

    # -----------------------------------------------------------------------
    # Boundary: invalid DPI returns None (no crash)
    # -----------------------------------------------------------------------

    def test_k1_invalid_dpi_returns_none(self):
        """
        Clinical Reason
        ---------------
        If device DPI is 0 or negative (misconfigured device), the pixel→cm
        conversion is undefined.  Returning None is safer than dividing by
        zero or emitting a nonsense value that could influence the clinical
        classification.
        """
        rms_cm = compute_k1_rms(
            [0.0, 100.0, 200.0],       # raw_x (positional)
            [0.0,   0.0,   0.0],       # raw_y (positional)
            0.0,                        # device_dpi = 0 → undefined px→cm
            t_arr=[0.0, 16.67, 33.33], # timestamps at ~60 Hz (irrelevant; returns None first)
        )
        assert rms_cm is None, (
            "compute_k1_rms must return None when device_dpi <= 0 "
            "to prevent undefined px→cm conversion."
        )


# ===========================================================================
# ===========================================================================
#  TEST K2 — VELOCITY (Trajectory A, balanced SG arc length)
# ===========================================================================
# ===========================================================================

class TestK2Velocity:
    """
    K2 measures average drawing velocity (cm/s) using the balanced
    Savitzky-Golay trajectory (Trajectory A) from preprocessing.
    Using the smoothed path prevents tremor artefacts from inflating
    arc length and therefore deflating the velocity reading.

    The formula is: K2 = Σ(arc_cm) / Σ(duration_s).
    """

    def _build_processed_stroke(
        self,
        stroke_id:    int,
        length_cm:    float,
        duration_ms:  float,
        n_points:     int = 100,
    ) -> dict:
        """
        Build a ``processed_stroke`` dict as preprocessing.process_strokes
        would return, but constructed analytically so path length is exact.

        A horizontal line with the exact arc length requested eliminates
        smoothing error from the test expectation.
        """
        length_px = length_cm * PX_PER_CM
        x_vals    = list(np.linspace(0.0, length_px, n_points))
        y_vals    = [0.0] * n_points

        return {
            "stroke_id":               stroke_id,
            "point_count":             n_points,
            "duration_ms":             duration_ms,
            "path_length_px":          length_px,
            # Smoothed == raw for a perfect straight line: no SG distortion.
            "smoothed_x":              x_vals,
            "smoothed_y":              y_vals,
            "raw_x":                   x_vals,
            "raw_y":                   y_vals,
            "pressure_values":         [0.5] * n_points,
            "eligible_for_timing":     True,
            "eligible_for_smoothing":  True,
            "eligible_for_kinematics": True,
            "jerk_magnitude":          None,
            "is_jerk_reliable":        False,
        }

    # -----------------------------------------------------------------------
    # Case A — Fast stroke: 10 cm in 2 seconds → 5.0 cm/s
    # -----------------------------------------------------------------------

    def test_k2_fast_stroke_exact_velocity(self):
        """
        Clinical Reason
        ---------------
        Bradykinesia (abnormally slow movement) is a core motor symptom of
        Parkinson's disease.  K2 must be accurate in both directions: a
        fast, healthy stroke should measure close to its theoretical velocity
        so the system does not incorrectly flag it as slow.

        Expected
        --------
        K2 = 10 cm / 2 s = 5.0 cm/s.
        Tolerance ±1 % to allow for floating-point rounding only.
        """
        stroke = self._build_processed_stroke(
            stroke_id=1, length_cm=10.0, duration_ms=2_000.0
        )
        velocity = compute_k2_velocity([stroke], device_dpi=DEVICE_DPI)

        assert velocity is not None, "K2 must not be None for a valid stroke"
        assert abs(velocity - 5.0) < 0.05, (
            f"K2 for 10 cm / 2 s must be ≈ 5.0 cm/s (got {velocity:.4f} cm/s)."
        )

    # -----------------------------------------------------------------------
    # Case B — Slow stroke: 10 cm in 10 seconds → 1.0 cm/s
    # -----------------------------------------------------------------------

    def test_k2_slow_stroke_exact_velocity(self):
        """
        Clinical Reason
        ---------------
        A stroke taking 10 s to cover 10 cm represents significant motor
        slowing.  K2 = 1.0 cm/s is well below a typical threshold of
        2 cm/s, and the pipeline must capture this accurately to generate
        a motor-abnormal signal.  Under-estimating slowness means the system
        misses bradykinesia.

        Expected
        --------
        K2 = 10 cm / 10 s = 1.0 cm/s.
        Tolerance ±1 %.
        """
        stroke = self._build_processed_stroke(
            stroke_id=1, length_cm=10.0, duration_ms=10_000.0
        )
        velocity = compute_k2_velocity([stroke], device_dpi=DEVICE_DPI)

        assert velocity is not None, "K2 must not be None for a valid stroke"
        assert abs(velocity - 1.0) < 0.01, (
            f"K2 for 10 cm / 10 s must be ≈ 1.0 cm/s (got {velocity:.4f} cm/s)."
        )

    # -----------------------------------------------------------------------
    # Edge case: ineligible strokes are skipped
    # -----------------------------------------------------------------------

    def test_k2_ineligible_strokes_skipped(self):
        """
        Clinical Reason
        ---------------
        Very short accidental touches or strokes below the kinematic
        eligibility threshold should not contaminate the velocity average.
        Including them would dilute the clinical signal from real drawing
        gestures.
        """
        good_stroke = self._build_processed_stroke(
            stroke_id=1, length_cm=10.0, duration_ms=2_000.0
        )
        bad_stroke = {
            **self._build_processed_stroke(stroke_id=2, length_cm=0.0, duration_ms=500.0),
            "eligible_for_kinematics": False,   # Mark as ineligible
        }
        velocity = compute_k2_velocity([good_stroke, bad_stroke], device_dpi=DEVICE_DPI)

        assert velocity is not None
        assert abs(velocity - 5.0) < 0.05, (
            "Ineligible strokes must not alter K2; expected ≈ 5.0 cm/s from "
            f"the eligible stroke only (got {velocity:.4f} cm/s)."
        )

    # -----------------------------------------------------------------------
    # Edge case: no eligible strokes returns None
    # -----------------------------------------------------------------------

    def test_k2_no_eligible_data_returns_none(self):
        """
        Clinical Reason
        ---------------
        If no kinematic-eligible strokes exist (e.g., patient lifted the
        pen immediately every time), K2 cannot be computed.  Returning None
        is the correct sentinel; a fabricated zero would incorrectly imply
        the patient drew at zero velocity, skewing downstream classification.
        """
        no_data: list[dict] = []
        velocity = compute_k2_velocity(no_data, device_dpi=DEVICE_DPI)
        assert velocity is None, (
            "K2 must return None when there are no eligible strokes."
        )


# ===========================================================================
# ===========================================================================
#  TEST K4 — %THINK TIME  (pen-up hesitation ratio)
# ===========================================================================
# ===========================================================================

class TestK4ThinkTime:
    """
    K4 captures the proportion of total task time the patient spent with
    the pen lifted (hesitating / planning).  A high %ThinkTime indicates
    cognitive load or planning difficulty.

    Formula:
        T_ink   = Σ stroke duration
        T_think = Σ inter-stroke gaps > 500 ms (noise filter)
        K4      = (T_think / (T_ink + T_think)) × 100

    Gaps ≤ 500 ms are treated as motor noise, not genuine hesitations
    (Souillard-Mandar et al., 2016, §3.5.4.2).
    """

    # -----------------------------------------------------------------------
    # Case A — Canonical 50 % scenario
    # -----------------------------------------------------------------------

    def test_k4_canonical_50_percent(self):
        """
        Clinical Reason
        ---------------
        Two strokes of 1 000 ms each with a 2 000 ms pause between them
        produces T_ink = 2 000 ms, T_think = 2 000 ms, K4 = 50 %.
        This is the simplest possible scenario for validating the ratio;
        an error here would indicate a fundamental arithmetic bug that
        could mis-classify any patient's cognitive hesitation score.

        Synthetic Data
        --------------
        Stroke 1: t = 0 → 1 000 ms   (id = 1)
        Pause:    t = 1 000 → 3 000 ms   (2 000 ms gap > 500 ms noise filter)
        Stroke 2: t = 3 000 → 4 000 ms  (id = 2)

        Expected: K4 = 50.0 %
        """
        stroke1 = make_stroke(
            stroke_id=1,
            x_values=list(np.linspace(0.0, 100.0, 50)),
            y_values=[100.0] * 50,
            t_start_ms=0.0,
            duration_ms=1_000.0,
        )
        stroke2 = make_stroke(
            stroke_id=2,
            x_values=list(np.linspace(0.0, 100.0, 50)),
            y_values=[200.0] * 50,
            t_start_ms=3_000.0,   # 2 000 ms gap after stroke 1 ends at 1 000 ms
            duration_ms=1_000.0,
        )
        all_pts = stroke1 + stroke2

        strokes_dict: dict[int, list] = {}
        for pt in all_pts:
            strokes_dict.setdefault(pt.id, []).append(pt)
        sorted_ids = sorted(strokes_dict.keys())

        result = compute_k4_think_time(strokes_dict, sorted_ids)

        assert result is not None, "K4 must return a result dict for valid strokes"
        assert abs(result["T_ink_ms"]   - 2_000.0) < 1.0, (
            f"T_ink should be 2000 ms (got {result['T_ink_ms']:.2f} ms)."
        )
        assert abs(result["T_think_ms"] - 2_000.0) < 1.0, (
            f"T_think should be 2000 ms (got {result['T_think_ms']:.2f} ms)."
        )
        assert abs(result["pct_think_time"] - 50.0) < 0.01, (
            f"K4 must be exactly 50.0 % for equal ink and think time "
            f"(got {result['pct_think_time']:.4f} %)."
        )

    # -----------------------------------------------------------------------
    # Case B — Short gap is filtered out (noise filter validation)
    # -----------------------------------------------------------------------

    def test_k4_short_gap_below_noise_filter_ignored(self):
        """
        Clinical Reason
        ---------------
        Brief pauses (e.g., 300 ms) between strokes reflect normal motor
        transitions, not cognitive hesitation.  Including them in T_think
        would inflate K4 and risk a false positive for cognitive impairment.
        Gaps ≤ 500 ms must be discarded by the noise filter.

        Synthetic Data
        --------------
        Stroke 1: t = 0 → 1 000 ms
        Short gap: 300 ms  (< 500 ms noise threshold — MUST BE IGNORED)
        Stroke 2: t = 1 300 → 2 300 ms

        Expected: T_think = 0 ms  →  K4 = 0 %
        """
        stroke1 = make_stroke(
            stroke_id=1,
            x_values=list(np.linspace(0.0, 100.0, 50)),
            y_values=[100.0] * 50,
            t_start_ms=0.0,
            duration_ms=1_000.0,
        )
        stroke2 = make_stroke(
            stroke_id=2,
            x_values=list(np.linspace(0.0, 100.0, 50)),
            y_values=[200.0] * 50,
            t_start_ms=1_300.0,   # 300 ms gap — below noise filter
            duration_ms=1_000.0,
        )
        all_pts = stroke1 + stroke2
        strokes_dict: dict[int, list] = {}
        for pt in all_pts:
            strokes_dict.setdefault(pt.id, []).append(pt)
        sorted_ids = sorted(strokes_dict.keys())

        result = compute_k4_think_time(strokes_dict, sorted_ids)

        assert result is not None
        assert result["T_think_ms"] == 0.0, (
            f"A 300 ms gap must not contribute to T_think "
            f"(got {result['T_think_ms']:.2f} ms)."
        )
        assert result["pct_think_time"] == 0.0, (
            "K4 must be 0 % when all inter-stroke gaps are below the noise filter."
        )

    # -----------------------------------------------------------------------
    # Case C — Multiple strokes, mixed gaps
    # -----------------------------------------------------------------------

    def test_k4_mixed_gaps_only_long_counted(self):
        """
        Clinical Reason
        ---------------
        A real clock-drawing session contains many short transitions between
        digits and a few long pauses before drawing the hands.  Only the
        long pauses (> 500 ms) should count toward T_think; the short ones
        must be invisible to K4.  This test proves the noise filter works
        correctly across multiple strokes.

        Synthetic Data
        --------------
        Stroke 1: 0 → 500 ms
        Gap: 200 ms (NOISE — ignored)
        Stroke 2: 700 → 1 200 ms
        Gap: 1 500 ms (THINK TIME — counted)
        Stroke 3: 2 700 → 3 200 ms

        T_ink   = 500 + 500 + 500 = 1 500 ms
        T_think = 1 500 ms (only the long gap)
        K4      = 1500 / 3000 × 100 = 50 %
        """
        s1 = make_stroke(1, list(np.linspace(0.0, 50.0, 30)), [100.0]*30, 0.0,    500.0)
        s2 = make_stroke(2, list(np.linspace(0.0, 50.0, 30)), [200.0]*30, 700.0,  500.0)
        s3 = make_stroke(3, list(np.linspace(0.0, 50.0, 30)), [300.0]*30, 2700.0, 500.0)

        all_pts = s1 + s2 + s3
        strokes_dict: dict[int, list] = {}
        for pt in all_pts:
            strokes_dict.setdefault(pt.id, []).append(pt)
        sorted_ids = sorted(strokes_dict.keys())

        result = compute_k4_think_time(strokes_dict, sorted_ids)

        assert result is not None
        assert abs(result["T_ink_ms"]   - 1_500.0) < 1.0, (
            f"T_ink should be 1500 ms (got {result['T_ink_ms']:.2f} ms)."
        )
        assert abs(result["T_think_ms"] - 1_500.0) < 1.0, (
            f"T_think should be 1500 ms (got {result['T_think_ms']:.2f} ms)."
        )
        assert abs(result["pct_think_time"] - 50.0) < 0.01, (
            f"K4 should be 50 % (got {result['pct_think_time']:.4f} %)."
        )


# ===========================================================================
# ===========================================================================
#  TEST K5 — PRE-FIRST HAND LATENCY (PFHL)
# ===========================================================================
# ===========================================================================

class TestK5HandIdentification:
    """
    K5 (Pre-First Hand Latency) measures the time from the start of the
    drawing session to the moment the patient begins drawing the first clock
    hand.  Prolonged latency indicates difficulty with spatial planning or
    motor initiation.

    The multi-feature classifier scores each stroke on three features:
        F1  Normalised arc-length  (hands are the longest strokes)
        F2  Straightness index     (hands are nearly straight lines)
        F3  Radial origin proximity (hands start near the clock centre)
    Plus a HIGH-PRIORITY OVERRIDE for strokes that are both very straight
    (F2 > 0.90) and start near the centre (within 35 % of clock radius).

    A score ≥ 0.50 classifies a stroke as a clock hand.
    """

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_digit_stroke(
        stroke_id:  int,
        cx:         float,
        cy:         float,
        radius:     float,
        angle_deg:  float,
        t_start_ms: float,
        n_pts:      int = 30,
        digit_r:    float = 20.0,
    ) -> list[_StrokePoint]:
        """
        Small semicircular arc near the clock rim — simulates a printed digit.

        Geometry rationale
        ------------------
        Clock digits are compact glyphs drawn near the circumference.
        A semicircle of radius ``digit_r`` centered on the rim has:
          * F1 (length) ≈ 0.37  — short relative to the long hand strokes
          * F2 (straightness) ≈ 0.64  — curved, not straight
          * F3 (radial origin) ≈ 0.0  — first point is on the rim, far from centre
        Weighted score = 0.40×0.37 + 0.35×0.64 + 0.25×0.0 ≈ 0.37 < 0.50 threshold.
        This ensures digit strokes are never misclassified as hands.
        """
        center_x = cx + radius * math.cos(math.radians(angle_deg))
        center_y = cy + radius * math.sin(math.radians(angle_deg))
        # Semicircle arc (180°): straightness = 2r/πr ≈ 0.637
        angles = np.linspace(0.0, math.pi, n_pts)
        x_vals = list(center_x + digit_r * np.cos(angles))
        y_vals = list(center_y + digit_r * np.sin(angles))
        return make_stroke(
            stroke_id=stroke_id,
            x_values=x_vals,
            y_values=y_vals,
            t_start_ms=t_start_ms,
            duration_ms=500.0,
            pressure=0.5,
        )

    @staticmethod
    def _make_hand_stroke(
        stroke_id:  int,
        cx:         float,
        cy:         float,
        length_px:  float,
        angle_deg:  float,
        t_start_ms: float,
        n_pts:      int = 40,
    ) -> list[_StrokePoint]:
        """
        Perfectly straight radial line from the clock centre — simulates
        a clock hand.  Straightness index ≈ 1.0 and first point is at the
        centre, ensuring F2 and F3 are near their maximum.
        """
        end_x = cx + length_px * math.cos(math.radians(angle_deg))
        end_y = cy + length_px * math.sin(math.radians(angle_deg))
        x_vals = list(np.linspace(cx, end_x, n_pts))
        y_vals = list(np.linspace(cy, end_y, n_pts))
        return make_stroke(
            stroke_id=stroke_id,
            x_values=x_vals,
            y_values=y_vals,
            t_start_ms=t_start_ms,
            duration_ms=800.0,
            pressure=0.5,
        )

    # -----------------------------------------------------------------------
    # Case A — Typical session: 12 digits first, 2 hands second
    # -----------------------------------------------------------------------

    def test_k5_hands_identified_and_pfhl_computed(self):
        """
        Clinical Reason
        ---------------
        A cognitively healthy patient typically draws the 12 clock digits
        before adding the hands.  K5 should correctly segment the session
        into digits (curved, peripheral) and hands (straight, from centre),
        and report PFHL as the time elapsed before the first hand stroke.

        A failure here means the pipeline either:
        (a) Mis-classifies digits as hands → PFHL under-estimated (false
            negative for planning difficulty), OR
        (b) Mis-classifies hands as digits → PFHL over-estimated (false
            positive for planning difficulty).

        Synthetic Data
        --------------
        Clock parameters: centre (300, 300), radius = 200 px, DPI = 96.
        • 12 digit strokes (curved arcs at 30° increments on rim)
          Each stroke starts 1 000 ms apart, so t ∈ [0, 11 000] ms.
        • 2 hand strokes (straight, from centre) starting at t = 15 000 ms
          and t = 16 000 ms respectively.

        Expected: PFHL = 15 000 ms (session start = t=0, first hand at t=15000)
        """
        cx, cy   = 300.0, 300.0
        radius   = 200.0
        all_pts: list[_StrokePoint] = []

        # 12 digit strokes (IDs 1–12)
        for i in range(12):
            angle = i * 30.0
            t     = i * 1_000.0
            pts   = self._make_digit_stroke(
                stroke_id=i + 1, cx=cx, cy=cy, radius=radius,
                angle_deg=angle, t_start_ms=t,
            )
            all_pts.extend(pts)

        # 2 hand strokes (IDs 13, 14) — minute and hour hand
        hand_minute = self._make_hand_stroke(
            stroke_id=13, cx=cx, cy=cy, length_px=radius * 0.90,
            angle_deg=90.0, t_start_ms=15_000.0,
        )
        hand_hour = self._make_hand_stroke(
            stroke_id=14, cx=cx, cy=cy, length_px=radius * 0.60,
            angle_deg=270.0, t_start_ms=16_000.0,
        )
        all_pts.extend(hand_minute)
        all_pts.extend(hand_hour)

        # Build strokes_dict
        strokes_dict: dict[int, list] = {}
        for pt in all_pts:
            strokes_dict.setdefault(pt.id, []).append(pt)
        sorted_ids = sorted(strokes_dict.keys())

        flags: list[str] = []
        pfhl_ms, seg_log = _compute_k5_pre_first_hand_latency(
            strokes_dict, sorted_ids, flags
        )

        # --- Segmentation must have succeeded ---
        assert K5_SEGMENTATION_FAILED_FLAG not in flags, (
            "K5 segmentation should not fail for a well-formed 14-stroke session. "
            f"Flags: {flags}"
        )
        assert pfhl_ms is not None, "PFHL must not be None for a valid session"

        # --- PFHL must reflect the first hand stroke (t=15 000 ms) ---
        # session starts at t=0 (first digit), so PFHL = 15 000 ms
        assert abs(pfhl_ms - 15_000.0) < 200.0, (
            f"PFHL should be ≈ 15 000 ms (first hand at t=15 000, session start "
            f"at t=0). Got {pfhl_ms:.2f} ms.  A large error here means the "
            "classifier is picking the wrong stroke as the first hand."
        )

        # --- Both hand strokes must appear in the seg_log as hands ---
        hand_log_entries = [e for e in seg_log if e.get("classified_as_hand")]
        hand_ids_found   = {e["stroke_id"] for e in hand_log_entries}
        assert 13 in hand_ids_found, (
            "Stroke 13 (minute hand) was not classified as a hand. "
            "The feature-based scorer failed to recognise a straight radial stroke."
        )
        assert 14 in hand_ids_found, (
            "Stroke 14 (hour hand) was not classified as a hand."
        )

    # -----------------------------------------------------------------------
    # Case B — Fallback activates when scorer cannot identify hands
    # -----------------------------------------------------------------------

    def test_k5_fallback_to_longest_strokes(self):
        """
        Clinical Reason
        ---------------
        Edge case: if a patient draws in an unusual style (e.g., looping
        or circular strokes instead of straight hands), the feature-based
        scorer may find no stroke with score ≥ 0.50.  The fallback rule
        promotes the 2 longest strokes to "hand" status so K5 always returns
        a result rather than silently failing.  The K5_FALLBACK_USED_FLAG
        must be set so clinicians know the output is an arc-length heuristic,
        not feature-based classification.

        Synthetic Data — geometry rationale
        ------------------------------------
        Full-circle loops drawn at the clock rim have:
          F2 (straightness) = 0.0   — start == end, chord = 0
          F3 (radial origin) ≈ 0.0  — first point is on the rim, far from centre
          Score = 0.40 × F1 + 0.35 × 0 + 0.25 × 0 = 0.40 × F1 < 0.50

        Session layout
        --------------
        Stroke 0: tiny 2-point stub at t=0 ms   (becomes sorted_ids[0])
            → never selected by the fallback (too short)
            → prevents DRAWING_ORDER_ANOMALY when fallback promotes stroke 1
        Stroke 1: circle r=20 px at 0°    rim,  t=1 000 ms
        Stroke 2: circle r=28 px at 120°  rim,  t=2 500 ms  ← longest, fallback hand
        Stroke 3: circle r=15 px at 240°  rim,  t=4 000 ms

        All 3 circle strokes score < 0.50 → fallback activates.
        Fallback selects strokes 2 & 1 (longest by arc).
        PFHL = t_start(stroke 1) − t_start(stroke 0) = 1 000 − 0 = 1 000 ms.
        """
        cx, cy   = 300.0, 300.0
        radius   = 200.0
        all_pts: list[_StrokePoint] = []

        # Stroke 0: tiny stub — establishes session start at t=0 without
        # being selected as a "hand" by the arc-length fallback.
        all_pts += make_stroke(
            stroke_id=0,
            x_values=[50.0, 52.0], y_values=[50.0, 52.0],
            t_start_ms=0.0, duration_ms=100.0,
        )

        # Strokes 1-3: full-circle loops at rim (score ≈ 0.30–0.47 < 0.50)
        circle_specs = [
            (1,  0.0,   1_000.0, 20.0),   # stroke_id, angle, t_start, circle_r
            (2, 120.0,  2_500.0, 28.0),   # largest circle → fallback hand
            (3, 240.0,  4_000.0, 15.0),
        ]
        for sid, angle_deg, t_start, cr in circle_specs:
            bx  = cx + radius * math.cos(math.radians(angle_deg))
            by  = cy + radius * math.sin(math.radians(angle_deg))
            ang = np.linspace(0.0, 2.0 * math.pi, 30, endpoint=False)
            xv  = list(bx + cr * np.cos(ang))
            yv  = list(by + cr * np.sin(ang))
            all_pts += make_stroke(
                stroke_id=sid, x_values=xv, y_values=yv,
                t_start_ms=t_start, duration_ms=600.0,
            )

        strokes_dict: dict[int, list] = {}
        for pt in all_pts:
            strokes_dict.setdefault(pt.id, []).append(pt)
        sorted_ids = sorted(strokes_dict.keys())   # [0, 1, 2, 3]

        flags: list[str] = []
        pfhl_ms, seg_log = _compute_k5_pre_first_hand_latency(
            strokes_dict, sorted_ids, flags
        )

        # Fallback flag must be set — all circle strokes score < 0.50
        assert K5_FALLBACK_USED_FLAG in flags, (
            "K5_FALLBACK_USED_FLAG must be set when no stroke scores ≥ 0.50. "
            "Without this flag, clinicians cannot distinguish a feature-based "
            "classification from an arc-length heuristic estimate."
        )
        # No drawing-order anomaly — stroke 0 (stub) is always first
        assert DRAWING_ORDER_ANOMALY_FLAG not in flags, (
            "DRAWING_ORDER_ANOMALY must NOT fire here; the fallback selects "
            "strokes 1 & 2 as hands, neither of which is sorted_ids[0]."
        )
        # A valid PFHL must still be produced
        assert pfhl_ms is not None, (
            "K5 must produce a PFHL value even via the fallback path."
        )
        assert pfhl_ms > 0.0, (
            f"PFHL must be positive (got {pfhl_ms} ms); the first 'hand' stroke "
            "starts well after the session-opening stub stroke."
        )

    # -----------------------------------------------------------------------
    # Case C — Drawing order anomaly: hand drawn first triggers flag
    # -----------------------------------------------------------------------

    def test_k5_drawing_order_anomaly_hand_drawn_first(self):
        """
        Clinical Reason
        ---------------
        Some patients with executive dysfunction draw clock hands before
        digits.  This produces a negative or zero PFHL (or PFHL == 0 when
        the first stroke IS the first hand), which is clinically meaningful
        but must not crash the pipeline.  The DRAWING_ORDER_ANOMALY flag
        must be raised and PFHL clamped to 0.0.

        Synthetic Data
        --------------
        Stroke 1 (ID=1): clock hand (straight, from centre) — drawn FIRST
        Strokes 2–4 (ID=2–4): digit arcs — drawn after
        """
        cx, cy = 300.0, 300.0
        radius = 200.0
        all_pts: list[_StrokePoint] = []

        # Hand drawn first (stroke id=1)
        hand_pts = self._make_hand_stroke(
            stroke_id=1, cx=cx, cy=cy, length_px=radius * 0.85,
            angle_deg=90.0, t_start_ms=0.0,
        )
        all_pts.extend(hand_pts)

        # Digits drawn after (strokes 2–4)
        for i in range(3):
            pts = self._make_digit_stroke(
                stroke_id=i + 2, cx=cx, cy=cy, radius=radius,
                angle_deg=i * 90.0, t_start_ms=2_000.0 + i * 1_000.0,
            )
            all_pts.extend(pts)

        strokes_dict: dict[int, list] = {}
        for pt in all_pts:
            strokes_dict.setdefault(pt.id, []).append(pt)
        sorted_ids = sorted(strokes_dict.keys())

        flags: list[str] = []
        pfhl_ms, _ = _compute_k5_pre_first_hand_latency(
            strokes_dict, sorted_ids, flags
        )

        assert DRAWING_ORDER_ANOMALY_FLAG in flags, (
            "DRAWING_ORDER_ANOMALY must be flagged when the first stroke "
            "is classified as a hand."
        )
        assert pfhl_ms == 0.0, (
            "PFHL must be clamped to 0.0 when the hand is drawn first "
            f"(got {pfhl_ms})."
        )


# ===========================================================================
# ===========================================================================
#  INTEGRATION SMOKE TEST — process_strokes → K1/K2 pipeline
# ===========================================================================
# ===========================================================================

class TestPreprocessingToKinematicsIntegration:
    """
    Verify that the full preprocessing → kinematics pipeline produces
    sensible K1 and K2 values when called end-to-end with raw StrokePoints.

    This catches regressions where a change in preprocessing (e.g. window
    size, array format) silently breaks the feature extraction step.
    """

    def test_integration_straight_stroke_k1_near_zero_k2_plausible(self):
        """
        Clinical Reason
        ---------------
        A single 5 cm straight stroke drawn at 200 Hz for 1 second should
        produce K1 ≈ 0 (no tremor) and K2 ≈ 5 cm/s.  If either value
        deviates substantially it indicates a unit-inconsistency bug
        somewhere in the pre-processing → kinematics boundary.

        Synthetic Data
        --------------
        200-point horizontal line, x: 0 → 5*PX_PER_CM, y = 0, 1 second.
        """
        N          = 200
        length_cm  = 5.0
        length_px  = length_cm * PX_PER_CM
        duration_ms = 1_000.0

        x_vals = list(np.linspace(0.0, length_px, N))
        y_vals = [0.0] * N

        pts = make_stroke(
            stroke_id=1,
            x_values=x_vals,
            y_values=y_vals,
            t_start_ms=0.0,
            duration_ms=duration_ms,
        )

        summary = process_strokes(pts)
        pstrokes = summary["processed_strokes"]

        assert len(pstrokes) == 1, "Exactly one processed stroke expected"
        ps = pstrokes[0]
        assert ps["eligible_for_kinematics"], (
            "A 200-point, 1-second stroke must be kinematic-eligible."
        )

        # K1 — should be near zero (straight stroke)
        # Extract timestamps from the raw stroke points to drive dynamic window selection.
        # At 200 Hz (1000 ms / 200 pts), the dynamic window = round(350 / 5) = 70 samples.
        t_arr_k1 = [pt.t for pt in pts]
        k1 = compute_k1_rms(ps["raw_x"], ps["raw_y"], device_dpi=DEVICE_DPI, t_arr=t_arr_k1)
        assert k1 is not None
        assert k1 < K1_THRESHOLD_CM, (
            f"Integration K1 for a straight 5 cm stroke must be < {K1_THRESHOLD_CM} cm "
            f"(got {k1:.6f} cm)."
        )

        # K2 — should be approximately length_cm / (duration_ms/1000)
        k2 = compute_k2_velocity(pstrokes, device_dpi=DEVICE_DPI)
        expected_k2 = length_cm / (duration_ms / 1_000.0)   # 5.0 cm/s
        assert k2 is not None
        assert abs(k2 - expected_k2) < 0.2, (
            f"Integration K2 for a 5 cm / 1 s stroke should be ≈ {expected_k2} cm/s "
            f"(got {k2:.4f} cm/s)."
        )