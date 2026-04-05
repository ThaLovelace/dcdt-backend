"""
kinematics.py
-------------
Kinematic feature extraction (K1–K5) for the dCDT analysis pipeline.

Each public function computes one biomarker family and returns None when
the data are insufficient or the hardware does not provide the required
sensor signal.  The orchestrator ``extract_all_features`` calls them in
order and assembles the final feature dict consumed by ``inference.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.schemas import StrokePoint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K4_NOISE_FILTER_MS: float = 500.0   # Gaps shorter than this are ignored (spec §3.5.4.3)
DRAWING_ORDER_ANOMALY_FLAG = "DRAWING_ORDER_ANOMALY"
PRESSURE_NOT_SUPPORTED_FLAG = "PRESSURE_NOT_SUPPORTED"
K5_SEGMENTATION_FAILED_FLAG = "K5_SEGMENTATION_FAILED"


# ---------------------------------------------------------------------------
# K1 — Tremor (RMS deviation)
# ---------------------------------------------------------------------------

def compute_k1_rms(
    raw_x:      list[float],
    raw_y:      list[float],
    smoothed_x: list[float],
    smoothed_y: list[float],
    device_dpi: float,
) -> float | None:
    """
    Compute K1: per-stroke RMS deviation of raw from smoothed trajectory.

    Formula (spec §3.5.4.1)::

        RMS_px  = sqrt( (1/N) * Σ [(xi − x̂i)² + (yi − ŷi)²] )
        RMS_cm  = RMS_px / (device_dpi / 2.54)

    Parameters
    ----------
    raw_x, raw_y:
        Original pixel coordinates captured from the canvas.
    smoothed_x, smoothed_y:
        Savitzky-Golay smoothed coordinates (same length as raw).
    device_dpi:
        Device screen resolution in dots per inch.

    Returns
    -------
    float
        RMS deviation in centimetres.
    None
        Returned when any guard condition is triggered:
        * ``len(raw_x) != len(smoothed_x)`` — array length mismatch
          (structural guarantee from preprocessing, but validated here
          as a defensive check).
        * ``len(raw_x) == 0`` — empty stroke.
        * ``device_dpi <= 0`` — invalid DPI.

    Raises
    ------
    ValueError
        If ``len(raw_x) != len(smoothed_x)`` — indicates a bug upstream.
    """
    # Guard: invalid DPI
    if device_dpi <= 0:
        return None

    # Guard: empty arrays
    if len(raw_x) == 0:
        return None

    # Guard: 1-to-1 mapping requirement
    if len(raw_x) != len(smoothed_x) or len(raw_y) != len(smoothed_y):
        raise ValueError(
            f"Array length mismatch in compute_k1_rms: "
            f"raw=({len(raw_x)}, {len(raw_y)}) "
            f"smoothed=({len(smoothed_x)}, {len(smoothed_y)}). "
            "Down-sampling or point deletion must never be applied."
        )

    raw_x_arr  = np.asarray(raw_x,      dtype=float)
    raw_y_arr  = np.asarray(raw_y,      dtype=float)
    sm_x_arr   = np.asarray(smoothed_x, dtype=float)
    sm_y_arr   = np.asarray(smoothed_y, dtype=float)

    sq_dev     = (raw_x_arr - sm_x_arr) ** 2 + (raw_y_arr - sm_y_arr) ** 2
    rms_px     = float(np.sqrt(np.mean(sq_dev)))

    px_per_cm  = device_dpi / 2.54
    return rms_px / px_per_cm


# ---------------------------------------------------------------------------
# K2 — Bradykinesia (average velocity)
# ---------------------------------------------------------------------------

def compute_k2_velocity(
    processed_strokes: list[dict],
    device_dpi:        float,
) -> float | None:
    """
    Compute K2: mean drawing velocity across all kinematic-eligible strokes.

    Formula (spec §3.5.4.1)::

        total_length_cm = Σ path_length_px / px_per_cm
        total_time_s    = Σ duration_ms / 1000
        velocity        = total_length_cm / total_time_s   [cm/s]

    Parameters
    ----------
    processed_strokes:
        List of stroke summaries from ``preprocessing.process_strokes``.
        Only strokes with ``eligible_for_kinematics == True`` contribute.
    device_dpi:
        Device screen resolution in dots per inch.

    Returns
    -------
    float
        Average velocity in cm/s.
    None
        Returned when ``total_time_s <= 0`` or ``device_dpi <= 0``.
    """
    if device_dpi <= 0:
        return None

    px_per_cm     = device_dpi / 2.54
    total_len_cm  = 0.0
    total_time_s  = 0.0

    for stroke in processed_strokes:
        if stroke.get("eligible_for_kinematics"):
            total_len_cm += stroke["path_length_px"] / px_per_cm
            total_time_s += stroke["duration_ms"] / 1000.0

    if total_time_s <= 0:
        return None

    return total_len_cm / total_time_s


# ---------------------------------------------------------------------------
# K3 — Micrographia (pressure)
# ---------------------------------------------------------------------------

def detect_pressure_support(all_strokes: list["StrokePoint"]) -> bool:
    """
    Determine whether the hardware provides meaningful pressure data.

    A constant pressure value (std < 0.01) or all-zero pressure indicates
    that the device does not support the Pointer Events pressure API and
    K3 should be skipped entirely (spec §3.5.4.1 footnote).

    Parameters
    ----------
    all_strokes:
        Every ``StrokePoint`` in the session.

    Returns
    -------
    bool
        True  — pressure varies; hardware is supported.
        False — pressure is constant or zero; K3 must be skipped.
    """
    if not all_strokes:
        return False

    pressures = np.asarray([pt.p for pt in all_strokes], dtype=float)

    if np.all(pressures == 0.0):
        return False

    if float(np.std(pressures)) < 0.01:
        return False

    return True


def compute_k3_pressure(
    strokes_dict:      dict[int, list["StrokePoint"]],
    sorted_stroke_ids: list[int],
    pressure_supported: bool,
) -> dict:
    """
    Compute K3 pressure variables: P_avg, P_first_stroke, P_last_stroke.

    ``P_first_stroke`` and ``P_last_stroke`` are the **mean pressure
    over all points in the first and last stroke respectively** — never
    a single data point — to reduce the effect of sensor noise
    (spec §3.5.4.3, note 3).

    Parameters
    ----------
    strokes_dict:
        Mapping stroke_id → list[StrokePoint].
    sorted_stroke_ids:
        Chronologically ordered stroke IDs.
    pressure_supported:
        Output of ``detect_pressure_support``.  If False, all fields
        are returned as None.

    Returns
    -------
    dict
        ``{"P_avg": float|None, "P_first": float|None, "P_last": float|None}``
    """
    empty = {"P_avg": None, "P_first": None, "P_last": None}

    if not pressure_supported or not sorted_stroke_ids:
        return empty

    all_pressures = [
        pt.p
        for sid in sorted_stroke_ids
        for pt in strokes_dict[sid]
    ]
    if not all_pressures:
        return empty

    p_avg   = float(np.mean(all_pressures))

    first_pts = strokes_dict[sorted_stroke_ids[0]]
    p_first   = float(np.mean([pt.p for pt in first_pts])) if first_pts else None

    last_pts  = strokes_dict[sorted_stroke_ids[-1]]
    p_last    = float(np.mean([pt.p for pt in last_pts])) if last_pts else None

    return {"P_avg": p_avg, "P_first": p_first, "P_last": p_last}


# ---------------------------------------------------------------------------
# K4 — Hesitation (%ThinkTime)
# ---------------------------------------------------------------------------

def compute_k4_think_time(
    strokes_dict:      dict[int, list["StrokePoint"]],
    sorted_stroke_ids: list[int],
    t_noise_ms:        float = K4_NOISE_FILTER_MS,
) -> dict | None:
    """
    Compute K4: percentage of think time (pen-up ratio).

    Definitions (spec §3.5.4.2)::

        T_ink   = Σ (t_end − t_start) for every stroke
        T_think = Σ gap_i  where gap_i = next.t_start − curr.t_end
                            and gap_i > t_noise_ms   (strictly greater)
                            Gaps ≤ t_noise_ms or negative are ignored.
        T_total = T_ink + T_think
        %ThinkTime = (T_think / T_total) × 100

    Parameters
    ----------
    strokes_dict:
        Mapping stroke_id → list[StrokePoint], points already in time order.
    sorted_stroke_ids:
        Chronologically ordered stroke IDs.
    t_noise_ms:
        Minimum gap duration to be counted as cognitive hesitation (default
        500 ms per Souillard-Mandar et al., 2016).

    Returns
    -------
    dict
        ``{T_ink_ms, T_think_ms, T_total_ms, pct_think_time}``
    None
        Returned immediately when ``T_total <= 0`` to prevent
        division by zero (spec §3.5.4.3 defensive guard).
    """
    if not sorted_stroke_ids:
        return None

    # --- Accumulate T_ink -----------------------------------------------
    T_ink: float = 0.0
    for sid in sorted_stroke_ids:
        pts = strokes_dict[sid]
        if len(pts) >= 2:
            T_ink += pts[-1].t - pts[0].t

    # --- Accumulate T_think (only gaps > t_noise_ms) --------------------
    T_think: float = 0.0
    for i in range(len(sorted_stroke_ids) - 1):
        current_end = strokes_dict[sorted_stroke_ids[i]][-1].t
        next_start  = strokes_dict[sorted_stroke_ids[i + 1]][0].t
        gap = next_start - current_end
        if gap > t_noise_ms:       # strictly greater — equal is ignored
            T_think += gap
        # gap <= t_noise_ms (including negative timestamps) → ignored

    T_total = T_ink + T_think

    # --- Defensive guard: prevent division by zero ----------------------
    if T_total <= 0:
        return None

    pct_think = (T_think / T_total) * 100.0

    return {
        "T_ink_ms":       T_ink,
        "T_think_ms":     T_think,
        "T_total_ms":     T_total,
        "pct_think_time": pct_think,
    }


# ---------------------------------------------------------------------------
# K5 — Pre-First Hand Latency (PFHL)
# ---------------------------------------------------------------------------

def _compute_bounding_box(
    points: list["StrokePoint"],
) -> tuple[float, float, float, float]:
    """Return (min_x, max_x, min_y, max_y) over the given points."""
    min_x = min(pt.x for pt in points)
    max_x = max(pt.x for pt in points)
    min_y = min(pt.y for pt in points)
    max_y = max(pt.y for pt in points)
    return min_x, max_x, min_y, max_y


def _stroke_is_clock_hand(
    points:           list["StrokePoint"],
    center_x:         float,
    center_y:         float,
    threshold_radius: float,
) -> bool:
    """
    Return True when the stroke centroid lies within *threshold_radius*
    of the canvas centre (centroid-based classifier for robustness).
    """
    if not points:
        return False

    centroid_x = sum(pt.x for pt in points) / len(points)
    centroid_y = sum(pt.y for pt in points) / len(points)
    distance   = math.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2)
    return distance <= threshold_radius


def _compute_k5_pre_first_hand_latency(
    strokes_dict:      dict[int, list["StrokePoint"]],
    sorted_stroke_ids: list[int],
    flags:             list[str],
) -> float | None:
    """
    Compute K5: time between the last digit stroke and the first hand stroke.

    Returns
    -------
    float
        PFHL in milliseconds (clamped to 0.0 when the patient drew hands
        before digits; ``DRAWING_ORDER_ANOMALY`` is appended to *flags*).
    None
        Returned — and ``K5_SEGMENTATION_FAILED`` appended to *flags* — in
        every edge case where a meaningful latency cannot be determined:
        * empty input
        * no hand stroke found
        * no digit stroke precedes the first hand stroke
        * the first stroke in the session is already a hand
    """
    if not strokes_dict or not sorted_stroke_ids:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None

    all_points: list["StrokePoint"] = [
        pt
        for sid in sorted_stroke_ids
        for pt in strokes_dict[sid]
    ]
    if not all_points:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None

    # Bounding box → canvas centre
    min_x, max_x, min_y, max_y = _compute_bounding_box(all_points)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # Threshold radius: 25 % of the smaller bounding dimension (floor 1 px)
    bbox_w            = max_x - min_x
    bbox_h            = max_y - min_y
    threshold_radius  = max(0.25 * min(bbox_w, bbox_h), 1.0)

    # Classify each stroke and record temporal boundaries
    # Each entry: (stroke_id, is_hand, t_start, t_end)
    classifications: list[tuple[int, bool, float, float]] = []
    for sid in sorted_stroke_ids:
        pts = strokes_dict[sid]
        if not pts:
            continue
        t_start  = pts[0].t
        t_end    = pts[-1].t
        is_hand  = _stroke_is_clock_hand(pts, center_x, center_y, threshold_radius)
        classifications.append((sid, is_hand, t_start, t_end))

    if not classifications:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None

    # Find first hand stroke
    first_hand_index: int | None = None
    for idx, (_, is_hand, _, _) in enumerate(classifications):
        if is_hand:
            first_hand_index = idx
            break

    if first_hand_index is None:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None

    if first_hand_index == 0:
        # First stroke is already a hand — no preceding digit exists
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None

    t_start_first_hand = classifications[first_hand_index][2]

    # Find the last digit/face stroke before the first hand
    t_end_last_digit: float | None = None
    for idx in range(first_hand_index - 1, -1, -1):
        _, is_hand, _, t_end = classifications[idx]
        if not is_hand:
            t_end_last_digit = t_end
            break

    if t_end_last_digit is None:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None

    latency_ms = t_start_first_hand - t_end_last_digit

    if latency_ms < 0:
        # Patient drew hands before digits; clamp and flag
        flags.append(DRAWING_ORDER_ANOMALY_FLAG)
        return 0.0

    return float(latency_ms)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_all_features(
    raw_strokes:        list["StrokePoint"],
    processed_summary:  dict,
    device_dpi:         float,
    pressure_supported: bool,
) -> dict:
    """
    Orchestrate K1–K5 extraction and return a unified feature dict.

    Parameters
    ----------
    raw_strokes:
        All original ``StrokePoint`` objects from the request.
    processed_summary:
        Output of ``preprocessing.process_strokes``.
    device_dpi:
        Device DPI for pixel→cm conversion (K1, K2).
    pressure_supported:
        Output of ``detect_pressure_support``.

    Returns
    -------
    dict
        Keys: K1_rms_cm, K2_velocity_cms, K3_pressure_avg,
              K3_pressure_decrement, K4_pct_think_time, K5_pfhl_ms, flags.
    """
    flags: list[str] = []

    if not raw_strokes:
        return {
            "K1_rms_cm":             None,
            "K2_velocity_cms":       None,
            "K3_pressure_avg":       None,
            "K3_pressure_decrement": None,
            "K4_pct_think_time":     None,
            "K5_pfhl_ms":            None,
            "flags":                 flags,
        }

    processed_strokes = processed_summary.get("processed_strokes", [])

    # --- Group raw strokes by ID ----------------------------------------
    strokes_dict: dict[int, list["StrokePoint"]] = {}
    for pt in raw_strokes:
        strokes_dict.setdefault(pt.id, []).append(pt)
    sorted_stroke_ids = sorted(strokes_dict.keys())

    # --- K1: RMS tremor -------------------------------------------------
    k1_rms_values: list[float] = []
    for stroke in processed_strokes:
        if not stroke.get("eligible_for_kinematics"):
            continue
        try:
            rms = compute_k1_rms(
                stroke["raw_x"],
                stroke["raw_y"],
                stroke["smoothed_x"],
                stroke["smoothed_y"],
                device_dpi,
            )
            if rms is not None:
                k1_rms_values.append(rms)
        except ValueError as exc:
            # Array mismatch — should never occur if preprocessing is correct
            flags.append(f"K1_ARRAY_MISMATCH: {exc}")

    k1_rms_cm: float | None = (
        float(np.mean(k1_rms_values)) if k1_rms_values else None
    )

    # --- K2: Velocity ---------------------------------------------------
    k2_velocity_cms = compute_k2_velocity(processed_strokes, device_dpi)

    # --- K3: Pressure ---------------------------------------------------
    if not pressure_supported:
        flags.append(PRESSURE_NOT_SUPPORTED_FLAG)

    k3_result   = compute_k3_pressure(strokes_dict, sorted_stroke_ids, pressure_supported)
    k3_avg      = k3_result["P_avg"]
    k3_decrement: float | None = None
    if k3_result["P_first"] is not None and k3_result["P_last"] is not None:
        if k3_result["P_first"] > 0:
            k3_decrement = k3_result["P_last"] / k3_result["P_first"]

    # --- K4: %ThinkTime -------------------------------------------------
    k4_result       = compute_k4_think_time(strokes_dict, sorted_stroke_ids)
    k4_pct_think    = k4_result["pct_think_time"] if k4_result is not None else None

    # --- K5: PFHL -------------------------------------------------------
    k5_pfhl_ms = _compute_k5_pre_first_hand_latency(
        strokes_dict, sorted_stroke_ids, flags
    )
    if k5_pfhl_ms is None and K5_SEGMENTATION_FAILED_FLAG not in flags:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)

    return {
        "K1_rms_cm":             k1_rms_cm,
        "K2_velocity_cms":       k2_velocity_cms,
        "K3_pressure_avg":       k3_avg,
        "K3_pressure_decrement": k3_decrement,
        "K4_pct_think_time":     k4_pct_think,
        "K5_pfhl_ms":            k5_pfhl_ms,
        "flags":                 flags,
    }