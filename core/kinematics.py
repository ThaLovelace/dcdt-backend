"""
kinematics.py
-------------
Kinematic feature extraction (K1–K5) for the dCDT analysis pipeline.

Each public function computes one biomarker family and returns None when
the data are insufficient or the hardware does not provide the required
sensor signal.  The orchestrator ``extract_all_features`` calls them in
order and assembles the final feature dict consumed by ``inference.py``.

K1 Dual-Trajectory Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
K1 tremor is measured as RMS( raw − Trajectory B ) where Trajectory B
is the "stiff" reference (350 ms window, poly=2) built in preprocessing.
This correctly isolates tremor oscillations from intended movement.
Trajectory A (50 ms, poly=3) is used only for K2 velocity.
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

K4_NOISE_FILTER_MS: float = 500.0
DRAWING_ORDER_ANOMALY_FLAG   = "DRAWING_ORDER_ANOMALY"
PRESSURE_NOT_SUPPORTED_FLAG  = "PRESSURE_NOT_SUPPORTED"
K5_SEGMENTATION_FAILED_FLAG  = "K5_SEGMENTATION_FAILED"
K5_FALLBACK_USED_FLAG        = "K5_FALLBACK_LONGEST_STROKES"


# ---------------------------------------------------------------------------
# K1 — Tremor (RMS deviation from Trajectory B)
# ---------------------------------------------------------------------------

def compute_k1_rms(
    raw_x:    list[float],
    raw_y:    list[float],
    stiff_x:  list[float],
    stiff_y:  list[float],
    device_dpi: float,
) -> float | None:
    """
    Compute K1: per-stroke RMS deviation of raw from Trajectory B (stiff reference).

    Dual-Trajectory Architecture (spec §3.5.4.1)::

        RMS_px  = sqrt( (1/N) * Σ [(xi − x̂B_i)² + (yi − ŷB_i)²] )
        RMS_cm  = RMS_px / (device_dpi / 2.54)

    Trajectory B is built with a 350 ms window and polyorder=2 so it
    follows the intended drawing direction without bending into tremor
    oscillations (4–8 Hz).  Subtracting raw from this stiff reference
    correctly isolates the tremor component.

    Parameters
    ----------
    raw_x, raw_y:
        Original pixel coordinates captured from the canvas.
    stiff_x, stiff_y:
        Trajectory B — stiff reference coordinates (same length as raw).
    device_dpi:
        Device screen resolution in dots per inch.

    Returns
    -------
    float
        RMS deviation in centimetres.
    None
        Returned when any guard condition is triggered:
        * ``len(raw_x) != len(stiff_x)`` — array length mismatch.
        * ``len(raw_x) == 0`` — empty stroke.
        * ``device_dpi <= 0`` — invalid DPI.

    Raises
    ------
    ValueError
        If ``len(raw_x) != len(stiff_x)`` — indicates a bug upstream.
    """
    if device_dpi <= 0:
        return None

    if len(raw_x) == 0:
        return None

    # Guard: 1-to-1 mapping requirement (raw vs Trajectory B)
    if len(raw_x) != len(stiff_x) or len(raw_y) != len(stiff_y):
        raise ValueError(
            f"Array length mismatch in compute_k1_rms: "
            f"raw=({len(raw_x)}, {len(raw_y)}) "
            f"stiff=({len(stiff_x)}, {len(stiff_y)}). "
            "Down-sampling or point deletion must never be applied."
        )

    raw_x_arr  = np.asarray(raw_x,   dtype=float)
    raw_y_arr  = np.asarray(raw_y,   dtype=float)
    stiff_x_arr = np.asarray(stiff_x, dtype=float)
    stiff_y_arr = np.asarray(stiff_y, dtype=float)

    # RMS deviation between raw and Trajectory B
    sq_dev  = (raw_x_arr - stiff_x_arr) ** 2 + (raw_y_arr - stiff_y_arr) ** 2
    rms_px  = float(np.sqrt(np.mean(sq_dev)))

    px_per_cm = device_dpi / 2.54
    return rms_px / px_per_cm


# ---------------------------------------------------------------------------
# K2 — Bradykinesia (average velocity from Trajectory A path length)
# ---------------------------------------------------------------------------

def _arc_length_px(points: list["StrokePoint"]) -> float:
    """
    Euclidean arc-length of a raw stroke in pixels.

    Used by K5 fallback when the classifier cannot identify any hand strokes;
    the two longest strokes (by raw arc-length) are promoted as candidates.
    """
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i].x - points[i - 1].x
        dy = points[i].y - points[i - 1].y
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _smoothed_arc_length_px(
    smoothed_x: list[float],
    smoothed_y: list[float],
) -> float:
    """
    Euclidean arc-length of a balanced-smoothed stroke (Trajectory A).

    Using smoothed rather than raw coordinates prevents tremor artefacts
    from inflating the measured path length and deflating velocity
    (Souillard-Mandar et al., 2016, §3.2).
    """
    if len(smoothed_x) < 2:
        return 0.0
    sx = np.asarray(smoothed_x, dtype=float)
    sy = np.asarray(smoothed_y, dtype=float)
    return float(np.sum(np.sqrt(np.diff(sx) ** 2 + np.diff(sy) ** 2)))


def compute_k2_velocity(
    processed_strokes: list[dict],
    device_dpi:        float,
) -> float | None:
    """
    Compute K2: mean drawing velocity across all kinematic-eligible strokes.

    Uses ``path_length_px`` which is pre-computed from **Trajectory A**
    (smoothed coordinates) in preprocessing to prevent tremor inflation
    (spec §3.5.4.2).

    Formula::

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

    px_per_cm    = device_dpi / 2.54
    total_len_cm = 0.0
    total_time_s = 0.0

    for stroke in processed_strokes:
        if not stroke.get("eligible_for_kinematics"):
            continue
        duration_s = stroke["duration_ms"] / 1000.0
        if duration_s <= 0:
            continue

        if "smoothed_x" in stroke and "smoothed_y" in stroke:
            arc_px = _smoothed_arc_length_px(stroke["smoothed_x"], stroke["smoothed_y"])
        else:
            arc_px = stroke.get("path_length_px", 0.0)

        if arc_px <= 0:
            continue

        total_len_cm += arc_px / px_per_cm
        total_time_s += duration_s

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
    strokes_dict:       dict[int, list["StrokePoint"]],
    sorted_stroke_ids:  list[int],
    pressure_supported: bool,
) -> dict:
    """
    Compute K3 pressure variables: P_avg, P_first_stroke, P_last_stroke.

    ``P_first_stroke`` and ``P_last_stroke`` are the **mean pressure
    over all points in the first and last stroke respectively** — never
    a single data point — to reduce the effect of sensor noise
    (spec §3.5.4.3, note 3).
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
        T_think = Σ gap_i  where gap_i > t_noise_ms (strictly greater)
        T_total = T_ink + T_think
        %ThinkTime = (T_think / T_total) × 100
    """
    if not sorted_stroke_ids:
        return None

    T_ink: float = 0.0
    for sid in sorted_stroke_ids:
        pts = strokes_dict[sid]
        if len(pts) >= 2:
            T_ink += pts[-1].t - pts[0].t

    T_think: float = 0.0
    for i in range(len(sorted_stroke_ids) - 1):
        current_end = strokes_dict[sorted_stroke_ids[i]][-1].t
        next_start  = strokes_dict[sorted_stroke_ids[i + 1]][0].t
        gap = next_start - current_end
        if gap > t_noise_ms:
            T_think += gap

    T_total = T_ink + T_think

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
    Multi-feature clock-hand classifier with override rule (spec §3.5.4.4).

    BUG-005 FIX: Previous implementation used only centroid distance, which
    misclassifies short digit strokes near the centre and long hand strokes
    whose centroid drifts to the periphery.

    Classification criteria (ALL features evaluated, override rule applied):

    Feature 1 — Centroid proximity:
        centroid distance to canvas centre <= threshold_radius  →  hand_vote

    Feature 2 — Stroke passes through centre zone:
        any point within (threshold_radius * 0.6) of centre  →  hand_vote

    Feature 3 — Arc length (relative to canvas diagonal):
        arc_length >= 0.15 * bbox_diagonal  →  hand_vote
        arc_length <  0.05 * bbox_diagonal  →  digit_vote  (short tick/mark)

    Override rule (spec §3.5.4.4, note on segmentation):
        • If arc_length < 0.04 * bbox_diagonal  →  force DIGIT regardless of votes
          (protects against tiny pen-down artefacts being promoted as hands)
        • If feature-2 fires AND arc_length >= 0.20 * bbox_diagonal  →  force HAND
          (a long stroke passing through the centre is almost certainly a hand)

    A stroke is classified as a hand when hand_votes >= 2 out of 3 features
    (majority vote), subject to the override rules above.
    """
    if not points:
        return False

    # --- Geometry helpers ------------------------------------------------
    centroid_x = sum(pt.x for pt in points) / len(points)
    centroid_y = sum(pt.y for pt in points) / len(points)
    centroid_dist = math.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2)

    # Arc length of this stroke
    arc_px = 0.0
    for i in range(1, len(points)):
        dx = points[i].x - points[i - 1].x
        dy = points[i].y - points[i - 1].y
        arc_px += math.sqrt(dx * dx + dy * dy)

    # Canvas diagonal (proxy for scale); threshold_radius is already 25 % of
    # min(bbox_w, bbox_h), so bbox_diagonal ≈ threshold_radius / 0.25 * sqrt(2).
    # We keep the scale factor relative to threshold_radius for robustness.
    bbox_diagonal = threshold_radius / 0.25 * math.sqrt(2)

    # --- Feature votes ---------------------------------------------------
    # Feature 1: centroid proximity
    f1_hand = centroid_dist <= threshold_radius

    # Feature 2: passes through centre zone
    centre_zone_r = threshold_radius * 0.6
    f2_hand = any(
        math.sqrt((pt.x - center_x) ** 2 + (pt.y - center_y) ** 2) <= centre_zone_r
        for pt in points
    )

    # Feature 3: arc length relative to canvas diagonal
    if arc_px >= 0.15 * bbox_diagonal:
        f3_hand = True
    elif arc_px < 0.05 * bbox_diagonal:
        f3_hand = False
    else:
        f3_hand = None  # neutral — no vote cast

    # --- Override rules (applied before majority vote) -------------------
    # Hard digit override: stroke too short to be a hand
    if arc_px < 0.04 * bbox_diagonal:
        return False

    # Hard hand override: long stroke passing through centre
    if f2_hand and arc_px >= 0.20 * bbox_diagonal:
        return True

    # --- Majority vote (hand_votes >= 2) ---------------------------------
    hand_votes = sum([
        1 if f1_hand else 0,
        1 if f2_hand else 0,
        1 if f3_hand is True else 0,
    ])
    return hand_votes >= 2


def _compute_k5_pre_first_hand_latency(
    strokes_dict:      dict[int, list["StrokePoint"]],
    sorted_stroke_ids: list[int],
    flags:             list[str],
) -> tuple[float | None, list[dict]]:
    """
    Compute K5: time between the last digit stroke and the first hand stroke.

    Returns a tuple of (latency_ms, seg_log) for compatibility with the
    audit-log contract expected by the orchestrator and any diagnostic routes.
    """
    seg_log: list[dict] = []

    if not strokes_dict or not sorted_stroke_ids:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None, seg_log

    all_points: list["StrokePoint"] = [
        pt
        for sid in sorted_stroke_ids
        for pt in strokes_dict[sid]
    ]
    if not all_points:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None, seg_log

    min_x, max_x, min_y, max_y = _compute_bounding_box(all_points)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    bbox_w           = max_x - min_x
    bbox_h           = max_y - min_y
    threshold_radius = max(0.25 * min(bbox_w, bbox_h), 1.0)

    classifications: list[tuple[int, bool, float, float]] = []
    for sid in sorted_stroke_ids:
        pts = strokes_dict[sid]
        if not pts:
            continue
        t_start = pts[0].t
        t_end   = pts[-1].t
        is_hand = _stroke_is_clock_hand(pts, center_x, center_y, threshold_radius)
        classifications.append((sid, is_hand, t_start, t_end))
        seg_log.append({
            "stroke_id":          sid,
            "classified_as_hand": is_hand,
            "fallback_promoted":  False,
        })

    if not classifications:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None, seg_log

    first_hand_index: int | None = None
    for idx, (_, is_hand, _, _) in enumerate(classifications):
        if is_hand:
            first_hand_index = idx
            break

    # Fallback: use the two longest strokes when no hand was detected
    if first_hand_index is None:
        arc_by_sid = {
            sid: _arc_length_px(strokes_dict[sid])
            for sid in sorted_stroke_ids
            if strokes_dict.get(sid)
        }
        candidate_ids = sorted(arc_by_sid, key=arc_by_sid.get, reverse=True)[:2]
        if not candidate_ids:
            flags.append(K5_SEGMENTATION_FAILED_FLAG)
            return None, seg_log
        flags.append(K5_FALLBACK_USED_FLAG)
        for entry in seg_log:
            if entry["stroke_id"] in candidate_ids:
                entry["classified_as_hand"] = True
                entry["fallback_promoted"]  = True
        # Re-derive first_hand_index from updated classifications
        for idx, (sid, _, _, _) in enumerate(classifications):
            if sid in candidate_ids:
                first_hand_index = idx
                break

    if first_hand_index is None or first_hand_index == 0:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None, seg_log

    # BUG-004 FIX: PFHL is defined as:
    #   PFHL = t_hand_start − t_digit_end
    # where t_hand_start is the timestamp of the FIRST point of the first
    # classified hand stroke, and t_digit_end is the timestamp of the LAST
    # point of the last digit stroke that precedes it.
    # The old (wrong) approach anchored to session/global start which caused
    # latency values to accumulate all preceding drawing time.
    t_hand_start = classifications[first_hand_index][2]   # t_start of first hand stroke

    t_digit_end: float | None = None
    for idx in range(first_hand_index - 1, -1, -1):
        _, is_hand, _, t_end = classifications[idx]
        if not is_hand:
            t_digit_end = t_end   # t_end of last digit stroke before first hand
            break

    if t_digit_end is None:
        flags.append(K5_SEGMENTATION_FAILED_FLAG)
        return None, seg_log

    latency_ms = t_hand_start - t_digit_end  # PFHL = t_hand_start − t_digit_end

    if latency_ms < 0:
        flags.append(DRAWING_ORDER_ANOMALY_FLAG)
        return 0.0, seg_log

    return float(latency_ms), seg_log


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

    K1 now uses ``stiff_x`` / ``stiff_y`` (Trajectory B) from
    ``processed_summary`` as the reference for RMS tremor measurement.
    K2 uses ``path_length_px`` which is pre-computed from Trajectory A.
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
            "k5_segmentation_log":   [],
        }

    processed_strokes = processed_summary.get("processed_strokes", [])

    # --- Group raw strokes by ID ----------------------------------------
    strokes_dict: dict[int, list["StrokePoint"]] = {}
    for pt in raw_strokes:
        strokes_dict.setdefault(pt.id, []).append(pt)
    sorted_stroke_ids = sorted(strokes_dict.keys())

    # --- K1: RMS tremor (raw vs Trajectory B / stiff reference) ---------
    k1_rms_values: list[float] = []
    for stroke in processed_strokes:
        if not stroke.get("eligible_for_kinematics"):
            continue
        try:
            rms = compute_k1_rms(
                stroke["raw_x"],
                stroke["raw_y"],
                stroke["stiff_x"],   # Trajectory B — stiff reference
                stroke["stiff_y"],
                device_dpi,
            )
            if rms is not None:
                k1_rms_values.append(rms)
        except ValueError as exc:
            flags.append(f"K1_ARRAY_MISMATCH: {exc}")

    k1_rms_cm: float | None = (
        float(np.mean(k1_rms_values)) if k1_rms_values else None
    )

    # --- K2: Velocity (path_length already from Trajectory A) -----------
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
    k5_pfhl_ms, seg_log = _compute_k5_pre_first_hand_latency(
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
        "k5_segmentation_log":   seg_log,
    }