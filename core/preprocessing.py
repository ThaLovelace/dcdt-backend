"""
preprocessing.py
----------------
Signal smoothing pipeline for the dCDT backend.

Key guarantee (K1 requirement)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``process_strokes`` returns **both** the original raw coordinates
(``raw_x``, ``raw_y``) and two smoothed trajectories:

* ``smoothed_x`` / ``smoothed_y``  — Trajectory A (50 ms window, poly=3)
  Used for K2 velocity calculation to remove tremor inflation.
* ``stiff_x``    / ``stiff_y``     — Trajectory B (350 ms window, poly=2)
  Used as the K1 tremor reference; "stiff" enough to not follow tremor oscillations.

Because ``scipy.signal.savgol_filter`` always returns an array whose
length is identical to its input, the 1-to-1 point mapping between
raw and smoothed arrays is structurally guaranteed — no down-sampling
or point deletion is ever performed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.schemas import StrokePoint

# ---------------------------------------------------------------------------
# Constants (spec §3.5.2)
# ---------------------------------------------------------------------------

TARGET_WINDOW_MS: float = 50.0    # Trajectory A: target temporal window (ms)
MIN_WINDOW:       int   = 5       # Absolute floor for window length (must be odd)
POLY_ORDER:       int   = 3       # Trajectory A: Savitzky-Golay polynomial order

# Trajectory B (Stiff Reference for K1 tremor measurement — spec §3.5.4.1)
TARGET_STIFF_WINDOW_MS: float = 350.0  # Wide enough to span tremor cycles (4–8 Hz)
K1_STIFF_MIN_WINDOW:    int   = 21     # Minimum samples for reliable stiff reference
STIFF_POLY_ORDER:       int   = 2      # Low order to prevent following tremor curves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_adaptive_window(timestamps_ms: np.ndarray) -> int:
    """
    Calculate the time-adaptive Savitzky-Golay window length for Trajectory A.

    The window is chosen so that it spans approximately
    ``TARGET_WINDOW_MS`` (50 ms) of signal regardless of the device
    sampling rate, matching Table 3.4 of the clinical specification.

    Parameters
    ----------
    timestamps_ms:
        1-D array of timestamps (in milliseconds) for a single stroke,
        sorted in ascending order.

    Returns
    -------
    int
        Odd window length in the range [MIN_WINDOW, …].

    Notes
    -----
    * If fewer than 2 timestamps are available, ``MIN_WINDOW`` is returned.
    * If the median inter-sample interval is ≤ 0 (duplicate timestamps),
      a fallback of 11 is returned (assumes 200 Hz).
    """
    if len(timestamps_ms) < 2:
        return MIN_WINDOW

    dt_median = float(np.median(np.diff(timestamps_ms)))

    if dt_median <= 0:
        return 11  # Fallback: assume 200 Hz device

    raw_window = int(round(TARGET_WINDOW_MS / dt_median))
    window = max(raw_window, MIN_WINDOW)

    # savgol_filter requires an odd window length
    return window if window % 2 == 1 else window + 1


def _build_stiff_reference(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
) -> tuple[list[float], list[float]]:
    """
    Build Trajectory B: the "stiff" reference for K1 tremor measurement.

    Uses a wide 350 ms window and low polynomial order (2) so the
    reference line follows the intended drawing direction without
    bending into tremor oscillations (spec §3.5.4.1, Dual-Trajectory).

    Parameters
    ----------
    x_arr, y_arr:
        Raw pixel coordinates.
    t_arr:
        Timestamps in milliseconds.

    Returns
    -------
    (stiff_x, stiff_y) : tuple[list[float], list[float]]
        Stiff reference coordinates.  Same length as input — 1-to-1
        mapping is preserved by savgol_filter.
    """
    dt_median = float(np.median(np.diff(t_arr))) if len(t_arr) >= 2 else 0.0

    if dt_median <= 0:
        # Fallback: assume 200 Hz (5 ms per sample)
        dt_median = 5.0

    raw_window = int(round(TARGET_STIFF_WINDOW_MS / dt_median))
    dynamic_window = max(raw_window, K1_STIFF_MIN_WINDOW)

    # Enforce odd window length
    if dynamic_window % 2 == 0:
        dynamic_window += 1

    # Enforce window does not exceed data length (savgol requirement)
    n = len(x_arr)
    if dynamic_window >= n:
        # Not enough points for stiff filter; fall back to Trajectory A values
        return x_arr.tolist(), y_arr.tolist()

    stiff_x = savgol_filter(
        x_arr, window_length=dynamic_window, polyorder=STIFF_POLY_ORDER,
        deriv=0, mode="interp"
    ).tolist()
    stiff_y = savgol_filter(
        y_arr, window_length=dynamic_window, polyorder=STIFF_POLY_ORDER,
        deriv=0, mode="interp"
    ).tolist()

    return stiff_x, stiff_y


def compute_jerk_signal(
    x:      np.ndarray,
    y:      np.ndarray,
    t_ms:   np.ndarray,
    window: int,
) -> tuple[np.ndarray, bool]:
    """
    Compute the 3rd-derivative (jerk) magnitude via a single Savitzky-Golay pass.

    Using ``deriv=3`` inside ``savgol_filter`` obtains the analytical
    derivative of the fitted polynomial, avoiding the noise amplification
    caused by repeated finite-difference steps (spec §3.5.2).

    Parameters
    ----------
    x, y:
        Spatial coordinates of the stroke in pixels.
    t_ms:
        Timestamps in milliseconds.
    window:
        Odd window length from ``compute_adaptive_window``.

    Returns
    -------
    j_magnitude : np.ndarray
        Per-sample jerk magnitude (pixels / s³).
    is_reliable : bool
        True when the stroke has enough points to fill at least one
        fully interior window (``n >= 2 * window - 2``).
    """
    dt_median = float(np.median(np.diff(t_ms)))
    dt_s = (dt_median / 1000.0) if dt_median > 0 else (1.0 / 200.0)

    is_reliable = bool(len(x) >= (2 * window - 2))

    jx = savgol_filter(x, window, POLY_ORDER, deriv=3, delta=dt_s, mode="interp")
    jy = savgol_filter(y, window, POLY_ORDER, deriv=3, delta=dt_s, mode="interp")

    return np.sqrt(jx**2 + jy**2), is_reliable


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_strokes(payload_strokes: list["StrokePoint"]) -> dict:
    """
    Pre-processing pipeline enforcing the Three-tier Eligibility rules
    (spec §3.5.2, Table 3.5).

    For every stroke the function returns the original raw coordinates,
    Trajectory A (smoothed for K2 velocity), and Trajectory B (stiff
    reference for K1 tremor) so that downstream kinematics modules
    receive all necessary data without re-computing filters.

    Parameters
    ----------
    payload_strokes:
        All ``StrokePoint`` objects from the ``AnalysisRequest``.

    Returns
    -------
    dict
        ``{"processed_strokes": [<stroke_summary>, …]}``

    Stroke summary keys
    -------------------
    stroke_id              : int
    point_count            : int
    duration_ms            : float
    path_length_px         : float
        Euclidean arc length computed from **Trajectory A** (smoothed_x/y)
        to prevent tremor inflation in K2 velocity (spec §3.5.4.2).
    raw_x / raw_y          : list[float]  — Original coordinates.
    smoothed_x / smoothed_y: list[float]  — Trajectory A (50 ms, poly=3).
    stiff_x    / stiff_y   : list[float]  — Trajectory B (350 ms, poly=2).
                             All three share the same length (1-to-1 mapping).
    pressure_values        : list[float]
    eligible_for_timing    : bool  — Tier 1
    eligible_for_smoothing : bool  — Tier 2
    eligible_for_kinematics: bool  — Tier 3
    jerk_magnitude         : float | None
    is_jerk_reliable       : bool

    Array-length guarantee
    ----------------------
    ``savgol_filter`` never changes the length of its input array;
    therefore ``len(raw_x) == len(smoothed_x) == len(stiff_x)`` holds
    by construction.
    """
    # Group points by stroke_id (preserving insertion order)
    strokes_data: dict[int, dict] = {}
    for pt in payload_strokes:
        if pt.id not in strokes_data:
            strokes_data[pt.id] = {"t": [], "x": [], "y": [], "p": []}
        strokes_data[pt.id]["t"].append(pt.t)
        strokes_data[pt.id]["x"].append(pt.x)
        strokes_data[pt.id]["y"].append(pt.y)
        strokes_data[pt.id]["p"].append(pt.p)

    processed_results: list[dict] = []

    for stroke_id, data in strokes_data.items():
        t_arr = np.array(data["t"], dtype=float)
        x_arr = np.array(data["x"], dtype=float)
        y_arr = np.array(data["y"], dtype=float)
        p_arr = data["p"]

        n_points = len(t_arr)
        if n_points < 2:
            continue

        t_duration = float(t_arr[-1] - t_arr[0])

        # --- Tier 1: Timing eligibility ---------------------------------
        eligible_for_timing = t_duration > 0

        # --- Adaptive window (Trajectory A) ----------------------------
        window = compute_adaptive_window(t_arr)

        # --- Tier 2: Smoothing eligibility ------------------------------
        eligible_for_smoothing = n_points >= window

        # --- Trajectory A: balanced smoothing (deriv=0) -----------------
        if eligible_for_smoothing:
            smoothed_x = savgol_filter(
                x_arr, window, POLY_ORDER, deriv=0, mode="interp"
            ).tolist()
            smoothed_y = savgol_filter(
                y_arr, window, POLY_ORDER, deriv=0, mode="interp"
            ).tolist()
        else:
            # Fall back to raw coordinates; 1-to-1 mapping is preserved.
            smoothed_x = x_arr.tolist()
            smoothed_y = y_arr.tolist()

        # --- Arc length from Trajectory A (not raw) ---------------------
        # Using smoothed coordinates prevents tremor inflation in K2.
        sm_x = np.array(smoothed_x)
        sm_y = np.array(smoothed_y)
        dx_sm = np.diff(sm_x)
        dy_sm = np.diff(sm_y)
        path_length_px = float(np.sum(np.sqrt(dx_sm**2 + dy_sm**2)))

        # --- Tier 3: Kinematic eligibility ------------------------------
        eligible_for_kinematics = eligible_for_smoothing and path_length_px > 0

        # --- Trajectory B: stiff reference (for K1 tremor) -------------
        if eligible_for_kinematics:
            stiff_x, stiff_y = _build_stiff_reference(x_arr, y_arr, t_arr)
        else:
            # Not enough data for K1; still preserve 1-to-1 mapping.
            stiff_x = x_arr.tolist()
            stiff_y = y_arr.tolist()

        # --- Jerk (for internal use / future features) ------------------
        jerk_magnitude: float | None = None
        is_jerk_reliable: bool = False

        if eligible_for_kinematics:
            j_mag, is_reliable = compute_jerk_signal(x_arr, y_arr, t_arr, window)
            jerk_magnitude = float(np.mean(j_mag))
            is_jerk_reliable = is_reliable

        processed_results.append(
            {
                "stroke_id":               stroke_id,
                "point_count":             int(n_points),
                "duration_ms":             t_duration,
                # Path length from Trajectory A — prevents tremor inflation (K2)
                "path_length_px":          path_length_px,
                # Raw coordinates — never modified after capture
                "raw_x":                   x_arr.tolist(),
                "raw_y":                   y_arr.tolist(),
                # Trajectory A — balanced smoothing for K2 velocity
                "smoothed_x":              smoothed_x,
                "smoothed_y":              smoothed_y,
                # Trajectory B — stiff reference for K1 tremor measurement
                "stiff_x":                 stiff_x,
                "stiff_y":                 stiff_y,
                "pressure_values":         p_arr,
                "eligible_for_timing":     bool(eligible_for_timing),
                "eligible_for_smoothing":  bool(eligible_for_smoothing),
                "eligible_for_kinematics": bool(eligible_for_kinematics),
                "jerk_magnitude":          jerk_magnitude,
                "is_jerk_reliable":        is_jerk_reliable,
            }
        )

    return {"processed_strokes": processed_results}