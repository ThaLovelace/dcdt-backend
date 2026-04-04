"""
preprocessing.py
----------------
Signal smoothing pipeline for the dCDT backend.

Key guarantee (K1 requirement)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``process_strokes`` returns **both** the original raw coordinates
(``raw_x``, ``raw_y``) and the Savitzky-Golay smoothed coordinates
(``smoothed_x``, ``smoothed_y``) for every stroke.

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

TARGET_WINDOW_MS: float = 50.0   # Desired temporal window length in ms
MIN_WINDOW:       int   = 5      # Absolute floor for window length (must be odd)
POLY_ORDER:       int   = 3      # Savitzky-Golay polynomial order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_adaptive_window(timestamps_ms: np.ndarray) -> int:
    """
    Calculate the time-adaptive Savitzky-Golay window length.

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
        fully interior window (``n >= 2 * window - 2``).  Boundary
        samples are extrapolated and carry higher uncertainty.

    Notes
    -----
    ``dt_s`` is computed from the **median** inter-sample interval so
    that occasional duplicate timestamps from coalesced browser events
    do not corrupt the time axis (spec §3.5.2 Algorithm 3.2).
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

    For every stroke the function returns **both** the original raw
    coordinates and the Savitzky-Golay smoothed coordinates so that
    ``kinematics.compute_k1_rms`` can compute the RMS deviation without
    any information loss.

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
    path_length_px         : float   – Euclidean arc length in pixels.
    raw_x                  : list[float]  – Original X coordinates.
    raw_y                  : list[float]  – Original Y coordinates.
    smoothed_x             : list[float]  – Savitzky-Golay smoothed X.
    smoothed_y             : list[float]  – Savitzky-Golay smoothed Y.
                             ``len(raw_x) == len(smoothed_x)`` is always True.
    pressure_values        : list[float]  – Raw pressure at every point.
    eligible_for_timing    : bool  – Tier 1: t_end > t_start.
    eligible_for_smoothing : bool  – Tier 2: n >= adaptive window.
    eligible_for_kinematics: bool  – Tier 3: Tier 2 AND path_length > 0.
    jerk_magnitude         : float | None
    is_jerk_reliable       : bool

    Array-length guarantee
    ----------------------
    ``savgol_filter`` never changes the length of its input array;
    therefore ``len(raw_x) == len(smoothed_x)`` holds by construction.
    No down-sampling or point deletion is performed at any stage.
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
            # Cannot form a stroke from a single point
            continue

        t_duration = float(t_arr[-1] - t_arr[0])

        # --- Tier 1: Timing eligibility ---------------------------------
        eligible_for_timing = t_duration > 0

        # --- Adaptive window -------------------------------------------
        window = compute_adaptive_window(t_arr)

        # --- Tier 2: Smoothing eligibility ------------------------------
        eligible_for_smoothing = n_points >= window

        # --- Arc length (pixels) ----------------------------------------
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        path_length_px = float(np.sum(np.sqrt(dx**2 + dy**2)))

        # --- Tier 3: Kinematic eligibility ------------------------------
        eligible_for_kinematics = eligible_for_smoothing and path_length_px > 0

        # --- Smoothing (deriv=0 = position only) ------------------------
        # Using deriv=0 here preserves point count while reducing sensor
        # noise.  The jerk derivative (deriv=3) is computed separately in
        # compute_jerk_signal to avoid compounding approximation errors.
        if eligible_for_smoothing:
            smoothed_x = savgol_filter(
                x_arr, window, POLY_ORDER, deriv=0, mode="interp"
            ).tolist()
            smoothed_y = savgol_filter(
                y_arr, window, POLY_ORDER, deriv=0, mode="interp"
            ).tolist()
        else:
            # Fall back to raw coordinates; the 1-to-1 mapping is preserved.
            smoothed_x = x_arr.tolist()
            smoothed_y = y_arr.tolist()

        # savgol_filter guarantees output length == input length, so
        # len(raw_x) == len(smoothed_x) is always True.

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
                "path_length_px":          path_length_px,
                # Raw coordinates — never modified after capture
                "raw_x":                   x_arr.tolist(),
                "raw_y":                   y_arr.tolist(),
                # Smoothed coordinates — same length as raw (1-to-1 mapping)
                "smoothed_x":              smoothed_x,
                "smoothed_y":              smoothed_y,
                "pressure_values":         p_arr,
                "eligible_for_timing":     bool(eligible_for_timing),
                "eligible_for_smoothing":  bool(eligible_for_smoothing),
                "eligible_for_kinematics": bool(eligible_for_kinematics),
                "jerk_magnitude":          jerk_magnitude,
                "is_jerk_reliable":        is_jerk_reliable,
            }
        )

    return {"processed_strokes": processed_results}