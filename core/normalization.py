"""
normalization.py
----------------
Age-adjusted dynamic threshold computation for the K-Series rules.

Formulae are taken directly from spec §3.5.4.4 and calibrated against
Müller et al. (2019) and Souillard-Mandar et al. (2016).  Because the
source coefficients were derived from Western populations, the returned
values should be treated as seed values pending local Thai-population
calibration (spec §3.5.4.3, note 3).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Fixed (age-independent) thresholds
# ---------------------------------------------------------------------------

K1_RMS_THRESHOLD_CM:  float = 0.05   # Smits et al. (2014); Zham et al. (2019)
K3_PRESSURE_AVG:      float = 0.25   # Dion et al. (2020)
K3_DECREMENT_RATIO:   float = 0.70   # P_last / P_first; Zham et al. (2019)
K4_NOISE_FILTER_MS:   float = 500.0  # Souillard-Mandar et al. (2016)

# ---------------------------------------------------------------------------
# Age-adjusted threshold formulae
# ---------------------------------------------------------------------------

def _threshold_k2(age: int) -> float:
    """
    Minimum drawing velocity threshold for K2 (Bradykinesia) in cm/s.

    Formula (spec §3.5.4.4)::

        max(0.5,  3.0 − (0.03 × max(0, age − 60)))

    Velocity decreases linearly with age **only past age 60** (age-60 decay),
    reflecting population-level norms for older adults.  The lower bound of
    **0.5 cm/s** prevents over-triggering at extreme ages.

    BUG-002 FIX: Previous formula used max(0.3, 1.2-(0.005*age)) which had
    a wrong baseline (0.3 vs 0.5), wrong intercept (1.2 vs 3.0), wrong slope
    (0.005 vs 0.03), and wrong age anchor (age vs age-60).

    References: Müller et al. (2019).
    """
    age_above_60 = max(0, age - 60)
    raw = 3.0 - (0.03 * age_above_60)
    return max(0.5, raw)


def _threshold_k4(age: int) -> float:
    """
    %ThinkTime threshold for K4 (Hesitation) as a percentage.

    Formula (spec §3.5.4.4)::

        40 + (3.0 × floor((age − 60) / 10))

    Increases by 3.0 % per decade past age 60; clamped to a minimum of 40 %.
    Below age 60 the step term is zero so the threshold is a flat 40 %.

    BUG-003 FIX: Previous formula used 25.0 + (0.2 * max(age, 30)) which had
    wrong baseline (25 vs 40), wrong slope (0.2/year vs 3.0/decade), wrong
    age anchor (age/30 vs decade steps past 60), and produced values far above
    the clinical spec for middle-aged patients.

    References: Souillard-Mandar et al. (2016).
    """
    decades_past_60 = max(0, math.floor((age - 60) / 10))
    return 40.0 + (3.0 * decades_past_60)


def _threshold_k5(age: int) -> float:
    """
    Pre-First Hand Latency threshold for K5 in milliseconds.

    Formula (spec §3.5.4.4)::

        8000 + (1500 × max(0, floor((age − 60) / 10)))

    Increases by 1.5 s per decade past age 60.

    References: Penney et al. (2011a) via Davis et al. (2014);
                Dion et al. (2020); Wiggins et al. (2021).
    """
    decades = max(0, math.floor((age - 60) / 10))
    return 8000.0 + (1500.0 * decades)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dynamic_thresholds(age: int) -> dict:
    """
    Return the complete threshold dictionary for a given patient age.

    Age-independent thresholds are included here so that ``inference.py``
    has a single source of truth.

    Parameters
    ----------
    age:
        Patient age in years.  Values ≤ 0 are treated as a young-adult
        healthy-control default (equivalent to age 30) to handle
        data-entry errors gracefully.

    Returns
    -------
    dict
        All thresholds required by ``evaluate_k_series``::

            {
                "K1_rms_threshold_cm" : float,  # 0.05 cm (fixed)
                "K2_velocity_cms"     : float,  # age-adjusted
                "K3_pressure_avg"     : float,  # 0.25 (fixed)
                "K3_decrement_ratio"  : float,  # 0.70 (fixed)
                "K4_pct_think_time"   : float,  # age-adjusted
                "K4_noise_filter_ms"  : float,  # 500 ms (fixed)
                "K5_pfhl_ms"          : float,  # age-adjusted
            }
    """
    if age <= 0:
        effective_age = 30
    else:
        effective_age = age

    return {
        "K1_rms_threshold_cm": K1_RMS_THRESHOLD_CM,
        "K2_velocity_cms":     _threshold_k2(effective_age),
        "K3_pressure_avg":     K3_PRESSURE_AVG,
        "K3_decrement_ratio":  K3_DECREMENT_RATIO,
        "K4_pct_think_time":   _threshold_k4(effective_age),
        "K4_noise_filter_ms":  K4_NOISE_FILTER_MS,
        "K5_pfhl_ms":          _threshold_k5(effective_age),
    }
