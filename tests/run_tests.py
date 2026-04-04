"""
Minimal test runner — executes all test cases without pytest.
Prints PASS / FAIL per test and a final summary.
"""
import sys, traceback

passed = []
failed = []

def run(name, fn):
    try:
        fn()
        passed.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        failed.append(name)
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {e}")

# ── imports ──────────────────────────────────────────────────────────────────
import math, numpy as np
from kinematics import (
    compute_k1_rms, compute_k2_velocity,
    detect_pressure_support, compute_k3_pressure,
    compute_k4_think_time,
    _compute_k5_pre_first_hand_latency,
)
from normalization import get_dynamic_thresholds
from inference import evaluate_k_series, apply_truth_table, classify_risk, EDUCATION_BIAS_WARNING

class SP:  # StrokePoint stub
    def __init__(self, t, x, y, p=0.5, az=0.0, alt=1.57, id=1):
        self.t=t; self.x=x; self.y=y; self.p=p; self.az=az; self.alt=alt; self.id=id

# ── K1 tests ─────────────────────────────────────────────────────────────────
print("\n── K1 ──")

def t_k1_mismatch():
    try:
        compute_k1_rms([1,2,3],[0,0,0],[1,2],[0,0],96.0)
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "mismatch" in str(e).lower()

def t_k1_empty(): assert compute_k1_rms([],[],[],[],96.0) is None
def t_k1_bad_dpi(): assert compute_k1_rms([1],[1],[1],[1],0.0) is None
def t_k1_neg_dpi(): assert compute_k1_rms([1],[1],[1],[1],-1.0) is None
def t_k1_zero_rms():
    c=[float(i) for i in range(10)]
    assert abs(compute_k1_rms(c,c,c,c,96.0)) < 1e-9
def t_k1_known():
    r=compute_k1_rms([0,0],[0,0],[1,1],[1,1],96.0)
    assert abs(r - math.sqrt(2)/(96/2.54)) < 1e-6

for name,fn in [("array_mismatch_raises",t_k1_mismatch),
                ("empty_returns_none",t_k1_empty),
                ("bad_dpi_returns_none",t_k1_bad_dpi),
                ("neg_dpi_returns_none",t_k1_neg_dpi),
                ("perfect_smooth_zero_rms",t_k1_zero_rms),
                ("known_deviation",t_k1_known)]:
    run(f"K1::{name}", fn)

# ── K2 tests ─────────────────────────────────────────────────────────────────
print("\n── K2 ──")
def _ps(path,dur): return [{"eligible_for_kinematics":True,"path_length_px":path,"duration_ms":dur}]

def t_k2_zero_dur(): assert compute_k2_velocity(_ps(100,0),96.0) is None
def t_k2_bad_dpi(): assert compute_k2_velocity(_ps(100,1000),0.0) is None
def t_k2_unit():
    r=compute_k2_velocity(_ps(96.0,1000.0),96.0)
    assert abs(r-2.54)<0.01
def t_k2_excludes_ineligible():
    s=[{"eligible_for_kinematics":False,"path_length_px":999,"duration_ms":500},
       {"eligible_for_kinematics":True, "path_length_px":96, "duration_ms":1000}]
    assert abs(compute_k2_velocity(s,96.0)-2.54)<0.01

for name,fn in [("zero_duration_none",t_k2_zero_dur),("bad_dpi_none",t_k2_bad_dpi),
                ("unit_conversion",t_k2_unit),("excludes_ineligible",t_k2_excludes_ineligible)]:
    run(f"K2::{name}", fn)

# ── K3 tests ─────────────────────────────────────────────────────────────────
print("\n── K3 ──")
def t_k3_constant_not_supported():
    pts=[SP(i,0,0,p=0.5) for i in range(20)]
    assert detect_pressure_support(pts) is False
def t_k3_zero_not_supported():
    pts=[SP(i,0,0,p=0.0) for i in range(10)]
    assert detect_pressure_support(pts) is False
def t_k3_varying_supported():
    pts=[SP(i,0,0,p=p) for i,p in enumerate([0.2,0.5,0.8,0.3,0.9])]
    assert detect_pressure_support(pts) is True
def t_k3_stroke_averages():
    s1=[SP(i,0,0,p=0.2,id=1) for i in range(5)]
    s2=[SP(10+i,0,0,p=0.8,id=2) for i in range(5)]
    r=compute_k3_pressure({1:s1,2:s2},[1,2],True)
    assert abs(r["P_first"]-0.2)<1e-6 and abs(r["P_last"]-0.8)<1e-6
def t_k3_none_when_unsupported():
    s=[SP(i,0,0,id=1) for i in range(5)]
    r=compute_k3_pressure({1:s},[1],False)
    assert r["P_avg"] is None and r["P_first"] is None

for name,fn in [("constant_not_supported",t_k3_constant_not_supported),
                ("zero_not_supported",t_k3_zero_not_supported),
                ("varying_supported",t_k3_varying_supported),
                ("stroke_averages",t_k3_stroke_averages),
                ("none_when_unsupported",t_k3_none_when_unsupported)]:
    run(f"K3::{name}", fn)

# ── K4 tests ─────────────────────────────────────────────────────────────────
print("\n── K4 ──")
def _sd(defs):
    d={}
    for ts,te,sid in defs:
        d[sid]=[SP(ts,0,0,id=sid),SP(te,1,1,id=sid)]
    return d, sorted(d.keys())

def t_k4_gap_300_ignored():
    d,s=_sd([(0,1000,1),(1300,2300,2)])
    r=compute_k4_think_time(d,s,500.0)
    assert r and abs(r["T_think_ms"])<1e-6

def t_k4_gap_500_ignored():
    d,s=_sd([(0,1000,1),(1500,2500,2)])
    r=compute_k4_think_time(d,s,500.0)
    assert r and abs(r["T_think_ms"])<1e-6

def t_k4_gap_600_counted():
    d,s=_sd([(0,1000,1),(1600,2500,2)])
    r=compute_k4_think_time(d,s,500.0)
    assert r and abs(r["T_think_ms"]-600)<1e-6

def t_k4_multiple_gaps():
    d,s=_sd([(0,1000,1),(1300,2300,2),(2900,3500,3),(3700,4500,4),(5300,6000,5)])
    r=compute_k4_think_time(d,s,500.0)
    assert r and abs(r["T_think_ms"]-1400)<1e-6

def t_k4_negative_gap_ignored():
    d,s=_sd([(0,2000,1),(1500,3000,2)])
    r=compute_k4_think_time(d,s,500.0)
    assert r and abs(r["T_think_ms"])<1e-6

def t_k4_total_zero_returns_none():
    d={1:[SP(100,0,0,id=1),SP(100,0,0,id=1)]}
    r=compute_k4_think_time(d,[1],500.0)
    assert r is None

def t_k4_pct_calculation():
    d,s=_sd([(0,1000,1),(1600,2500,2)])
    r=compute_k4_think_time(d,s,500.0)
    assert r and abs(r["pct_think_time"]-24.0)<1e-6

for name,fn in [("gap_300_ignored",t_k4_gap_300_ignored),
                ("gap_500_ignored",t_k4_gap_500_ignored),
                ("gap_600_counted",t_k4_gap_600_counted),
                ("multiple_gaps",t_k4_multiple_gaps),
                ("negative_gap_ignored",t_k4_negative_gap_ignored),
                ("t_total_zero_none",t_k4_total_zero_returns_none),
                ("pct_calculation",t_k4_pct_calculation)]:
    run(f"K4::{name}", fn)

# ── K5 tests ─────────────────────────────────────────────────────────────────
print("\n── K5 ──")
def t_k5_empty_none():
    flags=[]
    assert _compute_k5_pre_first_hand_latency({}, [], flags) is None
    assert "K5_SEGMENTATION_FAILED" in flags

def t_k5_no_hand_none():
    pts=[SP(i*25,300+i,300,id=1) for i in range(20)]
    flags=[]
    r=_compute_k5_pre_first_hand_latency({1:pts},[1],flags)
    assert r is None and "K5_SEGMENTATION_FAILED" in flags

def t_k5_drawing_order_anomaly():
    # Hand (id=3) is drawn first (t=0..40) but has the highest ID.
    # Two digit strokes (id=1, id=2) are symmetric around (200,200) so the
    # bounding-box centre falls at (200,200) and the hand centroid (200,200)
    # is within the threshold radius.  sorted_ids=[1,2,3] so the classifier
    # sees digits before the hand, finds hand at index 2, then computes
    # PFHL = hand.t_start(0) - last_digit.t_end(240) = -240 ms → clamped to 0.
    hand =[SP(i*10, 200, 200, id=3) for i in range(5)]           # t=0..40
    d1   =[SP(100+i*10, 50+i*0.1, 200, id=1) for i in range(5)] # t=100..140
    d2   =[SP(200+i*10, 350-i*0.1, 200, id=2) for i in range(5)]# t=200..240
    flags=[]
    r=_compute_k5_pre_first_hand_latency({1:d1,2:d2,3:hand},[1,2,3],flags)
    assert r==0.0, f"Expected 0.0 got {r}"
    assert "DRAWING_ORDER_ANOMALY" in flags, f"Expected anomaly flag, got {flags}"

for name,fn in [("empty_returns_none",t_k5_empty_none),
                ("no_hand_returns_none",t_k5_no_hand_none),
                ("drawing_order_anomaly",t_k5_drawing_order_anomaly)]:
    run(f"K5::{name}", fn)

# ── Normalization ─────────────────────────────────────────────────────────────
print("\n── Normalization ──")
cases_k2 = [(60,3.00),(70,2.70),(80,2.40),(90,2.10),(100,1.80)]
cases_k4 = [(60,40.0),(69,40.0),(70,43.0),(80,46.0),(90,49.0)]
cases_k5 = [(60,8000.0),(70,9500.0),(80,11000.0),(90,12500.0)]

for age,exp in cases_k2:
    run(f"Norm::K2_age{age}", lambda a=age,e=exp: (lambda t: None if abs(t["K2_velocity_cms"]-e)<1e-5 else (_ for _ in ()).throw(AssertionError(f"{t['K2_velocity_cms']} != {e}")))(get_dynamic_thresholds(a)))

def t_norm_lower_bound():
    assert abs(get_dynamic_thresholds(999)["K2_velocity_cms"]-0.5)<1e-6
run("Norm::lower_bound_age999", t_norm_lower_bound)

for age,exp in cases_k4:
    run(f"Norm::K4_age{age}", lambda a=age,e=exp: (lambda t: None if abs(t["K4_pct_think_time"]-e)<1e-5 else (_ for _ in ()).throw(AssertionError(f"{t['K4_pct_think_time']} != {e}")))(get_dynamic_thresholds(a)))

for age,exp in cases_k5:
    run(f"Norm::K5_age{age}", lambda a=age,e=exp: (lambda t: None if abs(t["K5_pfhl_ms"]-e)<1e-5 else (_ for _ in ()).throw(AssertionError(f"{t['K5_pfhl_ms']} != {e}")))(get_dynamic_thresholds(a)))

# ── Truth table ───────────────────────────────────────────────────────────────
print("\n── Truth Table ──")
tt_cases=[(False,False,False,"C0"),(False,True,False,"C1"),(False,False,True,"C2"),
          (False,True,True,"C3"),(True,False,False,"C4"),(True,False,True,"C5"),
          (True,True,False,"C6"),(True,True,True,"C7")]
for ai,mo,cg,exp in tt_cases:
    run(f"TT::{exp}", lambda a=ai,m=mo,c=cg,e=exp: None if apply_truth_table(a,m,c)==e else (_ for _ in ()).throw(AssertionError(f"{apply_truth_table(a,m,c)} != {e}")))

# ── Education warning ─────────────────────────────────────────────────────────
print("\n── Education Warning ──")
warn_cases=[("C4",6,True),("C5",0,True),("C6",7,True),("C7",5,True),
            ("C4",8,False),("C4",12,False),("C0",6,False),("C3",7,False)]
for cls,edu,expect in warn_cases:
    def _tw(c=cls,e=edu,ex=expect):
        r=classify_risk(c,e)
        has=EDUCATION_BIAS_WARNING in r["warnings"]
        if has!=ex: raise AssertionError(f"class={c} edu={e}: expected {ex} got {has}")
    run(f"EduWarn::{cls}_edu{edu}", _tw)

def t_no_auto_downgrade():
    r=classify_risk("C4",4)
    assert r["risk_level"]=="mild" and r["risk_color"]=="yellow"
run("EduWarn::no_auto_downgrade", t_no_auto_downgrade)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  {len(passed)} passed   {len(failed)} failed   {len(passed)+len(failed)} total")
print(f"{'='*55}")
if failed:
    print("\nFailed tests:")
    for f in failed: print(f"  • {f}")
    sys.exit(1)