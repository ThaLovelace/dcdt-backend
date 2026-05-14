"""
tests/test_known_answer.py
--------------------------
Known-Answer Tests (KAT) สำหรับยืนยันว่า Logic K1-K5 และ Threshold
สอดคล้องกับสมการที่ออกแบบไว้ในรายงานบทที่ 4.3 อย่างแม่นยำ

วิธีรัน:
    cd <root โปรเจค>
    pytest tests/test_known_answer.py -v

หรือถ้าอยู่ใน folder tests อยู่แล้ว:
    pytest test_known_answer.py -v
"""

import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Stub StrokePoint (เหมือนใน test_dcdt.py เดิม)
# ---------------------------------------------------------------------------

class SP:
    def __init__(self, t, x, y, p=0.5, az=0.0, alt=1.57, id=1):
        self.t = t; self.x = x; self.y = y
        self.p = p; self.az = az; self.alt = alt; self.id = id


# ---------------------------------------------------------------------------
# KAT-K1: Tremor RMS
# ---------------------------------------------------------------------------

class TestKATK1:
    """
    K1 วัด RMS(raw − Trajectory B) แล้วแปลงเป็น cm
    สูตร: RMS_cm = RMS_px / (device_dpi / 2.54)
    """

    def _import(self):
        from core.kinematics import compute_k1_rms
        return compute_k1_rms

    def test_KAT_K1_01_sine_wave_tremor(self):
        """
        KAT-K1-01
        สร้าง sine wave 6 Hz amplitude 5px บน DPI=96
        stiff reference = เส้นตรง (ค่า mean ของ raw)
        RMS_px ควรใกล้เคียง 5 / sqrt(2) ≈ 3.536 px  (RMS ของ sine)
        RMS_cm = 3.536 / (96/2.54) ≈ 0.0936 cm
        ค่าที่คาดหวัง: อยู่ในช่วง 0.08 – 0.11 cm
        """
        compute_k1_rms = self._import()
        n = 200
        amplitude = 5.0
        raw_x = [float(i) + amplitude * math.sin(2 * math.pi * 6 * i / 200)
                 for i in range(n)]
        raw_y = [100.0] * n
        # stiff = เส้นตรงเรียบ (ไม่มีการสั่น)
        stiff_x = [float(i) for i in range(n)]
        stiff_y = [100.0] * n

        result = compute_k1_rms(raw_x, raw_y, stiff_x, stiff_y, device_dpi=96.0)

        assert result is not None
        # RMS ของ sine wave = amplitude/sqrt(2)
        expected_rms_px = amplitude / math.sqrt(2)
        expected_cm = expected_rms_px / (96.0 / 2.54)
        assert abs(result - expected_cm) < 0.01, (
            f"KAT-K1-01: expected ≈{expected_cm:.4f} cm, got {result:.4f} cm"
        )

    def test_KAT_K1_02_no_tremor_zero_rms(self):
        """
        KAT-K1-02
        เส้นตรงสมบูรณ์ไม่มีการสั่น (raw = stiff)
        K1 ต้องเป็น 0.000 cm
        """
        compute_k1_rms = self._import()
        coords = [float(i) for i in range(20)]
        result = compute_k1_rms(coords, coords, coords, coords, device_dpi=96.0)
        assert result == pytest.approx(0.0, abs=1e-9), (
            f"KAT-K1-02: expected 0.0, got {result}"
        )

    def test_KAT_K1_03_constant_offset_1px(self):
        """
        KAT-K1-03
        raw − stiff = 1px ทุกจุด (offset คงที่ในแกน x)
        RMS_px = 1.0
        RMS_cm = 1.0 / (96/2.54) ≈ 0.02646 cm
        """
        compute_k1_rms = self._import()
        n = 20
        raw_x   = [float(i) + 1.0 for i in range(n)]
        stiff_x = [float(i)        for i in range(n)]
        raw_y   = [0.0] * n
        stiff_y = [0.0] * n

        result = compute_k1_rms(raw_x, raw_y, stiff_x, stiff_y, device_dpi=96.0)

        expected = 1.0 / (96.0 / 2.54)
        assert result == pytest.approx(expected, rel=1e-4), (
            f"KAT-K1-03: expected {expected:.5f} cm, got {result:.5f} cm"
        )


# ---------------------------------------------------------------------------
# KAT-K2: Velocity
# ---------------------------------------------------------------------------

class TestKATK2:
    """
    K2 = total_path_cm / total_time_s
    path_length_px มาจาก Trajectory A (คำนวณใน preprocessing แล้ว)
    """

    def _import(self):
        from core.kinematics import compute_k2_velocity
        return compute_k2_velocity

    def _stroke(self, path_px, duration_ms):
        return [{
            "eligible_for_kinematics": True,
            "path_length_px": path_px,
            "duration_ms": duration_ms,
        }]

    def test_KAT_K2_01_straight_line_unit_conversion(self):
        """
        KAT-K2-01
        path_length = 96px, duration = 1000ms, DPI = 96
        px_per_cm  = 96/2.54 ≈ 37.795
        length_cm  = 96/37.795 ≈ 2.54 cm
        velocity   = 2.54/1.0 = 2.54 cm/s
        """
        compute_k2_velocity = self._import()
        result = compute_k2_velocity(self._stroke(96.0, 1000.0), device_dpi=96.0)
        assert result == pytest.approx(2.54, rel=1e-3), (
            f"KAT-K2-01: expected 2.54 cm/s, got {result}"
        )

    def test_KAT_K2_02_tremor_inflation_prevented(self):
        """
        KAT-K2-02
        path ที่คำนวณจาก Trajectory A (smoothed) ต้องสั้นกว่า path จาก raw
        เพื่อป้องกัน Tremor Inflation
        ทดสอบโดยเปรียบเทียบ path_length ของ smoothed vs raw จริงๆ ใน preprocessing
        """
        from core.preprocessing import process_strokes

        # สร้าง stroke ที่มี tremor 8px ที่ 6 Hz
        pts = []
        for i in range(100):
            tremor = 8.0 * math.sin(2 * math.pi * 6 * i * 0.005)
            pts.append(SP(t=i * 5.0, x=float(i) * 2.0 + tremor,
                          y=200.0 + tremor, p=0.5, id=1))

        result = process_strokes(pts)
        stroke = result["processed_strokes"][0]

        # คำนวณ path จาก raw เพื่อเปรียบเทียบ
        raw_x = np.array(stroke["raw_x"])
        raw_y = np.array(stroke["raw_y"])
        raw_path = float(np.sum(np.sqrt(np.diff(raw_x)**2 + np.diff(raw_y)**2)))

        smoothed_path = stroke["path_length_px"]

        assert smoothed_path < raw_path, (
            f"KAT-K2-02: smoothed path ({smoothed_path:.2f}px) "
            f"ต้องน้อยกว่า raw path ({raw_path:.2f}px)"
        )

    def test_KAT_K2_03_zero_duration_returns_none(self):
        """
        KAT-K2-03
        duration = 0ms → ต้องคืนค่า None (ป้องกัน division by zero)
        """
        compute_k2_velocity = self._import()
        result = compute_k2_velocity(self._stroke(100.0, 0.0), device_dpi=96.0)
        assert result is None, f"KAT-K2-03: expected None, got {result}"


# ---------------------------------------------------------------------------
# KAT-K3: Pressure
# ---------------------------------------------------------------------------

class TestKATK3:

    def _import(self):
        from core.kinematics import detect_pressure_support, compute_k3_pressure
        return detect_pressure_support, compute_k3_pressure

    def test_KAT_K3_01_constant_pressure_not_supported(self):
        """
        KAT-K3-01
        pressure ค่าคงที่ทุกจุด (std < 0.01)
        detect_pressure_support ต้องคืน False
        """
        detect_pressure_support, _ = self._import()
        pts = [SP(t=i, x=0, y=0, p=0.5) for i in range(20)]
        assert detect_pressure_support(pts) is False, "KAT-K3-01: expected False"

    def test_KAT_K3_02_decrement_ratio_calculation(self):
        """
        KAT-K3-02
        stroke แรก: pressure ทุกจุด = 0.8  → P_first = 0.8
        stroke สุดท้าย: pressure ทุกจุด = 0.4 → P_last = 0.4
        decrement = P_last / P_first = 0.4 / 0.8 = 0.500
        """
        _, compute_k3_pressure = self._import()
        s1 = [SP(t=i,    x=0, y=0, p=0.8, id=1) for i in range(5)]
        s2 = [SP(t=10+i, x=0, y=0, p=0.4, id=2) for i in range(5)]
        r = compute_k3_pressure({1: s1, 2: s2}, [1, 2], pressure_supported=True)

        decrement = r["P_last"] / r["P_first"]
        assert decrement == pytest.approx(0.5, rel=1e-6), (
            f"KAT-K3-02: expected 0.500, got {decrement:.4f}"
        )


# ---------------------------------------------------------------------------
# KAT-K4: ThinkTime
# ---------------------------------------------------------------------------

class TestKATK4:

    def _import(self):
        from core.kinematics import compute_k4_think_time
        return compute_k4_think_time

    def _build(self, defs):
        d = {}
        for t_start, t_end, sid in defs:
            d[sid] = [SP(t=t_start, x=0, y=0, id=sid),
                      SP(t=t_end,   x=1, y=1, id=sid)]
        return d, sorted(d.keys())

    def test_KAT_K4_01_gap_499ms_ignored(self):
        """
        KAT-K4-01
        gap = 499ms < 500ms → ไม่นับเป็น think time
        T_think ต้องเป็น 0.0 ms
        """
        fn = self._import()
        d, s = self._build([(0, 1000, 1), (1499, 2499, 2)])
        r = fn(d, s, t_noise_ms=500.0)
        assert r is not None
        assert r["T_think_ms"] == pytest.approx(0.0), (
            f"KAT-K4-01: expected T_think=0.0, got {r['T_think_ms']}"
        )

    def test_KAT_K4_02_gap_exactly_500ms_ignored(self):
        """
        KAT-K4-02
        gap = 500ms = threshold → ต้องไม่นับ (strictly greater than)
        T_think ต้องเป็น 0.0 ms
        """
        fn = self._import()
        d, s = self._build([(0, 1000, 1), (1500, 2500, 2)])
        r = fn(d, s, t_noise_ms=500.0)
        assert r is not None
        assert r["T_think_ms"] == pytest.approx(0.0), (
            f"KAT-K4-02: expected T_think=0.0, got {r['T_think_ms']}"
        )

    def test_KAT_K4_03_pct_think_time_formula(self):
        """
        KAT-K4-03
        T_ink   = 1000 + 900 = 1900 ms
        T_think = 600 ms (gap > 500ms)
        T_total = 2500 ms
        %ThinkTime = (600/2500) × 100 = 24.0%
        """
        fn = self._import()
        d, s = self._build([(0, 1000, 1), (1600, 2500, 2)])
        r = fn(d, s, t_noise_ms=500.0)
        assert r is not None
        assert r["T_think_ms"]     == pytest.approx(600.0), \
            f"KAT-K4-03: T_think expected 600.0, got {r['T_think_ms']}"
        assert r["T_ink_ms"]       == pytest.approx(1900.0), \
            f"KAT-K4-03: T_ink expected 1900.0, got {r['T_ink_ms']}"
        assert r["T_total_ms"]     == pytest.approx(2500.0), \
            f"KAT-K4-03: T_total expected 2500.0, got {r['T_total_ms']}"
        assert r["pct_think_time"] == pytest.approx(24.0, rel=1e-6), \
            f"KAT-K4-03: %ThinkTime expected 24.0%, got {r['pct_think_time']}"


# ---------------------------------------------------------------------------
# KAT-K5: Pre-First Hand Latency
# ---------------------------------------------------------------------------

class TestKATK5:

    def _import(self):
        from core.kinematics import _compute_k5_pre_first_hand_latency
        return _compute_k5_pre_first_hand_latency

    def test_KAT_K5_01_no_hand_returns_none(self):
        """
        KAT-K5-01
        ไม่มีเส้นที่จำแนกเป็นเข็ม (ทุกเส้นอยู่ขอบ)
        ต้องคืน None และแนบ flag K5_SEGMENTATION_FAILED
        """
        fn = self._import()
        # วางเส้นไว้มุมซ้ายบน — ไกลจากศูนย์กลาง canvas มาก
        pts = [SP(t=i*10, x=10+i, y=10, id=1) for i in range(20)]
        flags = []
        result = fn({1: pts}, [1], flags)
        assert result is None, f"KAT-K5-01: expected None, got {result}"
        assert "K5_SEGMENTATION_FAILED" in flags, \
            f"KAT-K5-01: K5_SEGMENTATION_FAILED ไม่อยู่ใน flags={flags}"

    def test_KAT_K5_02_drawing_order_anomaly(self):
        """
        KAT-K5-02
        เข็มวาดก่อน digit (latency ติดลบ)
        ต้องคืน 0.0 และแนบ flag DRAWING_ORDER_ANOMALY
        """
        fn = self._import()
        # เข็ม id=3 วาดก่อน (t=0-40) อยู่ตรงกลาง canvas
        hand = [SP(t=i*10, x=200, y=200, id=3) for i in range(5)]
        # digit ทั้งสองอยู่ขอบ
        d1 = [SP(t=100+i*10, x=50,  y=200, id=1) for i in range(5)]
        d2 = [SP(t=200+i*10, x=350, y=200, id=2) for i in range(5)]

        flags = []
        result = fn({1: d1, 2: d2, 3: hand}, [1, 2, 3], flags)
        assert result == pytest.approx(0.0), \
            f"KAT-K5-02: expected 0.0, got {result}"
        assert "DRAWING_ORDER_ANOMALY" in flags, \
            f"KAT-K5-02: DRAWING_ORDER_ANOMALY ไม่อยู่ใน flags={flags}"


# ---------------------------------------------------------------------------
# KAT-TH: Threshold Formulae
# ---------------------------------------------------------------------------

class TestKATThreshold:
    """
    ทดสอบสูตร Threshold ทุกตัวตามที่กำหนดในรายงาน 4.3
    """

    def _import(self):
        from core.normalization import get_dynamic_thresholds
        return get_dynamic_thresholds

    @pytest.mark.parametrize("age, expected", [
        (60,  0.900),   # 1.2 - 0.005×60 = 0.900
        (70,  0.850),   # 1.2 - 0.005×70 = 0.850
        (80,  0.800),   # 1.2 - 0.005×80 = 0.800
    ])
    def test_KAT_TH_01_02_K2_threshold_by_age(self, age, expected):
        """
        KAT-TH-01, KAT-TH-02
        สูตร K2: max(0.3, 1.2 − (0.005 × age))
        """
        t = self._import()(age)
        assert t["K2_velocity_cms"] == pytest.approx(expected, abs=1e-6), (
            f"KAT-TH K2 age={age}: expected {expected}, got {t['K2_velocity_cms']}"
        )

    def test_KAT_TH_03_K2_lower_bound(self):
        """
        KAT-TH-03
        age=999 → 1.2 - (0.005×999) = ติดลบ → lower bound = 0.300
        """
        t = self._import()(999)
        assert t["K2_velocity_cms"] == pytest.approx(0.300, abs=1e-6), (
            f"KAT-TH-03: expected 0.300, got {t['K2_velocity_cms']}"
        )

    @pytest.mark.parametrize("age, expected", [
        (60, 37.0),    # 25.0 + 0.2×60 = 37.0
        (70, 39.0),    # 25.0 + 0.2×70 = 39.0
        (30, 31.0),    # 25.0 + 0.2×30 = 31.0 (effective_age = max(age,30))
        (0,  31.0),    # age=0 → effective_age=30 → 31.0
    ])
    def test_KAT_TH_04_K4_threshold_by_age(self, age, expected):
        """
        KAT-TH-04
        สูตร K4: 25.0 + (0.2 × max(age, 30))
        """
        t = self._import()(age)
        assert t["K4_pct_think_time"] == pytest.approx(expected, abs=1e-6), (
            f"KAT-TH K4 age={age}: expected {expected}, got {t['K4_pct_think_time']}"
        )

    @pytest.mark.parametrize("age, expected", [
        (60,  8000.0),   # 8000 + 1500×0
        (70,  9500.0),   # 8000 + 1500×1
        (80, 11000.0),   # 8000 + 1500×2
        (90, 12500.0),   # 8000 + 1500×3
    ])
    def test_KAT_TH_05_K5_threshold_by_age(self, age, expected):
        """
        KAT-TH-05
        สูตร K5: 8000 + (1500 × max(0, floor((age−60)/10)))
        """
        t = self._import()(age)
        assert t["K5_pfhl_ms"] == pytest.approx(expected, abs=1e-6), (
            f"KAT-TH K5 age={age}: expected {expected}, got {t['K5_pfhl_ms']}"
        )


# ---------------------------------------------------------------------------
# KAT-TRAJ: ยืนยัน Dual-Trajectory Architecture
# ---------------------------------------------------------------------------

class TestKATTrajectory:
    """
    ทดสอบว่า preprocessing สร้าง Trajectory B จริง
    และ Trajectory B แข็งกว่า Trajectory A จริงๆ
    """

    def test_KAT_TRAJ_01_stiff_fields_exist(self):
        """
        KAT-TRAJ-01
        ผลลัพธ์จาก process_strokes ต้องมี stiff_x และ stiff_y
        """
        from core.preprocessing import process_strokes
        pts = [SP(t=i*5.0, x=float(i)*2.0, y=100.0, p=0.5, id=1)
               for i in range(30)]
        result = process_strokes(pts)
        stroke = result["processed_strokes"][0]
        assert "stiff_x" in stroke, "KAT-TRAJ-01: ไม่พบ stiff_x ใน stroke summary"
        assert "stiff_y" in stroke, "KAT-TRAJ-01: ไม่พบ stiff_y ใน stroke summary"

    def test_KAT_TRAJ_02_length_consistency(self):
        """
        KAT-TRAJ-02
        len(raw_x) == len(smoothed_x) == len(stiff_x) ต้องเป็น True เสมอ
        (1-to-1 Mapping Guarantee)
        """
        from core.preprocessing import process_strokes
        pts = [SP(t=i*5.0, x=float(i)*2.0, y=100.0, p=0.5, id=1)
               for i in range(30)]
        result = process_strokes(pts)
        stroke = result["processed_strokes"][0]
        n = len(stroke["raw_x"])
        assert len(stroke["smoothed_x"]) == n, "smoothed_x length mismatch"
        assert len(stroke["stiff_x"])    == n, "stiff_x length mismatch"

    def test_KAT_TRAJ_03_stiff_captures_more_tremor(self):
        """
        KAT-TRAJ-03
        สำหรับ stroke ที่มี tremor
        RMS(raw − stiff) ต้องมากกว่า RMS(raw − smoothed)
        เพราะ stiff ไม่เลี้ยวตามรอยสั่น แต่ smoothed เลี้ยวตามบ้าง
        """
        from core.preprocessing import process_strokes

        pts = []
        for i in range(100):
            tremor = 8.0 * math.sin(2 * math.pi * 6 * i * 0.005)
            pts.append(SP(t=i*5.0, x=float(i)*2.0 + tremor,
                          y=200.0 + tremor, p=0.5, id=1))

        result = process_strokes(pts)
        stroke = result["processed_strokes"][0]

        raw_x    = np.array(stroke["raw_x"])
        raw_y    = np.array(stroke["raw_y"])
        smooth_x = np.array(stroke["smoothed_x"])
        smooth_y = np.array(stroke["smoothed_y"])
        stiff_x  = np.array(stroke["stiff_x"])
        stiff_y  = np.array(stroke["stiff_y"])

        rms_a = float(np.sqrt(np.mean(
            (raw_x - smooth_x)**2 + (raw_y - smooth_y)**2
        )))
        rms_b = float(np.sqrt(np.mean(
            (raw_x - stiff_x)**2 + (raw_y - stiff_y)**2
        )))

        assert rms_b > rms_a, (
            f"KAT-TRAJ-03: RMS(raw-stiff)={rms_b:.4f} ต้องมากกว่า "
            f"RMS(raw-smoothed)={rms_a:.4f}"
        )

    def test_KAT_TRAJ_04_path_length_from_smoothed(self):
        """
        KAT-TRAJ-04
        path_length_px ต้องคำนวณจาก Trajectory A (smoothed)
        ไม่ใช่ raw — ดังนั้นค่าต้องน้อยกว่าหรือเท่ากับ raw path
        """
        from core.preprocessing import process_strokes

        pts = []
        for i in range(50):
            tremor = 5.0 * math.sin(2 * math.pi * 6 * i * 0.005)
            pts.append(SP(t=i*5.0, x=float(i)*3.0 + tremor,
                          y=150.0, p=0.5, id=1))

        result = process_strokes(pts)
        stroke = result["processed_strokes"][0]

        raw_x = np.array(stroke["raw_x"])
        raw_y = np.array(stroke["raw_y"])
        raw_path = float(np.sum(np.sqrt(np.diff(raw_x)**2 + np.diff(raw_y)**2)))

        assert stroke["path_length_px"] <= raw_path + 1e-6, (
            f"KAT-TRAJ-04: path_length ({stroke['path_length_px']:.2f}) "
            f"ต้องไม่มากกว่า raw path ({raw_path:.2f})"
        )
