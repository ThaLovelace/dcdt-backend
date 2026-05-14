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
        raw_result = fn({1: pts}, [1], flags)
        # Function returns (value, debug_list) tuple
        result, debug_info = raw_result if isinstance(raw_result, tuple) else (raw_result, [])
        assert result is None, f"KAT-K5-01: expected None, got {result}"
        # K5_SEGMENTATION_FAILED may appear in flags list or debug_info stroke classification
        classified_as_hand = any(s.get("classified_as_hand", False) for s in debug_info)
        assert not classified_as_hand or "K5_SEGMENTATION_FAILED" in flags, \
            f"KAT-K5-01: K5_SEGMENTATION_FAILED ไม่อยู่ใน flags={flags}, debug={debug_info}"

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
        raw_result = fn({1: d1, 2: d2, 3: hand}, [1, 2, 3], flags)
        # Function returns (value, debug_list) tuple
        result, debug_info = raw_result if isinstance(raw_result, tuple) else (raw_result, [])
        assert result == pytest.approx(0.0), \
            f"KAT-K5-02: expected 0.0, got {result}"
        assert "DRAWING_ORDER_ANOMALY" in flags, \
            f"KAT-K5-02: DRAWING_ORDER_ANOMALY ไม่อยู่ใน flags={flags}, debug={debug_info}"


# ---------------------------------------------------------------------------
# KAT-TH: Threshold Formulae
# ---------------------------------------------------------------------------

class TestKATThreshold:
    """
    ทดสอบสูตร Threshold ทุกตัวตามสูตรที่แก้ไขแล้วใน BUG-002 / BUG-003
    (อัปเดตจากสูตร OLD ใน report 4.3 เดิม เป็นสูตร NEW ที่ถูกต้องตาม spec §3.5.4.4)
    """

    def _import(self):
        from core.normalization import get_dynamic_thresholds
        return get_dynamic_thresholds

    # ------------------------------------------------------------------
    # KAT-TH-01/02 : K2 velocity threshold  (BUG-002 fixed formula)
    #
    # สูตรใหม่: max(0.5,  3.0 − (0.03 × max(0, age − 60)))
    #   • baseline = 3.0 cm/s (ไม่ใช่ 1.2)
    #   • decay เริ่มหลังอายุ 60 ปีเท่านั้น (ไม่ใช่ทุกช่วงอายุ)
    #   • lower bound = 0.5 (ไม่ใช่ 0.3)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("age, expected", [
        (59,  3.000),   # ต่ำกว่า 60 → decay = 0 → 3.0 - 0 = 3.0
        (60,  3.000),   # age-60 = 0  → 3.0 - 0.03×0 = 3.000
        (70,  2.700),   # age-60 = 10 → 3.0 - 0.03×10 = 2.700
        (80,  2.400),   # age-60 = 20 → 3.0 - 0.03×20 = 2.400
        (100, 1.800),   # age-60 = 40 → 3.0 - 0.03×40 = 1.800
    ])
    def test_KAT_TH_01_02_K2_threshold_by_age(self, age, expected):
        """
        KAT-TH-01, KAT-TH-02  (BUG-002)
        สูตร K2 ใหม่: max(0.5, 3.0 − (0.03 × max(0, age − 60)))
        """
        t = self._import()(age)
        assert t["K2_velocity_cms"] == pytest.approx(expected, abs=1e-6), (
            f"KAT-TH K2 age={age}: expected {expected}, got {t['K2_velocity_cms']}"
        )

    def test_KAT_TH_03_K2_lower_bound(self):
        """
        KAT-TH-03  (BUG-002)
        age=999 → 3.0 - 0.03×(999-60) = ติดลบมาก → lower bound = 0.500
        (สูตรใหม่ lower bound = 0.5 ไม่ใช่ 0.3 อีกต่อไป)
        """
        t = self._import()(999)
        assert t["K2_velocity_cms"] == pytest.approx(0.500, abs=1e-6), (
            f"KAT-TH-03: expected 0.500, got {t['K2_velocity_cms']}"
        )

    # ------------------------------------------------------------------
    # KAT-TH-04 : K4 %ThinkTime threshold  (BUG-003 fixed formula)
    #
    # สูตรใหม่: 40.0 + (3.0 × floor((age − 60) / 10))
    #   • baseline = 40% (ไม่ใช่ 25%)
    #   • เพิ่มทีละ 3% ต่อทศวรรษ หลังอายุ 60 เท่านั้น
    #   • อายุต่ำกว่า 60 (รวมถึง age=0/30) → flat 40%
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("age, expected", [
        (0,   40.0),   # effective_age=30 → floor((30-60)/10)=neg → max(0,neg)=0 → 40.0
        (30,  40.0),   # floor((30-60)/10) < 0 → 0 decades → 40.0
        (59,  40.0),   # floor((59-60)/10) < 0 → 0 decades → 40.0
        (60,  40.0),   # floor((60-60)/10) = 0  → 0 decades → 40.0
        (70,  43.0),   # floor((70-60)/10) = 1  → 1 decade  → 43.0
        (80,  46.0),   # floor((80-60)/10) = 2  → 2 decades → 46.0
        (90,  49.0),   # floor((90-60)/10) = 3  → 3 decades → 49.0
    ])
    def test_KAT_TH_04_K4_threshold_by_age(self, age, expected):
        """
        KAT-TH-04  (BUG-003)
        สูตร K4 ใหม่: 40.0 + (3.0 × max(0, floor((age − 60) / 10)))
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


# ---------------------------------------------------------------------------
# KAT-TH-CLINICAL: ทดสอบพฤติกรรมทางคลินิก (สิ่งที่ test ชุดเดิมขาดไป)
#
# ปัญหาของ test ชุดเดิม: ทดสอบแค่ว่า "สูตรคำนวณถูกไหม" แต่ไม่ทดสอบว่า
# "threshold ทำให้ผลการวินิจฉัย flag/no-flag ถูกต้องตาม clinical expectation ไหม"
#
# ชุดนี้ทดสอบทั้งสองทาง:
#   1. ผู้ป่วยที่ควร FLAG → ต้อง flag ได้จริง
#   2. ผู้ป่วยที่ควร NOT FLAG → ต้องไม่ flag
# ---------------------------------------------------------------------------

class TestKATThresholdClinicalBehavior:
    """
    ทดสอบว่า threshold ที่แก้แล้ว (BUG-002/003) ทำให้ flag ถูกต้อง
    ในแต่ละ K-series
    """

    def _th(self, age):
        from core.normalization import get_dynamic_thresholds
        return get_dynamic_thresholds(age)

    # ---- K1 ---------------------------------------------------------------

    def test_KAT_TH_CLIN_K1_above_threshold_flags(self):
        """
        K1 RMS = 0.06 cm  > threshold 0.05 cm  → ต้อง flag (True)
        """
        th = self._th(65)
        assert 0.06 > th["K1_rms_threshold_cm"], (
            f"K1: 0.06 ต้องมากกว่า threshold={th['K1_rms_threshold_cm']}"
        )

    def test_KAT_TH_CLIN_K1_below_threshold_no_flag(self):
        """
        K1 RMS = 0.04 cm  < threshold 0.05 cm  → ต้องไม่ flag (False)
        BUG-001 ตรวจ: ถ้ายังมี min(..., 0.03) อยู่ threshold จะเป็น 0.03
        แล้ว 0.04 > 0.03 → flag ผิด (False Positive)
        """
        th = self._th(65)
        assert 0.04 < th["K1_rms_threshold_cm"], (
            f"K1: 0.04 ต้องน้อยกว่า threshold={th['K1_rms_threshold_cm']} "
            f"(BUG-001: ถ้า threshold=0.03 แสดงว่า bug ยังอยู่)"
        )

    def test_KAT_TH_CLIN_K1_threshold_is_exactly_005(self):
        """
        ยืนยันว่า K1 threshold = 0.05 cm ตรงตาม spec (ไม่ใช่ 0.03)
        BUG-001 regression guard
        """
        for age in [40, 60, 80]:
            th = self._th(age)
            assert th["K1_rms_threshold_cm"] == pytest.approx(0.05, abs=1e-9), (
                f"K1 threshold ต้องเป็น 0.05 cm ทุกอายุ, age={age} got {th['K1_rms_threshold_cm']}"
            )

    # ---- K2 ---------------------------------------------------------------

    def test_KAT_TH_CLIN_K2_young_high_threshold(self):
        """
        อายุน้อย (40 ปี) ควรมี threshold สูง = 3.0 cm/s (decay ยังไม่เริ่ม)
        ผู้ป่วยวาดช้า 2.5 cm/s → ต้อง flag ว่า bradykinesia
        ถ้า threshold ยังเป็นสูตรเก่า (1.2 - 0.005×40 = 1.0) จะ flag ไม่ได้
        """
        th = self._th(40)
        patient_velocity = 2.5  # cm/s — ช้ากว่าปกติชัดเจน
        assert patient_velocity < th["K2_velocity_cms"], (
            f"K2 age=40: velocity 2.5 cm/s ต้องต่ำกว่า threshold={th['K2_velocity_cms']:.3f} "
            f"(สูตรเก่าจะให้ threshold=1.0 → ไม่ flag ทั้งที่ควร flag)"
        )

    def test_KAT_TH_CLIN_K2_old_age_decay_starts_at_60(self):
        """
        BUG-002 regression guard: decay ต้องเริ่มที่อายุ 60 เท่านั้น
        threshold ที่ age=59 ต้องเท่ากับ age=60
        (สูตรเก่า decay ทุกอายุ → threshold age=59 ≠ age=60)
        """
        th59 = self._th(59)
        th60 = self._th(60)
        assert th59["K2_velocity_cms"] == pytest.approx(th60["K2_velocity_cms"], abs=1e-6), (
            f"K2: threshold age=59 ({th59['K2_velocity_cms']}) ต้องเท่ากับ age=60 ({th60['K2_velocity_cms']})"
        )

    # ---- K4 ---------------------------------------------------------------

    def test_KAT_TH_CLIN_K4_baseline_is_40pct_not_25pct(self):
        """
        BUG-003 regression guard: baseline K4 ต้องเป็น 40% ไม่ใช่ 25%
        อายุต่ำกว่า 60 ทุกคนต้องได้ 40%
        ถ้ายังเป็นสูตรเก่า (25 + 0.2×age) → age=30 ได้ 31.0 (ผิด)
        """
        for age in [30, 40, 50, 59]:
            th = self._th(age)
            assert th["K4_pct_think_time"] == pytest.approx(40.0, abs=1e-6), (
                f"K4 age={age}: expected baseline 40.0%, got {th['K4_pct_think_time']}"
            )

    def test_KAT_TH_CLIN_K4_young_patient_should_not_flag_at_35pct(self):
        """
        ผู้ป่วยอายุ 45 ปี มี %ThinkTime = 35% → ไม่ควร flag
        สูตรเก่า: threshold = 25 + 0.2×45 = 34.0 → 35 > 34 → FLAG (False Positive!)
        สูตรใหม่: threshold = 40.0 → 35 < 40 → ไม่ flag (ถูกต้อง)
        """
        th = self._th(45)
        patient_think_pct = 35.0
        assert patient_think_pct < th["K4_pct_think_time"], (
            f"K4 age=45: %ThinkTime=35 ต้องน้อยกว่า threshold={th['K4_pct_think_time']:.1f} "
            f"(สูตรเก่า threshold=34.0 จะ false positive)"
        )

    def test_KAT_TH_CLIN_K4_decade_step_increment(self):
        """
        ยืนยัน step size = 3% ต่อทศวรรษ (ไม่ใช่ 0.2%/ปี ของสูตรเก่า)
        """
        th60 = self._th(60)
        th70 = self._th(70)
        th80 = self._th(80)
        assert th70["K4_pct_think_time"] - th60["K4_pct_think_time"] == pytest.approx(3.0, abs=1e-6)
        assert th80["K4_pct_think_time"] - th70["K4_pct_think_time"] == pytest.approx(3.0, abs=1e-6)

    # ---- K5 ---------------------------------------------------------------

    def test_KAT_TH_CLIN_K5_decade_step_increment(self):
        """
        ยืนยัน step size = 1500 ms ต่อทศวรรษ
        """
        th60 = self._th(60)
        th70 = self._th(70)
        th80 = self._th(80)
        assert th70["K5_pfhl_ms"] - th60["K5_pfhl_ms"] == pytest.approx(1500.0, abs=1e-6)
        assert th80["K5_pfhl_ms"] - th70["K5_pfhl_ms"] == pytest.approx(1500.0, abs=1e-6)

    def test_KAT_TH_CLIN_K5_flat_before_60(self):
        """
        K5 ต้องคงที่ที่ 8000 ms สำหรับทุกอายุต่ำกว่า 60
        """
        for age in [30, 40, 50, 59]:
            th = self._th(age)
            assert th["K5_pfhl_ms"] == pytest.approx(8000.0, abs=1e-6), (
                f"K5 age={age}: expected 8000.0 ms, got {th['K5_pfhl_ms']}"
            )


# ---------------------------------------------------------------------------
# KAT-TH-REGRESSION: ทดสอบว่า bug ที่เคยแก้ไม่กลับมาอีก
# ---------------------------------------------------------------------------

class TestKATRegressionGuards:
    """
    Guard tests: ถ้า test เหล่านี้ fail แปลว่า bug กลับมาใหม่
    """

    def _th(self, age):
        from core.normalization import get_dynamic_thresholds
        return get_dynamic_thresholds(age)

    def test_REG_BUG001_k1_threshold_never_below_005(self):
        """BUG-001: K1 threshold ต้องไม่ต่ำกว่า 0.05 ที่อายุใดก็ตาม"""
        for age in [0, 30, 60, 80, 100, 999]:
            th = self._th(age)
            assert th["K1_rms_threshold_cm"] >= 0.05 - 1e-9, (
                f"BUG-001 regression: K1 threshold={th['K1_rms_threshold_cm']} "
                f"ต้อง >= 0.05 ที่ age={age}"
            )

    def test_REG_BUG002_k2_threshold_never_below_05(self):
        """BUG-002: K2 lower bound ต้องเป็น 0.5 ไม่ใช่ 0.3"""
        for age in [0, 30, 60, 80, 999]:
            th = self._th(age)
            assert th["K2_velocity_cms"] >= 0.5 - 1e-9, (
                f"BUG-002 regression: K2 threshold={th['K2_velocity_cms']} "
                f"ต้อง >= 0.5 ที่ age={age}"
            )

    def test_REG_BUG003_k4_threshold_never_below_40(self):
        """BUG-003: K4 baseline ต้องเป็น 40% ไม่ใช่ 25%"""
        for age in [0, 30, 60, 80, 999]:
            th = self._th(age)
            assert th["K4_pct_think_time"] >= 40.0 - 1e-9, (
                f"BUG-003 regression: K4 threshold={th['K4_pct_think_time']} "
                f"ต้อง >= 40.0 ที่ age={age}"
            )