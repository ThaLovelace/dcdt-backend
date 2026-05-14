"""
test_education_bias.py — TC-009 to TC-018
ทดสอบ Education Bias Warning logic
  - แนบ EDUCATION_BIAS_WARNING เมื่อ AI Abnormal AND education_years < 8
  - ห้ามเปลี่ยน risk level
  - Boundary: education_years = 8
"""
import pytest
from conftest import base_payload


class TestTC009_C4_EduBelow:
    """C4, edu=7 → ต้องมี WARNING"""
    def test_warning_appended(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=7))
        assert "EDUCATION_BIAS_WARNING" in r.json()["warnings"]

    def test_risk_unchanged(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=7))
        assert r.json()["risk_level"] == "mild"

    def test_class_unchanged(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=7))
        assert r.json()["class_id"] == "C4"


class TestTC010_C4_EduBoundary:
    """C4, edu=8 (ขอบเขต) → ห้ามมี WARNING"""
    def test_no_warning_at_boundary(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=8))
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]


class TestTC011_C4_EduAbove:
    """C4, edu=9 → ห้ามมี WARNING"""
    def test_no_warning_above(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=9))
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]


class TestTC012_C5_EduBelow:
    """C5, edu=5 → ต้องมี WARNING, risk ยังเป็น high"""
    def test_warning_appended(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=5))
        assert "EDUCATION_BIAS_WARNING" in r.json()["warnings"]

    def test_risk_still_high(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=5))
        assert r.json()["risk_level"] == "high"


class TestTC013_C5_EduBoundary:
    """C5, edu=8 → ห้ามมี WARNING"""
    def test_no_warning(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=8))
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]


class TestTC014_C6_EduBelow:
    """C6, edu=3 → ต้องมี WARNING"""
    def test_warning_appended(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=3))
        assert "EDUCATION_BIAS_WARNING" in r.json()["warnings"]

    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=3))
        assert r.json()["class_id"] == "C6"


class TestTC015_C6_EduBoundary:
    """C6, edu=8 → ห้ามมี WARNING"""
    def test_no_warning(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=8))
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]


class TestTC016_C7_EduMinimum:
    """C7, edu=0 (ต่ำสุด) → ต้องมี WARNING"""
    def test_warning_appended(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=0))
        assert "EDUCATION_BIAS_WARNING" in r.json()["warnings"]

    def test_risk_still_high(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=0))
        assert r.json()["risk_level"] == "high"


class TestTC017_C7_EduBoundary:
    """C7, edu=8 → ห้ามมี WARNING"""
    def test_no_warning(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload(education_years=8))
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]


class TestTC018_C0_AINormal_LowEdu:
    """C0 (AI Normal), edu=4 → ห้ามมี WARNING เลย (AI Normal ไม่ trigger warning)"""
    def test_no_warning(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=False)
        r = client.post("/analyze", json=base_payload(education_years=4))
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]

    def test_risk_normal(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=False)
        r = client.post("/analyze", json=base_payload(education_years=4))
        assert r.json()["risk_level"] == "normal"


@pytest.mark.parametrize("k1,k4,ai,edu,expected_class,expect_warning", [
    (False, False, True,  7, "C4", True),
    (False, True,  True,  7, "C5", True),
    (True,  False, True,  7, "C6", True),
    (True,  True,  True,  7, "C7", True),
    (False, False, True,  8, "C4", False),
    (False, True,  True,  8, "C5", False),
    (True,  False, True,  8, "C6", False),
    (True,  True,  True,  8, "C7", False),
    (False, False, False, 4, "C0", False),
    (True,  False, False, 4, "C1", False),
    (False, True,  False, 4, "C2", False),
    (True,  True,  False, 4, "C3", False),
], ids=["C4-edu7","C5-edu7","C6-edu7","C7-edu7",
        "C4-edu8","C5-edu8","C6-edu8","C7-edu8",
        "C0-edu4","C1-edu4","C2-edu4","C3-edu4"])
def test_education_bias_parametrized(client, mock_pipeline, k1, k4, ai, edu, expected_class, expect_warning):
    mock_pipeline(k1=k1, k4=k4, ai_abnormal=ai)
    r = client.post("/analyze", json=base_payload(education_years=edu))
    assert r.status_code == 200
    assert r.json()["class_id"] == expected_class
    if expect_warning:
        assert "EDUCATION_BIAS_WARNING" in r.json()["warnings"]
    else:
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]
