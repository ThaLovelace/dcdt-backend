"""
test_truth_table.py — TC-001 to TC-008
ครอบคลุม Class ID C0–C7 ทั้งหมดจาก Truth Table
"""

import pytest
from conftest import base_payload

AI_NORMAL   = False
AI_ABNORMAL = True


class TestTC001_C0:
    """AI=Normal, Motor=Normal, Cognitive=Normal → Normal (Green)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.status_code == 200
        assert r.json()["class_id"] == "C0"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "normal"

    def test_risk_color(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_color"] == "green"

    def test_no_warnings(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]


class TestTC002_C1:
    """AI=Normal, Motor=Abnormal(K1), Cognitive=Normal → Mild (Yellow)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C1"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "mild"

    def test_risk_color(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_color"] == "yellow"


class TestTC003_C2:
    """AI=Normal, Motor=Normal, Cognitive=Abnormal(K4) → Mild (Yellow)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C2"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "mild"


class TestTC004_C3:
    """AI=Normal, Motor=Abnormal(K1), Cognitive=Abnormal(K4) → Mild (Yellow)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C3"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=AI_NORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "mild"


class TestTC005_C4:
    """AI=Abnormal, Motor=Normal, Cognitive=Normal → Mild (Yellow)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C4"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "mild"

    def test_no_warning_when_edu_sufficient(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload(education_years=12))
        assert "EDUCATION_BIAS_WARNING" not in r.json()["warnings"]


class TestTC006_C5:
    """AI=Abnormal, Motor=Normal, Cognitive=Abnormal(K4) → High (Red)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C5"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "high"

    def test_risk_color(self, client, mock_pipeline):
        mock_pipeline(k4=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_color"] == "red"


class TestTC007_C6:
    """AI=Abnormal, Motor=Abnormal(K1), Cognitive=Normal → High (Red)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C6"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(k1=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "high"


class TestTC008_C7:
    """AI=Abnormal, Motor=Abnormal(K1), Cognitive=Abnormal(K4) → High (Red)"""
    def test_class_id(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C7"

    def test_risk_level(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "high"

    def test_risk_color(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True, ai_abnormal=AI_ABNORMAL)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_color"] == "red"


@pytest.mark.parametrize("k1,k4,ai,expected_class,expected_risk", [
    (False, False, False, "C0", "normal"),
    (True,  False, False, "C1", "mild"),
    (False, True,  False, "C2", "mild"),
    (True,  True,  False, "C3", "mild"),
    (False, False, True,  "C4", "mild"),
    (False, True,  True,  "C5", "high"),
    (True,  False, True,  "C6", "high"),
    (True,  True,  True,  "C7", "high"),
], ids=["C0","C1","C2","C3","C4","C5","C6","C7"])
def test_truth_table_parametrized(client, mock_pipeline, k1, k4, ai, expected_class, expected_risk):
    mock_pipeline(k1=k1, k4=k4, ai_abnormal=ai)
    r = client.post("/analyze", json=base_payload())
    assert r.status_code == 200
    assert r.json()["class_id"]   == expected_class
    assert r.json()["risk_level"] == expected_risk
