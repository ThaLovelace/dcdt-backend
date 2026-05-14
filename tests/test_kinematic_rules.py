"""
test_kinematic_rules.py — TC-025 to TC-034
ทดสอบ OR logic ของ K-rules
  Motor    = K1 OR K2 OR K3
  Cognitive = K4 OR K5
"""
import pytest
from conftest import base_payload


class TestTC025_AllFalse:
    """K1–K5=False → C0 Normal"""
    def test_class_c0(self, client, mock_pipeline):
        mock_pipeline()
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C0"


class TestTC026_K1Only:
    """K1=True เท่านั้น → Motor Abnormal → C1"""
    def test_class_c1(self, client, mock_pipeline):
        mock_pipeline(k1=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C1"

    def test_cognitive_not_triggered(self, client, mock_pipeline):
        mock_pipeline(k1=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] != "C3"


class TestTC027_K2Only:
    """K2=True เท่านั้น → Motor Abnormal → C1"""
    def test_class_c1(self, client, mock_pipeline):
        mock_pipeline(k2=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C1"


class TestTC028_K3Only:
    """K3=True เท่านั้น → Motor Abnormal → C1"""
    def test_class_c1(self, client, mock_pipeline):
        mock_pipeline(k3=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C1"


class TestTC029_AllMotorRules:
    """K1+K2+K3=True → Motor Abnormal (OR), Cognitive ยังเป็น Normal → C1"""
    def test_class_c1(self, client, mock_pipeline):
        mock_pipeline(k1=True, k2=True, k3=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C1"

    def test_not_c3(self, client, mock_pipeline):
        mock_pipeline(k1=True, k2=True, k3=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] != "C3"


class TestTC030_K4Only:
    """K4=True เท่านั้น → Cognitive Abnormal → C2"""
    def test_class_c2(self, client, mock_pipeline):
        mock_pipeline(k4=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C2"

    def test_motor_not_triggered(self, client, mock_pipeline):
        mock_pipeline(k4=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] != "C3"


class TestTC031_K5Only:
    """K5=True เท่านั้น → Cognitive Abnormal → C2"""
    def test_class_c2(self, client, mock_pipeline):
        mock_pipeline(k5=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C2"


class TestTC032_AllCognitiveRules:
    """K4+K5=True → Cognitive Abnormal (OR), Motor ยังเป็น Normal → C2"""
    def test_class_c2(self, client, mock_pipeline):
        mock_pipeline(k4=True, k5=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C2"


class TestTC033_K1andK4:
    """K1+K4=True → Motor+Cognitive Abnormal → C3"""
    def test_class_c3(self, client, mock_pipeline):
        mock_pipeline(k1=True, k4=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C3"
        assert r.json()["risk_color"] == "yellow"


class TestTC034_AllRules:
    """K1–K5=True ทั้งหมด + AI Normal → C3 (ไม่ High Risk เพราะ AI Normal)"""
    def test_class_c3(self, client, mock_pipeline):
        mock_pipeline(k1=True, k2=True, k3=True, k4=True, k5=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C3"

    def test_not_high_risk(self, client, mock_pipeline):
        mock_pipeline(k1=True, k2=True, k3=True, k4=True, k5=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] != "high"


@pytest.mark.parametrize("k1,k2,k3,k4,k5,expected_class", [
    (False, False, False, False, False, "C0"),
    (True,  False, False, False, False, "C1"),
    (False, True,  False, False, False, "C1"),
    (False, False, True,  False, False, "C1"),
    (True,  True,  True,  False, False, "C1"),
    (False, False, False, True,  False, "C2"),
    (False, False, False, False, True,  "C2"),
    (False, False, False, True,  True,  "C2"),
    (True,  False, False, True,  False, "C3"),
    (True,  True,  True,  True,  True,  "C3"),
], ids=["all-F","K1","K2","K3","K123","K4","K5","K45","K1+K4","all-T"])
def test_kinematic_parametrized(client, mock_pipeline, k1, k2, k3, k4, k5, expected_class):
    mock_pipeline(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, ai_abnormal=False)
    r = client.post("/analyze", json=base_payload())
    assert r.status_code == 200
    assert r.json()["class_id"] == expected_class
