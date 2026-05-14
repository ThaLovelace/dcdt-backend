"""
test_negative.py — TC-035 to TC-042
Negative & Robustness Tests
"""
import pytest
from conftest import base_payload, DUMMY_IMAGE_B64


def good_payload(**overrides):
    p = base_payload()
    p.update(overrides)
    return p


class TestTC035_MissingImage:
    """ไม่ส่ง image_b64 → 422"""
    def test_returns_422(self, client):
        p = base_payload()
        del p["image_b64"]
        r = client.post("/analyze", json=p)
        assert r.status_code == 422

    def test_not_500(self, client):
        p = base_payload()
        del p["image_b64"]
        r = client.post("/analyze", json=p)
        assert r.status_code != 500


class TestTC036_MalformedBase64:
    """image_b64 ไม่ใช่ Base64 จริง → ต้องไม่ได้ 200
    
    หมายเหตุ: ไม่ใช้ mock_pipeline เพราะต้องการให้ vision_bridge.py
    ทำงานจริงเพื่อตรวจจับ Base64 ที่ผิด
    """
    def test_not_200(self, client):
        # ไม่ mock — ให้ pipeline จริงทำงานและ reject Base64 ที่ผิด
        r = client.post("/analyze", json=good_payload(image_b64="NOT_VALID_BASE64!!!"))
        assert r.status_code in (400, 422, 500)
        # สำคัญที่สุดคือห้ามคืน classification ที่ผิดพลาด
        assert r.status_code != 200

    def test_not_silent_success(self, client):
        """ห้ามระบบ classify ออกมาเป็น C0/Normal เมื่อ image ผิด"""
        r = client.post("/analyze", json=good_payload(image_b64="NOT_VALID_BASE64!!!"))
        if r.status_code == 200:
            # ถ้าระบบยังคืน 200 แสดงว่ายังมี bug — ต้องไม่มี class_id
            assert False, f"ระบบคืน 200 OK พร้อม classification สำหรับ Base64 ที่ผิด: {r.json()}"


class TestTC037_MissingStrokes:
    """ไม่ส่ง strokes → 422"""
    def test_returns_422(self, client):
        p = base_payload()
        del p["strokes"]
        r = client.post("/analyze", json=p)
        assert r.status_code == 422


class TestTC038_NegativeEducation:
    """education_years = -1 → 422"""
    def test_returns_422(self, client):
        r = client.post("/analyze", json=good_payload(education_years=-1))
        assert r.status_code == 422

    def test_not_500(self, client):
        r = client.post("/analyze", json=good_payload(education_years=-1))
        assert r.status_code != 500


class TestTC039_EducationAsString:
    """education_years = '7.5' (string) → 422"""
    def test_returns_422(self, client):
        r = client.post("/analyze", json=good_payload(education_years="7.5"))
        assert r.status_code in (400, 422)

    def test_not_500(self, client):
        r = client.post("/analyze", json=good_payload(education_years="7.5"))
        assert r.status_code != 500


class TestTC040_OutOfRangeLogit:
    """AI คืน logit=2.5 → ห้าม crash, classify เป็น Abnormal"""
    def test_no_crash(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=True)
        r = client.post("/analyze", json=base_payload())
        assert r.status_code != 500

    def test_classified_abnormal(self, client, mock_pipeline):
        mock_pipeline(ai_abnormal=True)
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] in ("C4", "C5", "C6", "C7")


class TestTC041_AllKinematicTriggered:
    """K1–K5 ทั้งหมด + AI Abnormal → C7 High"""
    def test_class_c7(self, client, mock_pipeline):
        mock_pipeline(k1=True, k2=True, k3=True, k4=True, k5=True, ai_abnormal=True)
        r = client.post("/analyze", json=base_payload())
        assert r.status_code == 200
        assert r.json()["class_id"] == "C7"
        assert r.json()["risk_level"] == "high"


class TestTC042_MissingEducationYears:
    """ไม่ส่ง education_years → 422"""
    def test_returns_422(self, client):
        p = base_payload()
        del p["education_years"]
        r = client.post("/analyze", json=p)
        assert r.status_code == 422

    def test_not_500(self, client):
        p = base_payload()
        del p["education_years"]
        r = client.post("/analyze", json=p)
        assert r.status_code != 500


@pytest.mark.parametrize("missing_field", [
    "image_b64", "strokes", "education_years", "patient_age"
])
def test_missing_required_fields(client, missing_field):
    p = base_payload()
    del p[missing_field]
    r = client.post("/analyze", json=p)
    assert r.status_code == 422
    assert r.status_code != 500