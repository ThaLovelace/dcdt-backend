"""
test_ai_threshold.py — TC-019 to TC-024
ทดสอบ threshold ของ AI domain ที่ 0.2741
ใช้ mock run_real_vit_inference โดยตรงเพราะ threshold อยู่ใน inference.py แล้ว
"""
import pytest
import core.inference as inference_module
from conftest import base_payload, DUMMY_IMAGE_B64


class TestTC019_JustBelowThreshold:
    """logit=0.2740 → AI Normal → C0"""
    def test_class_c0(self, client, monkeypatch, mock_pipeline):
        # patch ViT คืน prob < 0.2741 → ai_abnormal=False
        monkeypatch.setattr(inference_module, "run_real_vit_inference",
                            lambda img: (False, 27.40, ""))
        monkeypatch.setattr(inference_module, "evaluate_k_series",
                            lambda f, t: {"K1":False,"K2":False,"K3":False,"K4":False,"K5":False})
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C0"
        assert r.json()["risk_level"] == "normal"


class TestTC020_ExactlyAtThreshold:
    """logit=0.2741 (exactly) → AI Abnormal → C4  ← จุดวิกฤตสุด"""
    def test_class_c4(self, client, monkeypatch):
        monkeypatch.setattr(inference_module, "run_real_vit_inference",
                            lambda img: (True, 27.41, ""))
        monkeypatch.setattr(inference_module, "evaluate_k_series",
                            lambda f, t: {"K1":False,"K2":False,"K3":False,"K4":False,"K5":False})
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C4"
        assert r.json()["risk_level"] == "mild"

    def test_not_normal(self, client, monkeypatch):
        """ห้าม classify ว่า Normal — false negative คือ patient safety failure"""
        monkeypatch.setattr(inference_module, "run_real_vit_inference",
                            lambda img: (True, 27.41, ""))
        monkeypatch.setattr(inference_module, "evaluate_k_series",
                            lambda f, t: {"K1":False,"K2":False,"K3":False,"K4":False,"K5":False})
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] != "C0"
        assert r.json()["risk_level"] != "normal"


class TestTC021_JustAboveThreshold:
    """logit=0.2742 → AI Abnormal → C4"""
    def test_class_c4(self, client, monkeypatch):
        monkeypatch.setattr(inference_module, "run_real_vit_inference",
                            lambda img: (True, 27.42, ""))
        monkeypatch.setattr(inference_module, "evaluate_k_series",
                            lambda f, t: {"K1":False,"K2":False,"K3":False,"K4":False,"K5":False})
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C4"


class TestTC022_MinimumLogit:
    """logit=0.0 → AI Normal → C0"""
    def test_class_c0(self, client, monkeypatch):
        monkeypatch.setattr(inference_module, "run_real_vit_inference",
                            lambda img: (False, 0.0, ""))
        monkeypatch.setattr(inference_module, "evaluate_k_series",
                            lambda f, t: {"K1":False,"K2":False,"K3":False,"K4":False,"K5":False})
        r = client.post("/analyze", json=base_payload())
        assert r.json()["risk_level"] == "normal"


class TestTC023_MaximumLogit:
    """logit=1.0, all K triggered → C7 High"""
    def test_class_c7(self, client, monkeypatch):
        monkeypatch.setattr(inference_module, "run_real_vit_inference",
                            lambda img: (True, 100.0, ""))
        monkeypatch.setattr(inference_module, "evaluate_k_series",
                            lambda f, t: {"K1":True,"K2":True,"K3":True,"K4":True,"K5":True})
        r = client.post("/analyze", json=base_payload())
        assert r.json()["class_id"] == "C7"
        assert r.json()["risk_level"] == "high"


class TestTC024_NegativeLogit:
    """logit<0 → AI Normal → C0 (ห้าม crash)"""
    def test_no_crash(self, client, monkeypatch):
        monkeypatch.setattr(inference_module, "run_real_vit_inference",
                            lambda img: (False, 0.0, ""))
        monkeypatch.setattr(inference_module, "evaluate_k_series",
                            lambda f, t: {"K1":False,"K2":False,"K3":False,"K4":False,"K5":False})
        r = client.post("/analyze", json=base_payload())
        assert r.status_code != 500
        assert r.json()["risk_level"] == "normal"
