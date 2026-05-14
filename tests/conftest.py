"""
conftest.py — Shared fixtures for CDSS dCDT Hybrid Decision Logic Test Suite
Document Reference: TP-CDSS-HDL-001 v1.0
python -m pytest tests/ --ignore=tests/archive/ -v
"""

import pytest
import sys
import os
import base64

# ---------------------------------------------------------------------------
# เชื่อมกับโปรเจกต์จริง
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app
import core.inference as inference_module

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    return TestClient(app)


DUMMY_IMAGE_B64 = base64.b64encode(b"FAKE_IMAGE_BYTES").decode()


def make_kinematic_strokes() -> list:
    """
    สร้าง strokes จำลองให้ตรงกับ StrokePoint schema ทุก field:
      t, x, y, p, az, alt, id
    """
    return [
        {"t": 0.0,   "x": 100.0, "y": 100.0, "p": 0.5, "az": 0.0, "alt": 1.5, "id": 0},
        {"t": 16.0,  "x": 101.0, "y": 101.0, "p": 0.5, "az": 0.0, "alt": 1.5, "id": 0},
        {"t": 32.0,  "x": 102.0, "y": 102.0, "p": 0.5, "az": 0.0, "alt": 1.5, "id": 0},
        {"t": 48.0,  "x": 103.0, "y": 103.0, "p": 0.5, "az": 0.0, "alt": 1.5, "id": 0},
        {"t": 64.0,  "x": 104.0, "y": 104.0, "p": 0.5, "az": 0.0, "alt": 1.5, "id": 0},
    ]


def base_payload(education_years: int = 12) -> dict:
    """
    Payload พื้นฐานตรงกับ AnalysisRequest schema ทุก field:
      strokes, image_b64, patient_age, education_years, device_dpi
    """
    return {
        "image_b64":       DUMMY_IMAGE_B64,
        "strokes":         make_kinematic_strokes(),
        "patient_age":     65,
        "education_years": education_years,
        "device_dpi":      96.0,
    }


@pytest.fixture
def mock_pipeline(monkeypatch):
    """
    Mock ฟังก์ชันหลัก 2 ตัวใน core/inference.py:
      1. evaluate_k_series      → คืน K1-K5 ตามที่ test กำหนด
      2. run_real_vit_inference → คืน (ai_abnormal, confidence, heatmap) ตามที่ test กำหนด

    วิธีใช้:
        def test_foo(client, mock_pipeline):
            mock_pipeline(k1=True, ai_abnormal=True)
            r = client.post("/analyze", json=base_payload(education_years=7))
    """
    def _setup(
        k1: bool = False, k2: bool = False, k3: bool = False,
        k4: bool = False, k5: bool = False,
        ai_abnormal: bool = False,
    ):
        monkeypatch.setattr(
            inference_module, "evaluate_k_series",
            lambda features, thresholds: {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5}
        )
        monkeypatch.setattr(
            inference_module, "run_real_vit_inference",
            lambda image_b64: (ai_abnormal, 99.0, "")
        )

    return _setup