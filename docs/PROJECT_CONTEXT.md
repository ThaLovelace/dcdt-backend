# dCDT Backend — Project Overview
uvicorn main:app --reload

## Summary

This is a **FastAPI** backend designed to receive raw drawing JSON data from a frontend application implementing the **Digital Clock Drawing Test (dCDT)**.

## Input Data Format

The backend accepts high-fidelity stroke data in JSON format. The payload is expected to be an object containing a `strokes` array, with the following fields per data point:

| Field | Type  | Description                                        |
|-------|-------|----------------------------------------------------|
| `t`   | float | High-resolution monotonic time (e.g., performance.now) in ms |
| `x`   | float | X-coordinate of the drawing stroke                 |
| `y`   | float | Y-coordinate of the drawing stroke                 |
| `p`   | float | Stylus/touch pressure value (0.0 to 1.0)           |
| `az`  | float | Azimuth angle in radians                           |
| `alt` | float | Altitude angle in radians                          |
| `id`  | int   | Global Stroke ID (auto-increments on pen-down)     |

## Processing Pipeline

Once raw data is received, it will be processed through the following pipeline:

1. **Savitzky-Golay Filter** — Smooths the raw stroke data to reduce noise while preserving the shape of the signal.
2. **K-Means Clustering** — Groups stroke segments into meaningful clusters for spatial analysis.
3. **K-Series Feature Extraction (K1–K5)** — Extracts a set of kinematic and geometric features (K1 through K5) from the processed strokes.
4. **ViT-B/16 Model Inference** — Passes extracted features or rendered images through a Vision Transformer (ViT-B/16) model to produce a final classification or scoring output.

## Technology Stack

- **Framework:** FastAPI
- **Signal Processing:** SciPy (Savitzky-Golay)
- **Machine Learning:** scikit-learn (K-Means), ONNX Runtime (ViT-B/16 inference)
- **Data Validation:** Pydantic
- **Server:** Uvicorn (ASGI)

## Purpose

This backend serves as the core computation engine for the dCDT project, which aims to digitize and automate the scoring of the Clock Drawing Test — a widely used neuropsychological screening tool for cognitive impairment.
