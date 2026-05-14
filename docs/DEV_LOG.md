# Development Log

> This file tracks all development changes made to the dCDT backend. Each entry should include the date, task description, files modified, and any relevant notes for the academic report.

---

## Log Format

```
### [YYYY-MM-DD] — <Short Task Title>
- **Files Modified/Created:** <list of files>
- **Description:** <what was done and why>
- **Notes:** <any relevant observations or decisions>
```

---

### [2026-03-31] — Project Initialization

- **Files Created:**
  - `requirements.txt`
  - `.clinerules`
  - `main.py`
  - `docs/PROJECT_CONTEXT.md`
  - `docs/DEV_LOG.md`
  - `docs/TASK_BOARD.md`
  - Directories: `api/`, `core/`, `models/`, `docs/`
- **Description:** Initialized the FastAPI backend project structure for the dCDT (Digital Clock Drawing Test). Set up the base directory layout, dependency list, project context documentation, and a minimal FastAPI application with a health-check and test POST endpoint.
- **Notes:** Project uses FastAPI + Uvicorn as the ASGI server. Processing pipeline will involve SciPy (Savitzky-Golay), scikit-learn (K-Means), and ONNX Runtime (ViT-B/16). Raw input data schema: `x`, `y`, `t`, `pressure`.

### 2026-04-03 - Initial Backend Scan

**Summary:** Performed a comprehensive scan of the existing dCDT backend repository to understand its architecture, structure, and current state. This report serves as a baseline before integrating the new dCDT analysis engine.

**Details:**
- **Tech Stack:** FastAPI (framework), Uvicorn (server), Pydantic (data validation), SciPy (signal processing), scikit-learn (ML), ONNX Runtime (ML inference).
- **Directory Architecture:** The project has a flat structure with `main.py` containing most of the logic. Directories like `api`, `core`, and `models` exist but are currently empty.
- **Existing Data Models:** Two Pydantic schemas exist: `StrokePoint` (x, y, t, pressure) and `DrawingPayload` (list of `StrokePoint`s). The `StrokePoint` schema does not include `p`, `az`, `alt`, or `id` fields from the new incoming JSON payload.
- **Existing API Endpoints:**
    - `GET /`: Health check.
    - `POST /test`: Accepts `DrawingPayload` and echoes a summary for testing purposes.
- **Pre-processing / ML Logic:** No existing implementation for stroke smoothing, kinematics, or ML inference. The `PROJECT_CONTEXT.md` outlines the planned pipeline, but the code is not yet present.
- **Gap Analysis:**
    1. **Data Model Mismatch:** The `StrokePoint` schema needs to be updated to include `p`, `az`, `alt`, and `id` to match the incoming high-fidelity JSON payload.
    2. **Missing Core Logic:** Dedicated modules for signal pre-processing (Savitzky-Golay), kinematic analysis, and ONNX model inference need to be implemented.
    3. **Endpoint for Processing:** A new endpoint (e.g., `/process/drawing`) will be required to receive the full drawing payload and trigger the analysis pipeline.
    4. **Module Organization:** The current flat structure in `main.py` will become messy with the new logic. It's recommended to organize processing logic into dedicated modules within `core/` and `api/`.

### [2026-04-03] — Milestone 2 Completion: Data Schema & API Endpoint
- **Files Modified/Created:** `docs/PROJECT_CONTEXT.md`, `models/schemas.py`, `api/routes.py`, `main.py`, `docs/TASK_BOARD.md`
- **Description:** Updated the project context to reflect the new 7-field high-fidelity JSON payload. Created Pydantic schemas (`StrokePoint`, `DrawingPayload`) in `models/schemas.py` to rigorously validate incoming data. Created the `/api/analyze` endpoint in `api/routes.py` to receive the payload and integrated the router cleanly into `main.py`.
- **Notes:** The backend is now fully capable of parsing the frontend's output without 422 Unprocessable Entity errors. Ready to begin Milestone 3 (Savitzky-Golay signal processing).

4/4/2026: Updated `requirements.txt` to include `python-multipart` for file uploads. Explained how to run `main.py` using Uvicorn.

---

### [2026-05-14] — Bug Fix: K1 Dual-Trajectory Architecture & K2 Path Length Correction

- **Files Modified:**
  - `core/preprocessing.py`
  - `core/kinematics.py`
  - `core/normalization.py`

- **Description:**
  During real-device testing (mouse and iPad), the system incorrectly flagged all drawings as having tremor (K1 always triggered). Root cause analysis identified three implementation gaps between the technical specification (Chapter 4) and the actual code:

  **Bug 1 — K1 used wrong reference trajectory (critical)**
  The specification (§4.3.2) explicitly requires a Dual-Trajectory Architecture where K1 tremor is measured as `RMS(raw − Trajectory B)`. Trajectory B is a "stiff" reference built with a 350 ms window and `polyorder=2`, wide enough to span tremor cycles (4–8 Hz) without bending into them. The original code incorrectly passed Trajectory A (50 ms, `polyorder=3`) as the reference, resulting in near-zero RMS values (Trajectory A follows the tremor) or inflated values depending on stroke length.

  **Fix applied to `preprocessing.py`:**
  - Added `_build_stiff_reference()` function that computes Trajectory B using `TARGET_STIFF_WINDOW_MS=350`, `K1_STIFF_MIN_WINDOW=21`, `STIFF_POLY_ORDER=2`.
  - `process_strokes()` now outputs `stiff_x` and `stiff_y` for every kinematic-eligible stroke, alongside the existing `smoothed_x`/`smoothed_y` (Trajectory A).
  - Verified: `RMS(raw − TrajB) ≈ 5.66 px` vs `RMS(raw − TrajA) ≈ 0.02 px` for a synthetic 6 Hz tremor signal, confirming Trajectory B correctly captures tremor amplitude.

  **Fix applied to `kinematics.py`:**
  - `compute_k1_rms()` signature changed from `(raw_x, raw_y, smoothed_x, smoothed_y, dpi)` to `(raw_x, raw_y, stiff_x, stiff_y, dpi)`.
  - Docstring and error messages updated to reflect Trajectory B semantics.
  - `extract_all_features()` orchestrator updated to pass `stroke["stiff_x"]` and `stroke["stiff_y"]` instead of `smoothed_x`/`smoothed_y`.

  **Bug 2 — K2 path length computed from raw coordinates (tremor inflation)**
  The specification (§4.3.3) states that K2 velocity must use Trajectory A coordinates to prevent tremor inflation — raw coordinates zigzag with each tremor cycle, making the path appear longer and velocity artificially high.

  **Fix applied to `preprocessing.py`:**
  - `path_length_px` is now computed from `smoothed_x`/`smoothed_y` (Trajectory A) instead of `x_arr`/`y_arr` (raw). This is computed once in preprocessing so `compute_k2_velocity()` in `kinematics.py` requires no changes.

  **Bug 3 — K2 threshold formula did not match specification**
  The specification (§4.3.3, Listing 4.8) defines the K2 threshold as `1.2 − (0.005 × age)`, a linear decrease across all ages. The original `normalization.py` used `3.0 − (0.03 × max(0, age − 60))`, which is a different formula that only applies a penalty after age 60.

  **Fix applied to `normalization.py`:**
  - `_threshold_k2()` updated to: `max(0.3, 1.2 − (0.005 × age))`.
  - Lower bound changed from 0.5 to 0.3 to match the linear formula's natural range.
  - `_threshold_k4()` updated to match spec formula: `25.0 + (0.2 × max(age, 30))` — replaces the decade-step formula previously used.

- **Notes:**
  - The technical specification (Chapter 4, บทที่ 4) was correct throughout; the bugs were implementation gaps, not design errors.
  - The `stiff_x`/`stiff_y` fields are backward-compatible additions to the stroke summary dict; no other modules are affected.
  - Unit tests in `tests/test_dcdt.py` that reference `compute_k1_rms` will need their call signature updated from `smoothed_x/y` to `stiff_x/y` to reflect the corrected interface.