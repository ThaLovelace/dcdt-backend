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
    4. **Module Organization:** The current flat structure in `main.py` will become messy with the new logic. It's recommended to organize processing logic into dedicated modules within `core/` and API routes into `api/`.

### [2026-04-03] — Milestone 2 Completion: Data Schema & API Endpoint
- **Files Modified/Created:** `docs/PROJECT_CONTEXT.md`, `models/schemas.py`, `api/routes.py`, `main.py`, `docs/TASK_BOARD.md`
- **Description:** Updated the project context to reflect the new 7-field high-fidelity JSON payload. Created Pydantic schemas (`StrokePoint`, `DrawingPayload`) in `models/schemas.py` to rigorously validate incoming data. Created the `/api/analyze` endpoint in `api/routes.py` to receive the payload and integrated the router cleanly into `main.py`.
- **Notes:** The backend is now fully capable of parsing the frontend's output without 422 Unprocessable Entity errors. Ready to begin Milestone 3 (Savitzky-Golay signal processing).

4/4/2026: Updated `requirements.txt` to include `python-multipart` for file uploads. Explained how to run `main.py` using Uvicorn.