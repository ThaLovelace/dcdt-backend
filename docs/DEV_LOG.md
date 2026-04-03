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
