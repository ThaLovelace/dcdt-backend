# Task Board

> Tracks the status of all development tasks for the dCDT backend project.

---

## Status Legend

| Symbol | Meaning     |
|--------|-------------|
| 🔲     | To Do       |
| 🔄     | In Progress |
| ✅     | Done        |
| ❌     | Blocked     |

---

## Milestone 1 — Project Setup

| Status | Task                                      |
|--------|-------------------------------------------|
| ✅     | Initialize project directory structure    |
| ✅     | Create `requirements.txt`                 |
| ✅     | Create `.clinerules`                      |
| ✅     | Create `docs/PROJECT_CONTEXT.md`          |
| ✅     | Create `docs/DEV_LOG.md`                  |
| ✅     | Create `docs/TASK_BOARD.md`               |
| ✅     | Create `main.py` with Hello World + POST  |

---

## Milestone 2 — Data Ingestion & Validation

| Status | Task                                                         |
|--------|--------------------------------------------------------------|
| ✅     | Define Pydantic schema for raw stroke input                  |
| ✅     | Create POST `/api/analyze` endpoint                          |
| ✅     | Validate incoming `t`, `x`, `y`, `p`, `az`, `alt`, `id` data |

---

## Milestone 3 — Signal Processing

| Status | Task                                                  |
|--------|-------------------------------------------------------|
| ✅     | Implement Savitzky-Golay filter in `core/`            |
| ✅     | Implement K-Means clustering in `core/`               |
| ✅     | Implement K-Series feature extraction (K1–K5)         |

---

## Milestone 4 — Model Inference

| Status | Task                                                  |
|--------|-------------------------------------------------------|
| 🔲     | Integrate ViT-B/16 ONNX model                         |
| 🔲     | Create inference pipeline in `core/`                  |
| 🔲     | Return structured scoring result to frontend          |

---

## Milestone 5 — Testing & Documentation

| Status | Task                                                  |
|--------|-------------------------------------------------------|
| 🔲     | Write unit tests for processing pipeline              |
| 🔲     | Write API integration tests                           |
| 🔲     | Finalize academic report documentation                |
