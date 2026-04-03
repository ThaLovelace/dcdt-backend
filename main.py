from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="dCDT Backend API",
    description="Backend for the Digital Clock Drawing Test (dCDT). Receives raw stroke data and processes it through a signal processing and ML inference pipeline.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class StrokePoint(BaseModel):
    """Represents a single point captured during the clock drawing."""
    x: float
    y: float
    t: float        # timestamp in milliseconds
    pressure: float


class DrawingPayload(BaseModel):
    """Full drawing payload sent from the frontend."""
    strokes: List[StrokePoint]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the server is running."""
    return {"message": "Hello from dCDT Backend! 🕐", "status": "ok"}


@app.post("/test", tags=["Test"])
def test_endpoint(payload: DrawingPayload):
    """
    Test POST endpoint.
    Accepts a drawing payload and echoes back a summary.
    Used to verify that the server correctly receives and parses stroke data.
    """
    point_count = len(payload.strokes)
    return {
        "message": "Payload received successfully.",
        "point_count": point_count,
        "first_point": payload.strokes[0] if point_count > 0 else None,
        "last_point": payload.strokes[-1] if point_count > 0 else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
