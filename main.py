from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as analyze_router

app = FastAPI(title="dCDT Backend API")

# Configure CORS to allow requests from the frontend (e.g., http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router for the analysis endpoints
app.include_router(analyze_router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "dCDT Backend is running"}