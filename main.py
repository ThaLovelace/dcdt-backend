from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as analyze_router

app = FastAPI(title="dCDT Backend API")

# Configure CORS explicitly to allow the frontend connection
app.add_middleware(
    CORSMiddleware,
    # Replace "*" with the EXACT URL of your frontend(s)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router for the analysis endpoints
# (This perfectly matches the frontend's /analyze endpoint)
app.include_router(analyze_router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "dCDT Backend is running"}