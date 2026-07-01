"""
MedPredict AI — FastAPI Backend
Compatible with: local uvicorn server AND Vercel serverless deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys, os

# ── Resolve paths (works locally and on Vercel) ───────────────────────────────
# On Vercel: files live under /var/task/ (project root)
# Locally: __file__ is at <project>/api/main.py
_api_dir     = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_api_dir)
_src_dir     = os.path.join(_project_dir, "src")

# Make sure src/ is importable
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from predict import predict, load_artifacts

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedPredict AI",
    description="Multi-disease risk prediction from 19 clinical biomarkers.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Schema ─────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    Age:                      float = Field(..., ge=0,   le=120)
    Gender:                   str   = Field(...)
    BMI:                      float = Field(..., ge=10,  le=70)
    Blood_Pressure_Systolic:  float = Field(..., ge=60,  le=300)
    Blood_Pressure_Diastolic: float = Field(..., ge=40,  le=200)
    Glucose:                  float = Field(..., ge=40,  le=600)
    HbA1c:                    float = Field(..., ge=3.0, le=15.0)
    Cholesterol:              float = Field(..., ge=50,  le=500)
    HDL_Cholesterol:          float = Field(..., ge=5,   le=150)
    LDL_Cholesterol:          float = Field(..., ge=10,  le=400)
    Triglycerides:            float = Field(..., ge=20,  le=1000)
    Creatinine:               float = Field(..., ge=0.1, le=20.0)
    ALT:                      float = Field(..., ge=1,   le=500)
    AST:                      float = Field(..., ge=1,   le=500)
    Smoking:                  int   = Field(..., ge=0, le=1)
    Physical_Activity:        int   = Field(..., ge=1, le=4)
    Alcohol_Use:              int   = Field(..., ge=0, le=2)
    Family_History_Diabetes:  int   = Field(..., ge=0, le=1)
    Family_History_Heart:     int   = Field(..., ge=0, le=1)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
        print("Models loaded successfully on startup.")
    except Exception as e:
        print(f"WARNING: Could not pre-load models: {e}")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "3.0.0", "service": "MedPredict AI"}


@app.post("/predict")
@app.post("/api/predict")
def predict_disease(data: PatientData):
    try:
        result = predict(data.dict())
        return {"success": True, "predictions": result}
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Models not ready: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Vercel handler ─────────────────────────────────────────────────────────────
# Vercel's @vercel/python runtime expects the app at module level
# The `app` variable above IS the ASGI app — no extra handler needed
