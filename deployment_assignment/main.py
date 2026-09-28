"""
Car Insurance Risk Prediction API
==================================
Endpoint : POST /predict
Input    : age, driving_experience, accidents_last_5_years, vehicle_age, annual_km
Output   : risk_level (Low | Medium | High) + confidence score

Run:
    uvicorn main:app --reload
Docs:
    http://127.0.0.1:8000/docs
"""

import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.pkl"
FEATURE_COLS = [
    "age",
    "driving_experience",
    "accidents_last_5_years",
    "vehicle_age",
    "annual_km",
]

# ── Global model bundle (loaded once at startup) ───────────────────────────────
_bundle: dict[str, Any] = {}


# ── Lifespan: load model on startup, release on shutdown ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from {}", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}. Run train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        _bundle.update(pickle.load(f))
    logger.info("Model loaded successfully. Classes: {}", list(_bundle["label_encoder"].classes_))
    yield
    _bundle.clear()
    logger.info("Server shut down. Model released.")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Car Insurance Risk Prediction API",
    description=(
        "Predicts car insurance risk level (Low / Medium / High) "
        "based on driver and vehicle attributes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Input features for insurance risk prediction."""

    age: int = Field(..., ge=18, le=80, description="Driver age in years", example=35)
    driving_experience: int = Field(
        ..., ge=0, le=60, description="Years of driving experience", example=10
    )
    accidents_last_5_years: int = Field(
        ..., ge=0, le=10, description="Number of accidents in the last 5 years", example=1
    )
    vehicle_age: int = Field(
        ..., ge=0, le=30, description="Age of the vehicle in years", example=4
    )
    annual_km: int = Field(
        ..., ge=1000, le=200_000, description="Estimated kilometers driven per year", example=25000
    )


class PredictResponse(BaseModel):
    """Prediction output."""

    risk_level: str = Field(..., description="Predicted insurance risk: Low | Medium | High")
    confidence: float = Field(..., description="Model confidence score (0.0 – 1.0)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root() -> dict:
    """API health check."""
    return {"status": "ok", "message": "Car Insurance Risk API is running. Visit /docs to test."}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict car insurance risk level.

    - **Low** : Safe driver, low exposure — minimal premium
    - **Medium** : Moderate risk — standard premium
    - **High** : High-risk profile — elevated premium required
    """
    if not _bundle:
        raise HTTPException(status_code=503, detail="Model not loaded. Server may still be starting.")

    t0 = time.perf_counter()
    try:
        model = _bundle["model"]
        le = _bundle["label_encoder"]

        import pandas as pd
        features = pd.DataFrame([{
            "age": request.age,
            "driving_experience": request.driving_experience,
            "accidents_last_5_years": request.accidents_last_5_years,
            "vehicle_age": request.vehicle_age,
            "annual_km": request.annual_km,
        }])

        pred_idx = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        risk_level = le.inverse_transform([pred_idx])[0]
        confidence = round(float(proba[pred_idx]), 4)

    except Exception as exc:
        logger.error("Prediction failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "Predicted risk={} confidence={} latency={}ms",
        risk_level, confidence, latency_ms
    )

    return PredictResponse(
        risk_level=risk_level,
        confidence=confidence,
        latency_ms=latency_ms,
    )
