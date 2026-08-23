from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "lead_score_model.joblib"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        model = None
    yield


app = FastAPI(
    title="Lead Score API",
    description=(
        "Scores inbound leads by conversion likelihood, trained on a real "
        "9,200-lead marketing dataset. Bring your own lead data to retrain "
        "for production use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Lead(BaseModel):
    lead_origin: str = Field(..., examples=["Landing Page Submission"])
    lead_source: str = Field(..., examples=["Google"])
    do_not_email: Literal["Yes", "No"] = "No"
    total_visits: float = Field(..., ge=0, examples=[5])
    total_time_spent_on_website: float = Field(..., ge=0, examples=[720])
    page_views_per_visit: float = Field(..., ge=0, examples=[2.5])
    last_activity: str = Field(..., examples=["Email Opened"])
    specialization: str | None = Field(default=None, examples=["Business Administration"])
    current_occupation: str | None = Field(default=None, examples=["Working Professional"])


class ScoreResponse(BaseModel):
    conversion_probability: float
    score_band: Literal["Hot", "Warm", "Cold"]


def band_for(probability: float) -> str:
    if probability >= 0.66:
        return "Hot"
    if probability >= 0.33:
        return "Warm"
    return "Cold"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/score", response_model=ScoreResponse)
def score_lead(lead: Lead):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first.")

    row = pd.DataFrame(
        [
            {
                "TotalVisits": lead.total_visits,
                "Total Time Spent on Website": lead.total_time_spent_on_website,
                "Page Views Per Visit": lead.page_views_per_visit,
                "Lead Origin": lead.lead_origin,
                "Lead Source": lead.lead_source,
                "Do Not Email": lead.do_not_email,
                "Last Activity": lead.last_activity,
                "Specialization": lead.specialization or "Unknown",
                "What is your current occupation": lead.current_occupation or "Unknown",
            }
        ]
    )
    probability = float(model.predict_proba(row)[0, 1])
    return ScoreResponse(conversion_probability=round(probability, 4), score_band=band_for(probability))
