"""
Phase 6b — FastAPI serving layer

Given a live match state, returns:
  - predicted win probability (from the trained XGBoost model)
  - the swing vs. the previous ball (if previous state provided)
  - a natural-language explanation of that swing (Groq)

Run: .venv/Scripts/uvicorn src.phase6b_api:app --reload
Then test at http://127.0.0.1:8000/docs
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .phase6a_explanation import generate_explanation

app = FastAPI(title="crickcast — win probability + explanation API")

MODEL = joblib.load("models/main_model.joblib")

MAIN_FEATURES = [
    "cum_runs", "cum_wickets", "balls_remaining", "runs_required",
    "required_run_rate", "current_run_rate", "recent_run_rate",
    "batting_team_prior", "bowling_team_prior",
]


class MatchState(BaseModel):
    cum_runs: int
    cum_wickets: int
    balls_remaining: int
    runs_required: int
    required_run_rate: float
    current_run_rate: float
    recent_run_rate: float
    batting_team_prior: float = 0.5
    bowling_team_prior: float = 0.5
    event_type: str = "other_runs"          # for explanation context
    previous_proba: Optional[float] = None  # if known, enables swing + explanation


@app.post("/predict")
def predict(state: MatchState):
    features = pd.DataFrame([{k: getattr(state, k) for k in MAIN_FEATURES}])
    proba = float(MODEL.predict_proba(features)[0, 1])

    result = {"win_probability": round(proba, 4)}

    if state.previous_proba is not None:
        swing = proba - state.previous_proba
        result["swing"] = round(swing, 4)

        explanation_context = {
            "event_type": state.event_type,
            "proba_before": state.previous_proba,
            "proba_after": proba,
            "swing": swing,
            "cum_runs": state.cum_runs,
            "cum_wickets": state.cum_wickets,
            "balls_remaining": state.balls_remaining,
            "required_run_rate": state.required_run_rate,
        }
        result["explanation"] = generate_explanation(explanation_context)

    return result


@app.get("/")
def health_check():
    return {"status": "ok", "message": "crickcast API is running"}