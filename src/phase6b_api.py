"""
Phase 6b — FastAPI serving layer

Given a live match state, returns:
  - predicted win probability (from the trained XGBoost model)
  - the swing vs. the previous ball (if previous state provided)
  - a natural-language explanation of that swing (Groq)

Run: .venv/Scripts/uvicorn src.phase6b_api:app --reload
Then test at http://127.0.0.1:8000/docs
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal

from .phase6a_explanation import generate_explanation

app = FastAPI(title="crickcast — win probability + explanation API")

# ---------- CORS ----------
# The frontend (Vercel) and backend (Render) live on different origins, so
# without this the browser blocks every /predict and /health_check call.
# Set ALLOWED_ORIGINS as a comma-separated env var in production (e.g.
# "https://crickcast-sooty.vercel.app"); falls back to a permissive dev
# default so local `npm run dev` keeps working out of the box.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL = joblib.load("models/main_model.joblib")

MAIN_FEATURES = [
    "cum_runs", "cum_wickets", "balls_remaining", "runs_required",
    "required_run_rate", "current_run_rate", "recent_run_rate",
    "batting_team_prior", "bowling_team_prior",
]


class MatchState(BaseModel):
    # Bounds here are physical/rulebook limits, not just types — this is a
    # public endpoint, and the frontend always sends sane values, but
    # nothing previously stopped a direct request with e.g.
    # batting_team_prior=-50 from reaching the model. Every field is
    # bounded to what's actually possible in a T20 innings.
    cum_runs: int = Field(ge=0, le=500, description="Runs scored so far this innings")
    cum_wickets: int = Field(ge=0, le=10, description="Wickets down")
    balls_remaining: int = Field(ge=0, le=120, description="Legal deliveries left in a 20-over innings")
    runs_required: int = Field(ge=0, le=500, description="Runs still needed to win")
    # required_run_rate can hit the 99.0 sentinel Phase 1 uses when
    # balls_remaining is 0 but the chase wasn't yet decided by the time a
    # row was recorded — le=100 leaves headroom for that without allowing
    # arbitrary values.
    required_run_rate: float = Field(ge=0, le=100)
    current_run_rate: float = Field(ge=0, le=60)
    recent_run_rate: float = Field(ge=0, le=60)
    batting_team_prior: float = Field(default=0.5, ge=0, le=1)
    bowling_team_prior: float = Field(default=0.5, ge=0, le=1)
    # Restricted to the model's actual 5 training categories (see Phase 1 /
    # phase5_event_impact.py) rather than an open string — anything else
    # isn't a category the model or the explanation prompt know how to use.
    event_type: Literal["dot_ball", "four", "six", "wicket", "other_runs"] = "other_runs"
    previous_proba: Optional[float] = Field(default=None, ge=0, le=1)
    # Optional — only used to personalize the Groq explanation (e.g. "India
    # slides to 47%" instead of "the batting team slides to 47%"). Never
    # used as a model feature: the model only ever sees the leakage-safe
    # rolling priors above, never raw team identity, by design (see
    # phase2_team_priors.py). A request that omits these still works
    # exactly as before — generate_explanation falls back to generic
    # phrasing when they're not provided.
    batting_team: Optional[str] = None
    bowling_team: Optional[str] = None


@app.post("/predict")
def predict(state: MatchState):
    # A chase is only ever a *probability* while the outcome is still
    # genuinely uncertain. Once the target is reached, the batting side is
    # bowled out, or the overs run out, the result is a fact, not something
    # to ask the model for — and Cricsheet's own deliveries stop the moment
    # any of these happen, so the model has never once been trained on a
    # ball bowled past that point. Asking it to score one anyway means
    # extrapolating into a state it's never seen, which is exactly how you
    # get an XGBoost model reporting something like 84% for a chase that's
    # already been won. Short-circuit to the deterministic answer instead.
    if state.runs_required <= 0:
        proba = 1.0
    elif state.cum_wickets >= 10 or state.balls_remaining <= 0:
        proba = 0.0
    else:
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
            "runs_required": state.runs_required,
            "required_run_rate": state.required_run_rate,
            "batting_team": state.batting_team,
            "bowling_team": state.bowling_team,
        }
        # generate_explanation degrades to None (rather than raising) if the
        # Groq call fails — a flaky/down explanation service shouldn't take
        # out a response that already has a perfectly good win_probability.
        result["explanation"] = generate_explanation(explanation_context)

    return result


@app.get("/")
@app.get("/health_check")
def health_check():
    return {"status": "ok", "message": "crickcast API is running"}