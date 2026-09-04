"""
Phase 6b — FastAPI serving layer

Given a live match state, returns the predicted win probability (from the
trained XGBoost model) and the swing vs. the previous ball. A separate
/explain endpoint generates the natural-language commentary (Groq) — see
the comment above /predict for why these are two endpoints, not one.

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

    proba = round(proba, 4)
    result = {"win_probability": proba}

    if state.previous_proba is not None:
        result["swing"] = round(proba - state.previous_proba, 4)

    # No Groq call here anymore — see /explain below. Deliberately kept
    # this way: the win probability is the one number every other part of
    # the UI (the scoreboard, the chart, the swing badge) depends on, and
    # it comes from a local model that answers in milliseconds. The Groq
    # commentary call, by contrast, is a third-party network request that
    # can take anywhere from under a second to several seconds — longer
    # still if Groq's free-tier rate limit (30 requests/minute) has been
    # hit and the client is mid-retry. Bundling both into one response
    # meant ANY Groq slowness froze the entire UI, including the
    # probability number that had nothing to do with it. Splitting them
    # means /predict is always fast, and a slow/failed /explain call only
    # ever delays or blanks the commentary panel specifically.
    return result


class ExplainRequest(BaseModel):
    """Everything generate_explanation() needs, sent directly by the
    frontend once it already has both probabilities from /predict — this
    endpoint does no model inference itself, it only talks to Groq."""
    event_type: Literal["dot_ball", "four", "six", "wicket", "other_runs"] = "other_runs"
    proba_before: float = Field(ge=0, le=1)
    proba_after: float = Field(ge=0, le=1)
    swing: float = Field(ge=-1, le=1)
    cum_runs: int = Field(ge=0, le=500)
    cum_wickets: int = Field(ge=0, le=10)
    balls_remaining: int = Field(ge=0, le=120)
    runs_required: int = Field(ge=0, le=500)
    required_run_rate: float = Field(ge=0, le=100)
    # Optional — only used to personalize the commentary (e.g. "India
    # slides to 47%" instead of "the batting team slides to 47%").
    batting_team: Optional[str] = None
    bowling_team: Optional[str] = None
    # "Before this ball" snapshot of the fields that actually move ball to
    # ball. Lets the prompt cite only what actually changed instead of
    # guessing from a single end-state snapshot — see phase6a_explanation.py
    # for the bug this fixes (citing an unchanged number, e.g. "0 wickets
    # down", as if it explained a swing it had nothing to do with).
    cum_runs_before: Optional[int] = Field(default=None, ge=0, le=500)
    cum_wickets_before: Optional[int] = Field(default=None, ge=0, le=10)
    balls_remaining_before: Optional[int] = Field(default=None, ge=0, le=120)
    required_run_rate_before: Optional[float] = Field(default=None, ge=0, le=100)
    # True only when this ball was the 6th legal delivery of an over.
    over_just_completed: bool = False
    # How many legal deliveries elapsed since the last commentary request —
    # lets the prompt know whether it's narrating one discrete ball or a
    # whole span of play (Auto-predict off, or a large Alter State edit).
    balls_elapsed: Optional[int] = Field(default=None, ge=0, le=120)
    # The literal delivery clicked, distinct from event_type above — that
    # field collapses single/two/three/wide/noball all into "other_runs"
    # for the model's categorical feature. This is what lets the prompt
    # tell a wide from a no-ball, which event_type alone can't.
    raw_event: Optional[Literal[
        "dot_ball", "single", "two", "three", "four", "six", "wicket", "wide", "noball"
    ]] = None


@app.post("/explain")
def explain(req: ExplainRequest):
    """Separate from /predict on purpose (see the comment in /predict) —
    the frontend calls this AFTER already showing the new win probability,
    so a slow or rate-limited Groq call only ever delays the commentary
    panel specifically."""
    context = {
        "event_type": req.event_type,
        "proba_before": req.proba_before,
        "proba_after": req.proba_after,
        "swing": req.swing,
        "cum_runs": req.cum_runs,
        "cum_wickets": req.cum_wickets,
        "balls_remaining": req.balls_remaining,
        "runs_required": req.runs_required,
        "required_run_rate": req.required_run_rate,
        "batting_team": req.batting_team,
        "bowling_team": req.bowling_team,
        "cum_runs_before": req.cum_runs_before,
        "cum_wickets_before": req.cum_wickets_before,
        "balls_remaining_before": req.balls_remaining_before,
        "required_run_rate_before": req.required_run_rate_before,
        "over_just_completed": req.over_just_completed,
        "balls_elapsed": req.balls_elapsed,
        "raw_event": req.raw_event,
    }
    # generate_explanation degrades to None (rather than raising) if the
    # Groq call fails — a flaky/down/rate-limited explanation service
    # should never turn into a 500 here.
    return {"explanation": generate_explanation(context)}


@app.get("/")
@app.get("/health_check")
def health_check():
    return {"status": "ok", "message": "crickcast API is running"}