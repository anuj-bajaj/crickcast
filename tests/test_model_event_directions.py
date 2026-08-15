"""
For dot balls, wickets, wides, and no-balls specifically, every one of the
9 model features moves either favorably-or-neutral, or unfavorably-or-
neutral, for the batting team — never a genuine mix of both directions in
the same event (see phase3_modeling.py's MONOTONE_CONSTRAINTS comment).
That makes the model's response to these four event types a hard
mathematical guarantee given the constraints hold, not just an empirical
tendency: the predicted win probability must never move the wrong way,
for any match state.

Four/six/singles are deliberately NOT tested here with a zero-tolerance
assertion — they're only *expected* to be favorable in the large majority
of states (balls_remaining moving unfavorably is a genuine, sometimes-
dominant trade-off for those events), not guaranteed. See
BACKEND_HANDOVER.md for the full breakdown of that distinction and the
actual measured rates.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import MAIN_FEATURES

# event -> expected sign of the win-probability swing (+1 = must help the
# batting team or be neutral, -1 = must hurt or be neutral)
GUARANTEED_EVENTS = {
    "dot": -1,
    "wicket": -1,
    "wide": 1,
    "noball": 1,
}


def _random_states(n, seed=42):
    rng = np.random.default_rng(seed)
    balls_bowled = rng.integers(1, 119, n)
    balls_remaining = 120 - balls_bowled
    cum_wickets = rng.integers(0, 9, n)
    cum_runs = rng.integers(0, 200, n)
    target = cum_runs + rng.integers(1, 150, n)
    runs_required = target - cum_runs
    current_run_rate = (cum_runs / balls_bowled) * 6
    required_run_rate = np.where(balls_remaining > 0, (runs_required / np.maximum(balls_remaining, 1)) * 6, 99)
    recent_run_rate = np.clip(current_run_rate + rng.normal(0, 2, n), 0, 36)
    df = pd.DataFrame({
        "cum_runs": cum_runs, "cum_wickets": cum_wickets, "balls_remaining": balls_remaining,
        "runs_required": runs_required, "required_run_rate": required_run_rate,
        "current_run_rate": current_run_rate, "recent_run_rate": recent_run_rate,
        "batting_team_prior": rng.uniform(0.15, 0.85, n),
        "bowling_team_prior": rng.uniform(0.15, 0.85, n),
    })
    # Exclude already-decided states — the API short-circuits those (see
    # test_api_predict.py), so they aren't a meaningful model-behavior test.
    valid = (df["runs_required"] > 0) & (df["cum_wickets"] < 10) & (df["balls_remaining"] > 0)
    return df[valid].reset_index(drop=True)


def _apply_event(before, event):
    s = before.copy()
    is_illegal = event in ("wide", "noball")
    runs_added = {"dot": 0, "wide": 1, "noball": 1, "wicket": 0}[event]
    wicket_added = 1 if event == "wicket" else 0

    s["cum_runs"] = before["cum_runs"] + runs_added
    s["cum_wickets"] = np.minimum(10, before["cum_wickets"] + wicket_added)
    s["balls_remaining"] = before["balls_remaining"] if is_illegal else np.maximum(0, before["balls_remaining"] - 1)
    s["runs_required"] = np.maximum(0, before["runs_required"] - runs_added)

    balls_bowled = 120 - s["balls_remaining"]
    s["current_run_rate"] = np.where(balls_bowled > 0, (s["cum_runs"] / np.maximum(balls_bowled, 1)) * 6, 0)
    s["required_run_rate"] = np.where(
        s["balls_remaining"] > 0, (s["runs_required"] / np.maximum(s["balls_remaining"], 1)) * 6, 99
    )
    s["recent_run_rate"] = (
        before["recent_run_rate"] if is_illegal
        else np.clip(before["recent_run_rate"] * 0.8 + (runs_added * 6) * 0.2, 0, 36)
    )
    return s


@pytest.mark.parametrize("event,expected_sign", GUARANTEED_EVENTS.items())
def test_guaranteed_event_direction(model, event, expected_sign):
    before = _random_states(2000)
    after = _apply_event(before, event)

    p_before = model.predict_proba(before[MAIN_FEATURES])[:, 1]
    p_after = model.predict_proba(after[MAIN_FEATURES])[:, 1]
    swing = p_after - p_before

    violations = swing * expected_sign < -1e-6
    n_bad = int(violations.sum())
    worst = float((-swing[violations] * expected_sign).max()) if n_bad else 0.0

    assert n_bad == 0, (
        f"'{event}' moved win probability the wrong way in {n_bad}/{len(swing)} "
        f"random match states (expected sign {expected_sign:+d}), worst case {worst:.4f}"
    )
