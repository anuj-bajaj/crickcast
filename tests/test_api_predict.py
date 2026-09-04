"""
End-to-end checks against the real FastAPI app: the decided-state
short-circuit in /predict and the response contract. See
test_api_explain.py for the commentary endpoint (/explain), which is
deliberately separate — a slow/down Groq call must never affect /predict.
"""


def _base_payload(**overrides):
    payload = dict(
        cum_runs=100, cum_wickets=3, balls_remaining=48, runs_required=60,
        required_run_rate=7.5, current_run_rate=8.0, recent_run_rate=7.5,
        batting_team_prior=0.5, bowling_team_prior=0.5,
    )
    payload.update(overrides)
    return payload


def test_health_check(client):
    r = client.get("/health_check")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_normal_state_returns_probability_in_range(client):
    r = client.post("/predict", json=_base_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    # No previous_proba supplied -> no swing/explanation computed at all
    assert "swing" not in body
    assert "explanation" not in body


def test_chase_won_short_circuits_to_one(client):
    """Once runs_required <= 0, the API must not ask the model at all —
    it's a certainty, not a probability. See phase6b_api.py's comment on
    why (the model has never seen a ball bowled after a chase is won)."""
    r = client.post("/predict", json=_base_payload(runs_required=0, previous_proba=0.8))
    assert r.status_code == 200
    assert r.json()["win_probability"] == 1.0


def test_all_out_short_circuits_to_zero(client):
    r = client.post("/predict", json=_base_payload(cum_wickets=10, previous_proba=0.3))
    assert r.status_code == 200
    assert r.json()["win_probability"] == 0.0


def test_overs_complete_short_circuits_to_zero(client):
    r = client.post("/predict", json=_base_payload(balls_remaining=0, previous_proba=0.2))
    assert r.status_code == 200
    assert r.json()["win_probability"] == 0.0


def test_swing_present_no_explanation_in_predict(client):
    """/predict computes swing (it needs both probabilities, which it
    already has), but no longer generates commentary — that moved to
    /explain, deliberately, so a slow/rate-limited Groq call can never
    delay the win-probability response. See phase6b_api.py's comment on
    /predict for the full reasoning."""
    r = client.post("/predict", json=_base_payload(previous_proba=0.5))
    assert r.status_code == 200
    body = r.json()
    assert "swing" in body
    assert body["swing"] == round(body["win_probability"] - 0.5, 4)
    assert "explanation" not in body


def test_team_names_are_optional(client):
    """Personalized-commentary fields must not be required — omitting
    them is the pre-existing, still-valid request shape."""
    r = client.post("/predict", json=_base_payload(previous_proba=0.5))
    assert r.status_code == 200

    r2 = client.post("/predict", json=_base_payload(
        previous_proba=0.5, batting_team="India", bowling_team="Australia",
    ))
    assert r2.status_code == 200