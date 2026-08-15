"""
End-to-end checks against the real FastAPI app: the decided-state
short-circuit in /predict, the response contract, and graceful
degradation when the Groq explanation service is unavailable.
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


def test_swing_and_explanation_keys_present_with_previous_proba(client):
    r = client.post("/predict", json=_base_payload(previous_proba=0.5))
    assert r.status_code == 200
    body = r.json()
    assert "swing" in body
    assert body["swing"] == round(body["win_probability"] - 0.5, 4)
    # "explanation" key must exist even if its value is null (Groq down /
    # no key) — see test below. A missing key vs. a null value is exactly
    # the kind of contract drift a frontend integration would silently
    # break on.
    assert "explanation" in body


def test_explanation_degrades_gracefully_without_groq_key(client, monkeypatch):
    """If GROQ_API_KEY is missing or invalid, /predict must still return
    200 with a valid win_probability — explanation should be null, never
    a 500. This is the fix for the original bug where a Groq outage took
    down a response that already had a perfectly good prediction.

    generate_explanation() caches its Groq client in a module-level
    singleton (phase6a_explanation._client) so it isn't rebuilt on every
    single call — sensible for the real app, but it means once an earlier
    test in this same session has already built a real client against a
    real, working key, just deleting the env var here doesn't force a
    rebuild; the cached client gets reused regardless. Resetting the
    singleton to None first is what actually forces _get_client() to
    re-check the environment and hit the missing-key path this test is
    meant to exercise.
    """
    from src import phase6a_explanation
    monkeypatch.setattr(phase6a_explanation, "_client", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/predict", json=_base_payload(previous_proba=0.5))
    assert r.status_code == 200
    assert r.json()["explanation"] is None


def test_team_names_are_optional(client):
    """Personalized-commentary fields must not be required — omitting
    them is the pre-existing, still-valid request shape."""
    r = client.post("/predict", json=_base_payload(previous_proba=0.5))
    assert r.status_code == 200

    r2 = client.post("/predict", json=_base_payload(
        previous_proba=0.5, batting_team="India", bowling_team="Australia",
    ))
    assert r2.status_code == 200