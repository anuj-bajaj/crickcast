"""
/explain — the commentary endpoint, split out from /predict so a slow or
rate-limited Groq call can never delay the win-probability response (see
phase6b_api.py's comment on /predict for the full reasoning). These tests
mirror the same graceful-degradation contract /predict used to guarantee
for its bundled explanation: a down/missing-key Groq service must return
200 with explanation: null, never a 500.
"""


def _base_payload(**overrides):
    payload = dict(
        event_type="wicket",
        proba_before=0.62,
        proba_after=0.47,
        swing=-0.15,
        cum_runs=88,
        cum_wickets=5,
        balls_remaining=24,
        runs_required=40,
        required_run_rate=11.25,
    )
    payload.update(overrides)
    return payload


def test_explain_returns_explanation_key(client):
    r = client.post("/explain", json=_base_payload())
    assert r.status_code == 200
    # "explanation" key must exist even if its value is null (Groq down /
    # rate-limited / no key in this environment) — a missing key vs. a
    # null value is exactly the kind of contract drift a frontend
    # integration would silently break on.
    assert "explanation" in r.json()


def test_explain_degrades_gracefully_when_all_providers_unavailable(client, monkeypatch):
    """If neither GROQ_API_KEY nor OPENROUTER_API_KEY is usable, /explain
    must still return 200 with explanation: null, never a 500 — a down/
    flaky explanation service should never look like a server error to
    the frontend. See test_explanation_fallback.py for the Groq-fails-
    but-OpenRouter-succeeds case, which this test deliberately does NOT
    cover (both providers are unavailable here, on purpose).

    generate_explanation() caches its Groq client in a module-level
    singleton (phase6a_explanation._client) so it isn't rebuilt on every
    call — sensible for the real app, but it means once an earlier test
    in this same session has already built a real client against a real,
    working key, just deleting the env var here doesn't force a rebuild;
    the cached client gets reused regardless. Resetting the singleton to
    None first is what actually forces _get_client() to re-check the
    environment and hit the missing-key path this test is meant to
    exercise. OPENROUTER_API_KEY is cleared too — otherwise, on a machine
    where that key is genuinely configured, the fallback would kick in,
    reach the real network, and this test's result would depend on
    whatever OpenRouter happens to return that moment rather than
    reliably testing the "everything is unavailable" path.
    """
    from src import phase6a_explanation
    monkeypatch.setattr(phase6a_explanation, "_client", None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = client.post("/explain", json=_base_payload())
    assert r.status_code == 200
    assert r.json()["explanation"] is None


def test_explain_team_names_optional(client):
    """Personalized-commentary fields must not be required — omitting
    them is a valid request shape."""
    r = client.post("/explain", json=_base_payload())
    assert r.status_code == 200

    r2 = client.post("/explain", json=_base_payload(
        batting_team="India", bowling_team="Australia",
    ))
    assert r2.status_code == 200


def test_explain_before_fields_optional(client):
    """The before/after snapshot fields all improve commentary quality
    but must degrade gracefully if omitted — same contract as before."""
    r = client.post("/explain", json=_base_payload())
    assert r.status_code == 200


def test_explain_rejects_out_of_range_proba(client):
    r = client.post("/explain", json=_base_payload(proba_after=1.5))
    assert r.status_code == 422


def test_explain_rejects_missing_required_field(client):
    payload = _base_payload()
    del payload["proba_after"]
    r = client.post("/explain", json=payload)
    assert r.status_code == 422