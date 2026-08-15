"""
Pydantic Field constraints on MatchState (phase6b_api.py). A malformed
request should be rejected with a 422 before it ever reaches the model —
previously nothing stopped e.g. batting_team_prior=-50 or cum_wickets=15
from being scored, silently.
"""

import pytest


def _base_payload(**overrides):
    payload = dict(
        cum_runs=100, cum_wickets=3, balls_remaining=48, runs_required=60,
        required_run_rate=7.5, current_run_rate=8.0, recent_run_rate=7.5,
        batting_team_prior=0.5, bowling_team_prior=0.5,
    )
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("field,bad_value", [
    ("cum_runs", -1),
    ("cum_wickets", 11),
    ("cum_wickets", -1),
    ("balls_remaining", 121),
    ("balls_remaining", -1),
    ("runs_required", -1),
    ("batting_team_prior", -0.1),
    ("batting_team_prior", 1.1),
    ("bowling_team_prior", 50),
    ("required_run_rate", -5),
    ("current_run_rate", -1),
    ("recent_run_rate", -1),
    ("previous_proba", 1.5),
    ("previous_proba", -0.2),
])
def test_out_of_range_value_rejected(client, field, bad_value):
    r = client.post("/predict", json=_base_payload(**{field: bad_value}))
    assert r.status_code == 422


def test_bad_event_type_rejected(client):
    r = client.post("/predict", json=_base_payload(event_type="not_a_real_event"))
    assert r.status_code == 422


def test_all_valid_event_types_accepted(client):
    for event_type in ["dot_ball", "four", "six", "wicket", "other_runs"]:
        r = client.post("/predict", json=_base_payload(event_type=event_type))
        assert r.status_code == 200


def test_missing_required_field_rejected(client):
    payload = _base_payload()
    del payload["cum_runs"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
