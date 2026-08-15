"""
Verifies the trained model actually respects the monotonicity constraints
it was trained with (see phase3_modeling.py's MONOTONE_CONSTRAINTS and the
comment above it — this is what fixed the "a four decreased win
probability in 14.6% of match states" bug).

XGBoost enforces this structurally at tree-build time when
monotone_constraints is set, so a violation here means either the wrong
model file is loaded, or a future retrain dropped the constraints — not
"the model got a little worse." This should always pass exactly; it isn't
a statistical/tolerance check.

Scaled down from the 4,800-comparisons-per-feature manual audit used to
originally find and verify-fix the bug (200 base states x 25 sweep points)
to keep the test suite fast; still large enough to catch a real
regression, just not exhaustive.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import MAIN_FEATURES

# +1 = predicted P(batting team wins) must be non-decreasing in that
# feature; -1 = non-increasing. Must match phase3_modeling.py exactly.
CONSTRAINTS = dict(zip(MAIN_FEATURES, (1, -1, 1, -1, -1, 1, 1, 1, -1)))

RANGES = {
    "cum_runs": (0, 220), "cum_wickets": (0, 9), "balls_remaining": (1, 119),
    "runs_required": (0, 220), "required_run_rate": (0, 36), "current_run_rate": (0, 36),
    "recent_run_rate": (0, 36), "batting_team_prior": (0.0, 1.0), "bowling_team_prior": (0.0, 1.0),
}


def _random_base_states(n, seed=7):
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
    return pd.DataFrame({
        "cum_runs": cum_runs, "cum_wickets": cum_wickets, "balls_remaining": balls_remaining,
        "runs_required": runs_required, "required_run_rate": required_run_rate,
        "current_run_rate": current_run_rate, "recent_run_rate": recent_run_rate,
        "batting_team_prior": rng.uniform(0.15, 0.85, n),
        "bowling_team_prior": rng.uniform(0.15, 0.85, n),
    })


@pytest.mark.parametrize("feature", MAIN_FEATURES)
def test_feature_is_monotonic(model, feature):
    """Sweeping any single feature across its full range, holding a random
    base state's other 8 features fixed, must never move the predicted
    probability in the direction that feature's constraint forbids."""
    lo, hi = RANGES[feature]
    base_states = _random_base_states(200)
    sweep_vals = np.linspace(lo, hi, 25)
    expected_sign = CONSTRAINTS[feature]

    worst_violation = 0.0
    n_violations = 0
    n_pairs = 0

    for _, base in base_states.iterrows():
        rows = pd.DataFrame([base.to_dict()] * len(sweep_vals))
        rows[feature] = sweep_vals
        preds = model.predict_proba(rows[MAIN_FEATURES])[:, 1]
        diffs = np.diff(preds)
        n_pairs += len(diffs)

        bad = diffs * expected_sign < -1e-6
        if bad.any():
            n_violations += bad.sum()
            worst_violation = max(worst_violation, float((-diffs[bad] * expected_sign).max()))

    assert n_violations == 0, (
        f"{feature} (constraint {expected_sign:+d}) violated monotonicity in "
        f"{n_violations}/{n_pairs} sweep steps, worst violation "
        f"{worst_violation:.4f} probability points"
    )
