"""
Diagnostic — targeted calibration check for extreme "death overs" states

The overall calibration curve (Phase 4) averages across ALL match states.
That can hide a real problem in a specific, rare corner of the state space
-- e.g. very high required run rate with few balls left -- where the model
might be poorly calibrated even if it looks great on average.

This filters the validation set to states similar to a reported scenario
and checks: does the model's average predicted probability match the
ACTUAL observed win rate for balls in that same tight corner?
"""

import pandas as pd

OUT_DIR = "data/processed"

# Tune these to bracket the scenario you're checking.
# Example values below match: RRR ~18-23, balls_remaining <= 8, wickets 4-6
MIN_RRR = 18
MAX_RRR = 23
MAX_BALLS_REMAINING = 8
MIN_WICKETS = 4
MAX_WICKETS = 6

if __name__ == "__main__":
    df = pd.read_csv(f"{OUT_DIR}/val_with_predictions.csv")

    subset = df[
        (df["required_run_rate"] >= MIN_RRR) &
        (df["required_run_rate"] <= MAX_RRR) &
        (df["balls_remaining"] <= MAX_BALLS_REMAINING) &
        (df["cum_wickets"] >= MIN_WICKETS) &
        (df["cum_wickets"] <= MAX_WICKETS)
    ]

    print(f"Matching balls in validation set: {len(subset)}")
    print(f"Matching unique matches: {subset['match_id'].nunique()}")

    if len(subset) == 0:
        print("No matching rows — widen the filter ranges above and re-run.")
    else:
        actual_win_rate = subset["batting_team_won"].mean()
        mean_predicted = subset["main_model_proba"].mean()

        print(f"\nActual observed win rate (batting/chasing team): {actual_win_rate:.3f}")
        print(f"Model's mean predicted probability:                {mean_predicted:.3f}")
        print(f"Gap:                                                {abs(actual_win_rate - mean_predicted):.3f}")

        # Also show the distribution, not just the mean — a wide spread matters
        print("\nPredicted probability distribution in this subset:")
        print(subset["main_model_proba"].describe())
