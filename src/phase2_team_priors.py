"""
Phase 2 — Team strength prior (leakage-safe)

Cricsheet JSON per-match info includes a match date. We use that to build,
for every match, a "going in" win-rate prior for each team based ONLY on
that team's matches strictly before the current match's date. This gets
merged onto the ball-by-ball train/val/test CSVs from Phase 1.

Run from the project root, after phase1_data_pipeline.py has been run once.
"""

import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw_matches")
OUT_DIR = Path("data/processed")
ROLLING_WINDOW = 15  # last N matches used for the prior; tune later if needed

def extract_match_level_info():
    """Build a match_id -> {date, batting_team, bowling_team, winner} table."""
    records = []
    for fpath in RAW_DIR.glob("*.json"):
        match_id = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                match_json = json.load(f)

            info = match_json.get("info", {})
            dates = info.get("dates", [])
            if not dates:
                continue
            match_date = dates[0]  # start date, string 'YYYY-MM-DD'

            innings_list = match_json.get("innings", [])
            if len(innings_list) < 2:
                continue

            team1 = innings_list[0].get("team")  # batted 1st
            team2 = innings_list[1].get("team")  # batted 2nd (the chase)

            winner = info.get("outcome", {}).get("winner")
            if winner is None:
                continue

            records.append({
                "match_id": match_id,
                "date": match_date,
                "team1": team1,
                "team2": team2,
                "winner": winner,
            })
        except Exception as e:
            print(f"Skipping {match_id} in match-level extraction: {e}")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Match-level table: {len(df)} matches with valid date + result")
    return df


def compute_rolling_priors(match_df: pd.DataFrame):
    """
    For each match, compute each involved team's win rate over its last
    ROLLING_WINDOW matches STRICTLY BEFORE this match's date.
    Returns a dict: match_id -> {team1_prior, team2_prior}
    """
    # Build a long-format history: one row per (team, match, won_flag), sorted by date
    history_rows = []
    for _, row in match_df.iterrows():
        history_rows.append({"date": row["date"], "match_id": row["match_id"],
                              "team": row["team1"], "won": int(row["winner"] == row["team1"])})
        history_rows.append({"date": row["date"], "match_id": row["match_id"],
                              "team": row["team2"], "won": int(row["winner"] == row["team2"])})
    hist = pd.DataFrame(history_rows).sort_values("date").reset_index(drop=True)

    priors = {}  # (team, match_id) -> prior win rate at that point
    team_recent_results = {}  # team -> list of recent won flags (chronological)

    for _, row in hist.iterrows():
        team = row["team"]
        recent = team_recent_results.get(team, [])
        if len(recent) == 0:
            prior = 0.5  # no history yet — neutral prior
        else:
            window = recent[-ROLLING_WINDOW:]
            prior = sum(window) / len(window)
        priors[(team, row["match_id"])] = prior

        # update history AFTER computing this match's prior (so this match's
        # own result doesn't leak into its own prior)
        recent.append(row["won"])
        team_recent_results[team] = recent

    return priors


def attach_priors_to_ball_data(match_df: pd.DataFrame, priors: dict):
    """Build a match_id -> (batting_team_prior, bowling_team_prior) lookup.
    Note: team2 is always the batting (chasing) team per Phase 1's scope."""
    rows = []
    for _, row in match_df.iterrows():
        batting_prior = priors.get((row["team2"], row["match_id"]), 0.5)
        bowling_prior = priors.get((row["team1"], row["match_id"]), 0.5)
        rows.append({
            "match_id": row["match_id"],
            "batting_team_prior": round(batting_prior, 3),
            "bowling_team_prior": round(bowling_prior, 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    match_df = extract_match_level_info()
    priors = compute_rolling_priors(match_df)
    prior_lookup = attach_priors_to_ball_data(match_df, priors)

    for split in ["train", "val", "test"]:
        path = OUT_DIR / f"{split}.csv"
        df = pd.read_csv(path, dtype={"match_id": str})
        prior_lookup["match_id"] = prior_lookup["match_id"].astype(str)
        before_cols = set(df.columns)
        df = df.merge(prior_lookup, on="match_id", how="left")
        # Any match without a prior lookup (shouldn't normally happen) gets neutral 0.5
        df["batting_team_prior"] = df["batting_team_prior"].fillna(0.5)
        df["bowling_team_prior"] = df["bowling_team_prior"].fillna(0.5)
        df.to_csv(path, index=False)
        new_cols = set(df.columns) - before_cols
        print(f"{split}.csv: added columns {new_cols}, {len(df)} rows, "
              f"{df['match_id'].nunique()} matches")

    print("Done. team_strength_prior features merged into train/val/test CSVs.")