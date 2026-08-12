"""
Phase 1 — Cricsheet data pipeline
Downloads T20 ball-by-ball data from Cricsheet and parses it into a flat,
per-ball dataframe scoped to 2nd-innings chases (win probability target).

Run this in Antigravity / your local environment (needs internet access).

Steps:
  1. Download & unzip Cricsheet T20 json bundle
  2. Parse each match into per-ball rows
  3. Compute the core state features
  4. Split by match_id (train/val/test) to avoid leakage
  5. Save flat CSVs
"""

import json
import os
import zipfile
import io
import requests
import pandas as pd
from pathlib import Path

# ---------- Config ----------
# Cricsheet T20 male matches, JSON format. Check cricsheet.org/downloads for
# the current bundle name/URL if this changes.
CRICSHEET_URL = "https://cricsheet.org/downloads/t20s_male_json.zip"
RAW_DIR = Path("data/raw_matches")
OUT_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Step 1: Download & unzip ----------
# Cricsheet is behind bot-protection (Cloudflare-style JS challenge) that
# python-requests cannot pass. Download the zip manually once through a
# real browser and place it at LOCAL_ZIP_PATH — the script extracts from
# there instead of hitting the network.
LOCAL_ZIP_PATH = Path("data/downloads/t20s_male_json.zip")

def download_and_extract():
    if not LOCAL_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Expected zip at {LOCAL_ZIP_PATH} but it's not there.\n"
            f"Download it manually: open "
            f"https://cricsheet.org/downloads/t20s_male_json.zip in your browser, "
            f"save it to {LOCAL_ZIP_PATH.parent}/, then re-run this script."
        )

    print(f"Extracting local zip: {LOCAL_ZIP_PATH}")
    with zipfile.ZipFile(LOCAL_ZIP_PATH) as z:
        z.extractall(RAW_DIR)
    print(f"Extracted to {RAW_DIR}")

# ---------- Step 2 & 3: Parse one match into ball-by-ball rows ----------
def parse_match(match_json: dict, match_id: str):
    """
    Returns a list of row-dicts for the 2nd innings only (the chase),
    since that's the only innings where 'required run rate' is meaningful.
    """
    rows = []
    innings_list = match_json.get("innings", [])
    if len(innings_list) < 2:
        return rows  # no result / abandoned / super-over-only edge cases

    first_innings = innings_list[0]
    second_innings = innings_list[1]

    # Target = first innings total + 1
    first_innings_runs = sum(
        d.get("runs", {}).get("total", 0)
        for over in first_innings.get("overs", [])
        for d in over.get("deliveries", [])
    )
    target = first_innings_runs + 1

    batting_team = second_innings.get("team")
    bowling_team = first_innings.get("team")

    # Determine winner from match_json['info']['outcome']
    outcome = match_json.get("info", {}).get("outcome", {})
    winner = outcome.get("winner")
    if winner is None:
        return rows  # no result (tie/no result/abandoned) — skip for now
    batting_team_won = 1 if winner == batting_team else 0

    total_balls_in_innings = 20 * 6  # T20 = 20 overs, adjust if data has different over counts

    cum_runs = 0
    cum_wickets = 0
    ball_count = 0
    recent_runs_window = []  # last 12 balls, for short-term momentum

    for over_data in second_innings.get("overs", []):
        over_num = over_data.get("over")
        for delivery in over_data.get("deliveries", []):
            extras = delivery.get("extras", {})
            is_wide = "wides" in extras
            is_noball = "noballs" in extras
            is_illegal_delivery = is_wide or is_noball  # doesn't count toward the 6 balls of an over

            runs_this_ball = delivery.get("runs", {}).get("total", 0)  # includes extras, correct as-is
            is_wicket = 1 if "wickets" in delivery else 0

            cum_runs += runs_this_ball
            cum_wickets += is_wicket

            # Only legal deliveries advance the ball count / overs — this is the fix
            if not is_illegal_delivery:
                ball_count += 1
                recent_runs_window.append(runs_this_ball)
                if len(recent_runs_window) > 12:
                    recent_runs_window.pop(0)
            else:
                # extras still count toward the recent-form signal, just not as
                # a distinct "ball" — add their runs to whatever the last legal
                # ball's contribution was, so a flurry of extras isn't invisible
                if recent_runs_window:
                    recent_runs_window[-1] += runs_this_ball

            balls_remaining = max(total_balls_in_innings - ball_count, 0)
            runs_required = max(target - cum_runs, 0)
            required_run_rate = (
                (runs_required / (balls_remaining / 6)) if balls_remaining > 0 else 99.0
            )
            current_run_rate = (cum_runs / (ball_count / 6)) if ball_count > 0 else 0.0
            recent_run_rate = (
                sum(recent_runs_window) / (len(recent_runs_window) / 6)
                if recent_runs_window else 0.0
            )

            rows.append({
                "match_id": match_id,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "over": over_num,
                "ball_count": ball_count,
                "cum_runs": cum_runs,
                "cum_wickets": cum_wickets,
                "balls_remaining": balls_remaining,
                "runs_required": runs_required,
                "required_run_rate": round(required_run_rate, 2),
                "current_run_rate": round(current_run_rate, 2),
                "recent_run_rate": round(recent_run_rate, 2),
                "is_wicket_this_ball": is_wicket,
                "runs_this_ball": runs_this_ball,
                "is_wide": int(is_wide),
                "is_noball": int(is_noball),
                "target": target,
                "batting_team_won": batting_team_won,
            })

    return rows

# ---------- Step 4: Build full dataset ----------
def build_dataset():
    all_rows = []
    match_files = list(RAW_DIR.glob("*.json"))
    print(f"Found {len(match_files)} match files")

    for i, fpath in enumerate(match_files):
        match_id = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                match_json = json.load(f)
            rows = parse_match(match_json, match_id)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Skipping {match_id}: {e}")

        if i % 200 == 0:
            print(f"Processed {i}/{len(match_files)} matches...")

    df = pd.DataFrame(all_rows)
    print(f"Total rows: {len(df)}, total matches with data: {df['match_id'].nunique()}")
    return df

# ---------- Step 5: Match-level train/val/test split ----------
def split_by_match(df: pd.DataFrame, seed: int = 42):
    match_ids = df["match_id"].unique()
    rng = pd.Series(match_ids).sample(frac=1.0, random_state=seed).values  # shuffle

    n = len(rng)
    train_ids = set(rng[: int(0.7 * n)])
    val_ids = set(rng[int(0.7 * n): int(0.85 * n)])
    test_ids = set(rng[int(0.85 * n):])

    train_df = df[df["match_id"].isin(train_ids)]
    val_df = df[df["match_id"].isin(val_ids)]
    test_df = df[df["match_id"].isin(test_ids)]

    print(f"Train matches: {len(train_ids)} ({len(train_df)} rows)")
    print(f"Val matches:   {len(val_ids)} ({len(val_df)} rows)")
    print(f"Test matches:  {len(test_ids)} ({len(test_df)} rows)")

    return train_df, val_df, test_df

if __name__ == "__main__":
    if not any(RAW_DIR.glob("*.json")):
        download_and_extract()

    df = build_dataset()
    df.to_csv(OUT_DIR / "all_balls.csv", index=False)

    train_df, val_df, test_df = split_by_match(df)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print("Done. Files written to data/processed/")