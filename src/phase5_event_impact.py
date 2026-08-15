"""
Phase 5 — Event-impact / swing analysis

For each ball, compute how much the main model's win-probability estimate
shifted from the previous ball to this one, then aggregate that shift by
event type (wicket, six, four, dot ball, other). This answers: "which
events actually move the win probability needle, and by how much?" —
an analysis layer most public win-probability tools don't expose.

Sign convention: probability is P(batting/chasing team wins). So a wicket
should, on average, produce a NEGATIVE swing (bad for the batting team),
while a six/four should produce a POSITIVE swing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "data/processed"


def categorize_event(row):
    if row["is_wicket_this_ball"] == 1:
        return "wicket"
    if row["runs_this_ball"] == 6:
        return "six"
    if row["runs_this_ball"] == 4:
        return "four"
    if row["runs_this_ball"] == 0:
        return "dot_ball"
    return "other_runs"  # 1, 2, 3, 5


def compute_swings(df: pd.DataFrame, proba_col: str):
    """
    Sort each match chronologically by ball_count, compute the change in
    predicted probability from the previous ball to this one, within each
    match (so swings never bleed across match boundaries).
    """
    df = df.sort_values(["match_id", "ball_count"]).copy()
    df["proba_prev"] = df.groupby("match_id")[proba_col].shift(1)
    df["swing"] = df[proba_col] - df["proba_prev"]

    # first ball of each match has no "previous" — drop those rows for swing analysis
    df = df.dropna(subset=["swing"])

    df["event_type"] = df.apply(categorize_event, axis=1)
    return df


def summarize_by_event(df: pd.DataFrame):
    summary = df.groupby("event_type")["swing"].agg(
        count="count",
        mean_swing="mean",
        mean_abs_swing=lambda s: s.abs().mean(),
        std_swing="std",
    ).reset_index()
    summary = summary.sort_values("mean_abs_swing", ascending=False)
    return summary


def plot_swings(summary: pd.DataFrame):
    from plot_style import apply_brand_style, MOSS, WICKET
    apply_brand_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [WICKET if v < 0 else MOSS for v in summary["mean_swing"]]
    ax.bar(summary["event_type"], summary["mean_swing"], color=colors, width=0.6, edgecolor="none")
    ax.axhline(0, color="#1b3a2b", linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean win-probability swing (batting team)")
    ax.set_title("Average win-probability swing by event type")
    ax.grid(axis="x")  # only horizontal gridlines matter for a bar chart

    fig.savefig(f"{OUT_DIR}/event_impact_swing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/event_impact_swing.png")


if __name__ == "__main__":
    df = pd.read_csv(f"{OUT_DIR}/val_with_predictions.csv")

    df = compute_swings(df, proba_col="main_model_proba")
    summary = summarize_by_event(df)

    print("\n--- Event-impact summary (main model) ---")
    print(summary.to_string(index=False))

    plot_swings(summary)

    # Sanity check worth eyeballing: wickets should be negative, six/four positive
    print("\nSanity check — expected directions:")
    print("  wicket:     should be negative (bad for batting team)")
    print("  six/four:   should be positive (good for batting team)")
    print("  dot_ball:   should be slightly negative (required rate creeps up)")