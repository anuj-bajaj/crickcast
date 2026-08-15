"""
Phase 4 — Calibration analysis

A model can have great AUC while still being poorly calibrated — e.g.
consistently saying "90%" for situations that actually resolve in the
predicted team's favor only 75% of the time. AUC only cares about ranking
predictions correctly relative to each other, not whether the probability
numbers themselves are trustworthy.

This script:
  1. Bins predicted probabilities into deciles (0-10%, 10-20%, ... 90-100%)
  2. For each bin, compares mean predicted probability vs. actual observed
     win rate within that bin
  3. Computes Expected Calibration Error (ECE) — a single summary number
  4. Saves a calibration plot for both baseline and main model
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "data/processed"
N_BINS = 10


def compute_calibration(y_true, y_proba, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    ece = 0.0
    n_total = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue
        mean_pred = y_proba[mask].mean()
        actual_rate = y_true[mask].mean()
        rows.append({
            "bin_range": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "count": int(count),
            "mean_predicted": round(mean_pred, 3),
            "actual_win_rate": round(actual_rate, 3),
            "gap": round(abs(mean_pred - actual_rate), 3),
        })
        ece += (count / n_total) * abs(mean_pred - actual_rate)

    return pd.DataFrame(rows), ece


def plot_calibration(cal_baseline, cal_main, ece_baseline, ece_main):
    from plot_style import apply_brand_style, PINE, AMBER
    apply_brand_style()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color=PINE, alpha=0.35, linewidth=1.5, label="Perfect calibration")

    ax.plot(cal_baseline["mean_predicted"], cal_baseline["actual_win_rate"],
             marker="o", markersize=5, color=AMBER, linewidth=2, alpha=0.9,
             label=f"Baseline (ECE={ece_baseline:.3f})")
    ax.plot(cal_main["mean_predicted"], cal_main["actual_win_rate"],
             marker="o", markersize=6.5, color=PINE, linewidth=2.5,
             label=f"Main model / XGBoost (ECE={ece_main:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual observed win rate")
    ax.set_title("Calibration curve: predicted vs. actual win probability")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left")

    fig.savefig(f"{OUT_DIR}/calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/calibration_curve.png")


if __name__ == "__main__":
    df = pd.read_csv(f"{OUT_DIR}/val_with_predictions.csv")

    y_true = df["batting_team_won"].values
    baseline_proba = df["baseline_proba"].values
    main_proba = df["main_model_proba"].values

    cal_baseline, ece_baseline = compute_calibration(y_true, baseline_proba)
    cal_main, ece_main = compute_calibration(y_true, main_proba)

    print("\n--- Baseline calibration table ---")
    print(cal_baseline.to_string(index=False))
    print(f"\nBaseline ECE: {ece_baseline:.4f}")

    print("\n--- Main model calibration table ---")
    print(cal_main.to_string(index=False))
    print(f"\nMain model ECE: {ece_main:.4f}")

    plot_calibration(cal_baseline, cal_main, ece_baseline, ece_main)

    # ---- Add ECE to the shared model-stats file (see phase3_modeling.py) ----
    # Phase 3 writes this file first with accuracy/AUC/etc; this appends the
    # calibration numbers onto the same file rather than creating a second
    # one, so the frontend only ever has to fetch a single source of truth.
    stats_path = f"{OUT_DIR}/model_stats.json"
    try:
        with open(stats_path) as f:
            stats = json.load(f)
    except FileNotFoundError:
        stats = {}
    stats["baseline_ece"] = ece_baseline
    stats["main_model_ece"] = ece_main
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nUpdated {stats_path} with calibration stats")