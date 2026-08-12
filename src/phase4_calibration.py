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
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    ax.plot(cal_baseline["mean_predicted"], cal_baseline["actual_win_rate"],
             marker="o", label=f"Baseline (ECE={ece_baseline:.3f})")
    ax.plot(cal_main["mean_predicted"], cal_main["actual_win_rate"],
             marker="o", label=f"Main model / XGBoost (ECE={ece_main:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual observed win rate")
    ax.set_title("Calibration curve: predicted vs. actual win probability")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(f"{OUT_DIR}/calibration_curve.png", dpi=150, bbox_inches="tight")
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
