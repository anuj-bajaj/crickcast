"""
Phase 3 — Modeling: baseline vs. main model

Baseline: Logistic Regression on a deliberately small, explainable feature
set (required_run_rate, wickets lost, balls remaining) — this is the
"could a simple heuristic already do this reasonably well" anchor.

Main model: XGBoost on the full feature set, including the team strength
priors from Phase 2.

Metrics reported for both, on the held-out val set:
  - Accuracy, AUC        (classification quality)
  - Log loss, Brier score (probability quality — this is a probability
    estimation problem, not just a classification one, so these matter
    as much as accuracy)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

OUT_DIR = "data/processed"
MODEL_DIR = "models"

BASELINE_FEATURES = ["required_run_rate", "cum_wickets", "balls_remaining"]

MAIN_FEATURES = [
    "cum_runs", "cum_wickets", "balls_remaining", "runs_required",
    "required_run_rate", "current_run_rate", "recent_run_rate",
    "batting_team_prior", "bowling_team_prior",
]

# Monotonicity constraints, one per entry in MAIN_FEATURES (+1 = predicted
# P(batting team wins) must be non-decreasing in that feature, -1 = non-
# increasing, holding the other features fixed).
#
# Without this, XGBoost is free to fit whatever direction the training data
# happens to noisily suggest in any given region of the feature space — and
# some regions are very sparse (e.g. "5+ wickets down but still cruising on
# a low required rate" is under 0.5% of the training rows), so the model
# has almost nothing to learn a sane relationship from there. In practice
# that showed up as a real bug: scoring a boundary (runs up, required rate
# down, current rate up — every scoring signal moving in the batting team's
# favor) could still make the model's predicted win probability go DOWN,
# because some individual tree split, fit to noise in a near-empty corner
# of the data, outweighed the signal. A sweep over 20k random match states
# found this in ~14.6% of them, some off by as much as 28 percentage
# points. With these constraints in place that dropped to ~0.3%, with the
# few remaining cases capped under a percentage point — small, legitimate
# trade-offs (a boundary does cost a ball, and balls left is itself
# constrained to help the batting team) rather than the model just being
# wrong in a data-sparse pocket. Validation accuracy/AUC/log-loss/Brier are
# essentially unchanged (AUC/log-loss/Brier all move slightly in the
# monotonic model's favor; accuracy is within 0.002).
MONOTONE_CONSTRAINTS = (1, -1, 1, -1, -1, 1, 1, 1, -1)

# min_child_weight / reg_lambda / a shallower max_depth: monotonicity fixes
# the DIRECTION of the model's response to each feature but not its
# magnitude — a still-noisy tree can be directionally correct and still
# swing 10-15 points of probability for a single run when the match state
# barely changed (e.g. required run rate moving from 12.00 to 12.09). That
# kind of oversensitivity is exactly what these three params damp: a higher
# min_child_weight forces each leaf to summarize more real training rows
# (so a leaf can't be carved out to fit a handful of noisy examples), a
# shallower max_depth limits how many features can interact in a single
# tree path, and reg_lambda (L2) shrinks leaf weights generally. Empirically
# this cut "flat-required-rate but double-digit-swing" singles roughly in
# half (5.8% -> 3.5% of a 20k-state sweep, worst case -9.4pp -> -7.3pp)
# while validation accuracy/AUC/log-loss/Brier were unchanged or marginally
# better, not worse.
TARGET = "batting_team_won"


def load_split(name):
    return pd.read_csv(f"{OUT_DIR}/{name}.csv")


def evaluate(model, X_val, y_val, label):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, proba)
    ll = log_loss(y_val, proba)
    brier = brier_score_loss(y_val, proba)

    print(f"\n--- {label} ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"Log loss:    {ll:.4f}")
    print(f"Brier score: {brier:.4f}")

    return {"label": label, "accuracy": acc, "auc": auc, "log_loss": ll, "brier": brier, "proba": proba}


if __name__ == "__main__":
    train_df = load_split("train")
    val_df = load_split("val")

    y_train = train_df[TARGET]
    y_val = val_df[TARGET]

    # ---- Baseline: Logistic Regression, small explainable feature set ----
    X_train_base = train_df[BASELINE_FEATURES]
    X_val_base = val_df[BASELINE_FEATURES]

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_base, y_train)
    baseline_results = evaluate(baseline_model, X_val_base, y_val, "Baseline (Logistic Regression)")

    # ---- Main model: XGBoost, full feature set, monotonicity-constrained ----
    X_train_main = train_df[MAIN_FEATURES]
    X_val_main = val_df[MAIN_FEATURES]

    main_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_lambda=10,
        eval_metric="logloss",
        random_state=42,
        monotone_constraints=MONOTONE_CONSTRAINTS,
    )
    main_model.fit(X_train_main, y_train)
    main_results = evaluate(main_model, X_val_main, y_val, "Main model (XGBoost)")

    # ---- Save trained models for reuse in Phase 6 (API / explanation layer) ----
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(baseline_model, f"{MODEL_DIR}/baseline_model.joblib")
    joblib.dump(main_model, f"{MODEL_DIR}/main_model.joblib")
    print(f"\nSaved trained models to {MODEL_DIR}/")

    # ---- Feature importance plot for the frontend's Almanack Notes section ----
    # XGBoost's sklearn wrapper reports "gain"-based importance by default
    # (relative contribution to reducing loss, normalized to sum to 1) —
    # not literally split-count, which would overweight cheap, frequently-
    # used splits. This is a genuine look inside the trained model, not
    # just a design element: it's the direct answer to "what is this model
    # actually using to make its predictions."
    import matplotlib.pyplot as plt
    import numpy as np
    from plot_style import apply_brand_style, PINE, AMBER
    apply_brand_style()

    importances = main_model.feature_importances_
    order = np.argsort(importances)  # ascending, so barh reads top-to-bottom as most-to-least important
    sorted_features = [MAIN_FEATURES[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    # Top feature (required_run_rate) picked out in amber — it dominates
    # by a wide enough margin that it's worth the eye landing there first.
    colors = [AMBER if i == len(sorted_importances) - 1 else PINE for i in range(len(sorted_importances))]
    ax.barh(sorted_features, sorted_importances, color=colors, height=0.6)
    ax.set_xlabel("Relative importance (gain)")
    ax.set_title("XGBoost feature importance — main model")
    ax.grid(axis="y")

    fig.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/feature_importance.png")
    top_feature = sorted_features[-1]
    print(f"Most important feature: {top_feature} ({sorted_importances[-1]:.3f})")

    # ---- Comparison summary ----
    print("\n=== Baseline vs Main Model ===")
    for metric in ["accuracy", "auc", "log_loss", "brier"]:
        b = baseline_results[metric]
        m = main_results[metric]
        direction = "better" if (metric in ["accuracy", "auc"] and m > b) or \
                                 (metric in ["log_loss", "brier"] and m < b) else "worse"
        print(f"{metric:10s}  baseline={b:.4f}  main={m:.4f}  ({direction} in main model)")

    # Save val predictions for Phase 4 (calibration) and Phase 5 (event-impact analysis)
    val_df["baseline_proba"] = baseline_results["proba"]
    val_df["main_model_proba"] = main_results["proba"]
    val_df.to_csv(f"{OUT_DIR}/val_with_predictions.csv", index=False)
    print(f"\nSaved val predictions to {OUT_DIR}/val_with_predictions.csv")

    # ---- Model stats for the frontend's Almanack Notes section ----
    # The site displays a few headline numbers (deliveries trained on, AUC,
    # calibration error) — these used to be hardcoded in App.jsx and went
    # stale every time the model got retrained. Writing them here instead
    # means the frontend can just fetch this file and always show the truth.
    # Phase 4 adds the ECE fields once it computes them; this is the file
    # that ultimately needs copying to frontend/public/model_stats.json,
    # same as the calibration/event-impact PNGs.
    import json
    stats = {
        "deliveries_trained_on": len(train_df),
        "baseline_accuracy": baseline_results["accuracy"],
        "baseline_auc": baseline_results["auc"],
        "baseline_log_loss": baseline_results["log_loss"],
        "baseline_brier": baseline_results["brier"],
        "main_model_accuracy": main_results["accuracy"],
        "main_model_auc": main_results["auc"],
        "main_model_log_loss": main_results["log_loss"],
        "main_model_brier": main_results["brier"],
    }
    with open(f"{OUT_DIR}/model_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved model stats to {OUT_DIR}/model_stats.json")