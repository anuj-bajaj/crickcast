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

    # ---- Main model: XGBoost, full feature set ----
    X_train_main = train_df[MAIN_FEATURES]
    X_val_main = val_df[MAIN_FEATURES]

    main_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    main_model.fit(X_train_main, y_train)
    main_results = evaluate(main_model, X_val_main, y_val, "Main model (XGBoost)")

    # ---- Save trained models for reuse in Phase 6 (API / explanation layer) ----
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(baseline_model, f"{MODEL_DIR}/baseline_model.joblib")
    joblib.dump(main_model, f"{MODEL_DIR}/main_model.joblib")
    print(f"\nSaved trained models to {MODEL_DIR}/")

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