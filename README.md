# Crickcast

A live win-probability engine for T20 cricket run-chases — a trained ML model, a FastAPI backend, an LLM explanation layer, and a React frontend, tied together end to end.

Given the current match state (score, wickets, overs, target), Crickcast predicts the chasing team's win probability in real time and generates a natural-language explanation of the key factors driving that number.

## How it works

- **Data pipeline** — Cricsheet T20 international data (3,490 matches), parsed into ball-by-ball records for second-innings chases, with a match-level train/val/test split (70/15/15) to prevent leakage.
- **Team-strength priors** — a leakage-safe rolling prior computed from each team's last 15 matches, strictly pre-match chronologically. A separate "current strength" snapshot (same logic, evaluated as of the latest known match) powers a real team-selection dropdown in the live app — not just a bare manual slider.
- **Modeling** — a logistic regression baseline compared against the main model, XGBoost, trained on the full feature set with explicit monotonicity constraints and regularization (see Results below).
- **Calibration** — calibration tables and Expected Calibration Error (ECE) computed for both models to confirm predicted probabilities are trustworthy, not just accurate.
- **Event-impact analysis** — quantifies how much each event type (wicket, six, four, dot ball, other runs) swings the win probability.
- **Testing** — a `pytest` suite verifies the trained model's monotonicity guarantees and the API's decided-state/validation behavior directly against the real model artifact, not just unit-testing code paths in isolation.
- **Explanation layer** — a Groq-powered LLM turns the model's raw output into a plain-language explanation of what's driving the current probability, using real team names when the user has selected them.
- **API** — a FastAPI `/predict` endpoint ties the model and explanation layer together, with input validation and a deterministic short-circuit for already-decided match states (won / all out / overs complete) rather than extrapolating the model into states it never trained on.
- **Frontend** — a React + Vite + Tailwind single-page app with a live ball-by-ball simulator (undo, quick-start scenarios, a ball-by-ball scorecard strip), real team selection, real-time probability/explanation display, a probability evolution chart, and a model insights section (calibration curve, event-impact chart, feature importance) generated fresh by the pipeline on every retrain.

## Results

| Metric | Baseline (Logistic Regression) | Main model (XGBoost, monotonicity-constrained) |
|---|---|---|
| Accuracy | 0.824 | 0.835 |
| AUC | 0.915 | 0.923 |
| Log loss | 0.365 | 0.346 |
| Brier score | 0.118 | 0.112 |
| ECE | 0.018 | 0.012 |

The main model is trained with explicit per-feature monotonicity
constraints (e.g. scoring more runs can never *decrease* the predicted
win probability, all else held fixed) plus added regularization — both
found necessary by directly auditing the trained model's behavior across
thousands of simulated match states, not just its validation metrics. A
`pytest` suite (`tests/`) locks these properties in going forward: it
verifies the model's response to every feature and event type is exactly
in the direction cricket logic says it must be, and fails loudly if a
future retrain ever regresses that. See `BACKEND_HANDOVER.md` for the
full writeup, including the two real bugs this caught.

Note: exact metrics vary slightly run-to-run (the train/val/test split
depends on filesystem enumeration order, not just the random seed — a
known, documented limitation, not noise to worry about).

## Live demo

- Frontend: [crickcast-sooty.vercel.app](https://crickcast-sooty.vercel.app/)
- API docs: [crickcast-backend.onrender.com/docs](https://crickcast-backend.onrender.com/docs)

Note: the backend is hosted on Render's free tier, so it may take 30-60 seconds to wake up on the first request after a period of inactivity.

## Tech stack

Python, pandas, XGBoost, scikit-learn, FastAPI, pytest, Groq, React, Vite, Tailwind CSS

## Project structure

```
crickcast/
├── src/
│   ├── phase1_data_pipeline.py     # Cricsheet data → ball-by-ball dataframe
│   ├── phase2_team_priors.py       # rolling team-strength priors + current team snapshot
│   ├── phase3_modeling.py          # baseline + XGBoost training, feature importance
│   ├── phase4_calibration.py       # calibration tables + ECE
│   ├── phase5_event_impact.py      # event-type swing analysis
│   ├── phase6a_explanation.py      # Groq explanation layer
│   └── phase6b_api.py              # FastAPI /predict endpoint
├── tests/                          # pytest suite — model + API behavior
├── models/                         # trained models (joblib)
├── data/processed/                 # train/val/test splits + generated diagnostics
├── frontend/                       # React + Vite + Tailwind app
│   └── public/                     # static files the pipeline generates (calibration
│                                    # curve, event-impact chart, feature importance,
│                                    # model stats, team priors) — copied here after
│                                    # every retrain, not committed as source
├── BACKEND_HANDOVER.md             # detailed writeup of the ML/backend work
└── README.md
```

## Running locally

**Backend**
```bash
pip install -r requirements.txt
uvicorn src.phase6b_api:app --reload
```
Visit `http://localhost:8000/docs` for the interactive API docs.

**Tests**
```bash
pytest          # run from the project root
pytest -v       # verbose
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

**Retraining the pipeline** (in order — each phase depends on the previous one's output):
```bash
python src/phase1_data_pipeline.py
python src/phase2_team_priors.py
python src/phase3_modeling.py
python src/phase4_calibration.py
python src/phase5_event_impact.py
```
Then copy the generated static files into the frontend:
```bash
cp data/processed/calibration_curve.png data/processed/event_impact_swing.png \
   data/processed/feature_importance.png data/processed/model_stats.json \
   data/processed/team_priors.json frontend/public/
```