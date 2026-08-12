# Crickcast

A live win-probability engine for T20 cricket run-chases — a trained ML model, a FastAPI backend, an LLM explanation layer, and a React frontend, tied together end to end.

Given the current match state (score, wickets, overs, target), Crickcast predicts the chasing team's win probability in real time and generates a natural-language explanation of the key factors driving that number.

## How it works

- **Data pipeline** — Cricsheet T20 international data (3,490 matches), parsed into ball-by-ball records for second-innings chases, with a match-level train/val/test split (70/15/15) to prevent leakage.
- **Team-strength priors** — a leakage-safe rolling prior computed from each team's last 15 matches, strictly pre-match chronologically.
- **Modeling** — a logistic regression baseline compared against the main model, XGBoost, trained on the full feature set.
- **Calibration** — calibration tables and Expected Calibration Error (ECE) computed for both models to confirm predicted probabilities are trustworthy, not just accurate.
- **Event-impact analysis** — quantifies how much each event type (wicket, six, four, dot ball, other runs) swings the win probability.
- **Explanation layer** — a Groq-powered LLM turns the model's raw output into a plain-language explanation of what's driving the current probability.
- **API** — a FastAPI `/predict` endpoint ties the model and explanation layer together.
- **Frontend** — a React + Vite + Tailwind single-page app with a live ball-by-ball simulator, real-time probability/explanation display, a probability evolution chart, and a model insights section (calibration curve + event-impact charts).

## Results

| Metric | Baseline (Logistic Regression) | Main model (XGBoost) |
|---|---|---|
| Accuracy | 0.824 | 0.836 |
| AUC | 0.915 | 0.923 |
| Log loss | 0.365 | 0.346 |
| Brier score | 0.118 | 0.112 |
| ECE | 0.018 | 0.012 |

## Live demo

- Frontend: _add link after deploy_
- API docs: _add link after deploy_

## Tech stack

Python, pandas, XGBoost, scikit-learn, FastAPI, Groq, React, Vite, Tailwind CSS

## Project structure

```
crickcast/
├── src/
│   ├── phase1_data_pipeline.py     # Cricsheet data → ball-by-ball dataframe
│   ├── phase2_team_priors.py       # rolling team-strength priors
│   ├── phase3_modeling.py          # baseline + XGBoost training
│   ├── phase4_calibration.py       # calibration tables + ECE
│   ├── phase5_event_impact.py      # event-type swing analysis
│   ├── phase6a_explanation.py      # Groq explanation layer
│   └── phase6b_api.py              # FastAPI /predict endpoint
├── models/                         # trained models (joblib)
├── frontend/                       # React + Vite + Tailwind app
└── README.md
```

## Running locally

**Backend**
```bash
pip install -r requirements.txt
uvicorn src.phase6b_api:app --reload
```
Visit `http://localhost:8000/docs` for the interactive API docs.

**Frontend**
```bash
cd frontend
npm install
npm run dev
```
