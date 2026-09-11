# Crickcast

A live win-probability engine for T20 cricket run-chases — a trained ML model, a FastAPI backend, a hallucination-guarded LLM commentary layer, and a React frontend, tied together end to end.

Given the current match state (score, wickets, overs, target), Crickcast predicts the chasing team's win probability in real time and generates a natural-language line of commentary explaining what's driving that number.

## How it works

- **Data pipeline** — Cricsheet T20 international data (3,490 match files; 3,374 with a valid date and result), parsed into 359,309 ball-by-ball rows for second-innings chases, with a match-level train/val/test split (2,361/506/507 matches, i.e. 70/15/15) to prevent leakage.
- **Team-strength priors** — a leakage-safe rolling prior computed from each team's last 15 matches, strictly pre-match chronologically. A separate "current strength" snapshot (same logic, evaluated as of the latest known match) powers a real team-selection dropdown in the live app — not just a bare manual slider.
- **Modeling** — a logistic regression baseline compared against the main model, XGBoost, trained on the full feature set with explicit monotonicity constraints and regularization (see Results below).
- **Calibration** — calibration tables and Expected Calibration Error (ECE) computed for both models to confirm predicted probabilities are trustworthy, not just accurate.
- **Event-impact analysis** — quantifies how much each event type (wicket, six, four, dot ball, other runs) actually swings the win probability, and checks the direction against cricket logic (see Results below).
- **Explanation layer** — turns the model's raw output into one line of live commentary through a deliberately narrow three-stage design rather than trusting an LLM's free text directly: `build_facts()` computes every fact about a ball (score, wickets, probability swing, chase context, streaks) in plain Python, already correctly worded, so the model never computes a number or decides who did what; the LLM weaves 2-4 of those facts into a single natural sentence; `validate_output()` then checks the model's own text against the facts it was given — hallucinated numbers, leaked labels, ALL-CAPS, unfinished sentences — before it's ever shown to a user. A failed check triggers one corrective retry, then a guaranteed-correct fallback sentence assembled with no LLM involved at all. Handles genuine cricket edge cases directly in code: consecutive-event streaks ("3 fours on the trot", a wicket-taking cluster), free hits, and a chase that's mathematically over (declared won/lost from the arithmetic itself — `runs_required > 6 × balls_remaining`, say — independent of what the trained model's own probability estimate still shows for that state). Groq is the primary provider, with OpenRouter as an independent fallback on a separate free-tier quota.
- **Testing** — a 124-test `pytest` suite: model-behavior tests that verify monotonicity and guaranteed event directions directly against the real trained model artifact (not just unit-testing code paths in isolation), API validation/decided-state tests, and a full suite for the explanation layer's fact-building, validation, and fallback guarantees.
- **API** — a FastAPI backend exposing `/predict` (win probability + swing) and `/explain` (commentary) as separate endpoints — deliberately split so a slow or rate-limited LLM call never delays the probability number, which comes from a local model and answers in milliseconds. `/predict` includes input validation and a deterministic short-circuit for already-decided match states (won / all out / overs complete) rather than extrapolating the model into states it never trained on.
- **Frontend** — a React + Vite + Tailwind single-page app with a live ball-by-ball simulator (undo, quick-start scenarios, a ball-by-ball scorecard strip), real team selection, real-time probability/commentary display, a probability evolution chart, and a model insights section (calibration curve, event-impact chart, feature importance) generated fresh by the pipeline on every retrain.

## Results

| Metric | Baseline (Logistic Regression) | Main model (XGBoost, monotonicity-constrained) |
|---|---|---|
| Accuracy | 0.8243 | 0.8353 |
| AUC | 0.9152 | 0.9240 |
| Log loss | 0.3653 | 0.3440 |
| Brier score | 0.1183 | 0.1115 |
| ECE | 0.0179 | 0.0118 |

The main model's top feature by gain-based importance is `required_run_rate`
(0.533) — by a wide margin the single biggest driver of the predicted
win probability, which matches cricket intuition: how far ahead or
behind the required rate is dominates the picture more than any other
single signal.

The main model is trained with explicit per-feature monotonicity
constraints (e.g. scoring more runs can never *decrease* the predicted
win probability, all else held fixed) plus added regularization — both
found necessary by directly auditing the trained model's behavior across
thousands of simulated match states, not just its validation metrics. A
`pytest` suite (`tests/`) locks these properties in going forward: it
verifies the model's response to every feature and event type is exactly
in the direction cricket logic says it must be, and fails loudly if a
future retrain ever regresses that.

Two real bugs were only found this way, not from validation metrics
alone:
- **Non-monotonic response to scoring events**, in ~14.6% of simulated
  match states — a boundary (runs up, required rate down, current rate
  up: every signal favoring the batting side) could still make the
  predicted win probability go *down*, because XGBoost had no
  constraint stopping an individual tree split fit to noise in a sparse
  region of the feature space from outweighing the real signal.
  Explicit monotonicity constraints (`MONOTONE_CONSTRAINTS` in
  `phase3_modeling.py`) dropped this to ~0.3%, with the remainder capped
  under a percentage point.
- **Oversensitivity to small changes** — the model could be
  directionally correct and still swing 10-15 probability points for a
  single run when the underlying state barely moved. `min_child_weight`,
  a shallower `max_depth`, and L2 regularization together roughly halved
  this (~5.8% → ~3.5% of the same sweep).

Validation accuracy/AUC/log-loss/Brier were essentially unchanged by
either fix (some moved slightly in the constrained model's favor) —
these were behavioral corrections, not accuracy trade-offs.

**Event-impact analysis** — the mean win-probability swing (batting
team's perspective) attributable to each event type, measured directly
on the validation set:

| Event | Mean swing | Direction |
|---|---|---|
| Six | +0.0661 | ✅ helps batting side |
| Wicket | −0.0645 | ✅ hurts batting side |
| Four | +0.0409 | ✅ helps batting side |
| Dot ball | −0.0131 | ✅ hurts batting side (required rate creeps up) |
| Other runs (1/2/3/5) | +0.0034 | ✅ mildly helps |

Every event lines up with cricket intuition, at sensible relative
magnitudes — a wicket costs roughly as much as a six gains, and a dot
ball costs about a fifth of what a wicket does. This is a direct,
quantified check of "does the model actually understand cricket," not
just an aggregate accuracy number.

The main model is served from `models/main_model.json` (XGBoost's own
portable format via `save_model()`/`load_model()`), not the `.joblib`
pickle — XGBoost's docs warn that pickle serialization of a `Booster`
isn't guaranteed compatible across XGBoost versions, which is exactly
the failure mode `save_model()` exists to avoid. The `.joblib` copy is
still saved alongside it for anyone doing offline analysis who wants the
exact fitted sklearn wrapper object back; the API itself only ever
reads the `.json` file.

Note: the train/val/test split is fully determined by `random_state=42`
in `phase1_data_pipeline.py`'s `split_by_match()` — match IDs are sorted
before shuffling, so re-running the pipeline reproduces the same split
and the same metrics every time, independent of filesystem enumeration
order.

## Live demo

- Frontend: [crickcast-sooty.vercel.app](https://crickcast-sooty.vercel.app/)
- API docs: [crickcast-backend.onrender.com/docs](https://crickcast-backend.onrender.com/docs)

Note: the backend is hosted on Render's free tier, so it may take 30-60 seconds to wake up on the first request after a period of inactivity.

## Tech stack

Python, pandas, XGBoost, scikit-learn, FastAPI, pytest, Groq, OpenRouter, React, Vite, Tailwind CSS

## Project structure

```
crickcast/
├── src/
│   ├── phase1_data_pipeline.py     # Cricsheet data → ball-by-ball dataframe
│   ├── phase2_team_priors.py       # rolling team-strength priors + current team snapshot
│   ├── phase3_modeling.py          # baseline + XGBoost training, feature importance
│   ├── phase4_calibration.py       # calibration tables + ECE
│   ├── phase5_event_impact.py      # event-type swing analysis
│   ├── phase6a_explanation.py      # commentary layer (facts → LLM → validation → fallback)
│   └── phase6b_api.py              # FastAPI /predict + /explain endpoints
├── tests/                          # pytest suite — model, API, and explanation-layer behavior
├── models/                         # trained models: main_model.json (served
│                                    # by the API) + .joblib copies of both
│                                    # models (offline analysis convenience)
├── data/processed/                 # train/val/test splits + generated diagnostics
├── frontend/                       # React + Vite + Tailwind app
│   └── public/                     # static files the pipeline generates (calibration
│                                    # curve, event-impact chart, feature importance,
│                                    # model stats, team priors) — copied here after
│                                    # every retrain, not committed as source
└── README.md
```

## Environment variables

Set these in your shell, or in a `.env` file in the project root
(gitignored — never commit real keys):

| Variable | Required? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes, for commentary | Primary LLM provider for `/explain`. Free key at [console.groq.com/keys](https://console.groq.com/keys). |
| `OPENROUTER_API_KEY` | No | Independent fallback provider, used only if Groq fails outright. Free key at [openrouter.ai/keys](https://openrouter.ai/keys). |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS allowlist for production. Defaults to `localhost:5173` for local dev. |

`/predict` works with no environment variables set at all. Without
`GROQ_API_KEY` (and no `OPENROUTER_API_KEY`), `/explain` degrades
gracefully — it returns `{"explanation": null}` rather than an error.

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