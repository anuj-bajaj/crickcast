"""
Shared pytest fixtures. Run pytest from the project root — phase6b_api.py
loads models/main_model.joblib via a relative path, so the working
directory matters.

    pytest
    pytest -v                    # verbose
    pytest tests/test_api_predict.py   # just one file
"""

import sys
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MAIN_FEATURES = [
    "cum_runs", "cum_wickets", "balls_remaining", "runs_required",
    "required_run_rate", "current_run_rate", "recent_run_rate",
    "batting_team_prior", "bowling_team_prior",
]


@pytest.fixture(scope="session")
def model():
    """The trained main model, loaded once for the whole test session.
    Requires models/main_model.joblib to exist (run phase3_modeling.py
    at least once first)."""
    model_path = REPO_ROOT / "models" / "main_model.joblib"
    if not model_path.exists():
        pytest.skip(f"{model_path} not found — run phase3_modeling.py first")
    return joblib.load(model_path)


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient against the real app — same relative-path
    dependency on cwd as the model fixture above."""
    from src.phase6b_api import app
    return TestClient(app)
