"""
Shared pytest fixtures. Run pytest from the project root.

    pytest
    pytest -v                    # verbose
    pytest tests/test_api_predict.py   # just one file
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MAIN_FEATURES = [
    "cum_runs", "cum_wickets", "balls_remaining", "runs_required",
    "required_run_rate", "current_run_rate", "recent_run_rate",
    "batting_team_prior", "bowling_team_prior",
]


@pytest.fixture(scope="session")
def model():
    """The trained main model, loaded once for the whole test session —
    from the same .json artifact (and the same load_model() call) that
    phase6b_api.py actually serves from, not the .joblib copy. Testing a
    different file/loading path than what's deployed would mean these
    tests could pass while the live API behaves differently."""
    model_path = REPO_ROOT / "models" / "main_model.json"
    if not model_path.exists():
        pytest.skip(f"{model_path} not found — run phase3_modeling.py first")
    m = XGBClassifier()
    m.load_model(str(model_path))
    return m


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient against the real app."""
    from src.phase6b_api import app
    return TestClient(app)