"""
Unit tests for InvestigatorPipeline.

Run with: pytest tests/ -v
All 5 tests should pass before committing.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from src.investigator import InvestigatorPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_model_and_csv(tmp_path_factory):
    """
    Creates a small test model + mispredictions CSV using the
    breast cancer dataset — no Kaggle download needed for tests.
    """
    tmp = tmp_path_factory.mktemp("test_data")

    # Train a small model
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:100], columns=data.feature_names)
    y = pd.Series(data.target[:100])

    model = RandomForestClassifier(
        n_estimators=10, random_state=42)
    model.fit(X, y)

    model_path = str(tmp / "test_model.pkl")
    joblib.dump(model, model_path)

    # Create fake mispredictions CSV (just use a subset of the data)
    wrong = X.copy()
    wrong["predicted"] = 0
    wrong["actual"] = 1
    csv_path = str(tmp / "test_wrong.csv")
    wrong.to_csv(csv_path, index=False)

    return model_path, csv_path


@pytest.fixture
def pipeline():
    return InvestigatorPipeline()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_output_has_required_keys(pipeline, sample_model_and_csv):
    """Agent 1 output must contain all keys defined in schema.md"""
    model_path, csv_path = sample_model_and_csv
    result = pipeline.run(model_path, csv_path, model_name="test")

    required_keys = {"agent", "version", "model",
                     "total_failures", "top_features", "failure_patterns"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}")


def test_top_features_returns_three(pipeline, sample_model_and_csv):
    """top_features should have exactly 3 entries (or fewer if model has < 3 features)"""
    model_path, csv_path = sample_model_and_csv
    result = pipeline.run(model_path, csv_path, top_n_features=3)
    assert len(result["top_features"]) <= 3
    assert len(result["top_features"]) >= 1


def test_each_top_feature_has_correct_schema(pipeline, sample_model_and_csv):
    """Each item in top_features must have 'feature' and 'mean_abs_shap'"""
    model_path, csv_path = sample_model_and_csv
    result = pipeline.run(model_path, csv_path)
    for feat in result["top_features"]:
        assert "feature" in feat, f"Missing 'feature' key in {feat}"
        assert "mean_abs_shap" in feat, f"Missing 'mean_abs_shap' in {feat}"
        assert isinstance(feat["mean_abs_shap"], float), (
            f"mean_abs_shap must be float, got {type(feat['mean_abs_shap'])}")
        assert feat["mean_abs_shap"] >= 0, "SHAP values must be non-negative"


def test_handles_missing_model_gracefully(pipeline, sample_model_and_csv):
    """Pipeline must not crash if the model file doesn't exist."""
    _, csv_path = sample_model_and_csv
    result = pipeline.run("nonexistent/model.pkl", csv_path)
    assert "error" in result, "Should return error key when model is missing"
    assert result["total_failures"] == 0


def test_total_failures_matches_csv_row_count(pipeline, sample_model_and_csv):
    """total_failures must equal the number of rows in the mispredictions CSV"""
    model_path, csv_path = sample_model_and_csv
    expected_rows = len(pd.read_csv(csv_path))
    result = pipeline.run(model_path, csv_path)
    assert result["total_failures"] == expected_rows, (
        f"Expected {expected_rows} failures, got {result['total_failures']}")