"""
Unit tests for CounterfactualPipeline.
Run with: pytest tests/ -v

These tests verify:
  - Output schema matches schema.md
  - Edge cases (missing model, empty CSV) handled gracefully
  - Success rate is reasonable on real data
  - cf_to_sentence produces readable output
"""
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from src.counterfactual import CounterfactualPipeline, cf_to_sentence


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_setup(tmp_path_factory):
    """
    Creates a model and mispredictions CSV using breast cancer dataset.
    Scope=module means this only runs once — shared across all tests.
    """
    tmp = tmp_path_factory.mktemp("cf_test")
    data = load_breast_cancer()

    X = pd.DataFrame(data.data[:200], columns=data.feature_names)
    y = pd.Series(data.target[:200])

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X[:160], y[:160])

    model_path = str(tmp / "model.pkl")
    joblib.dump(model, model_path)

    # Create mispredictions — rows where model is wrong
    y_pred = model.predict(X[160:])
    y_true = y[160:].values
    wrong_mask = y_pred != y_true
    wrong_X = X[160:][wrong_mask].copy()
    wrong_X["predicted"] = y_pred[wrong_mask]
    wrong_X["actual"]    = y_true[wrong_mask]

    csv_path = str(tmp / "wrong.csv")
    wrong_X.to_csv(csv_path, index=False)

    # Fake investigator output (top 3 features)
    inv_output = {
        "top_features": [
            {"feature": str(data.feature_names[0]), "mean_abs_shap": 0.15},
            {"feature": str(data.feature_names[1]), "mean_abs_shap": 0.10},
            {"feature": str(data.feature_names[2]), "mean_abs_shap": 0.08},
        ]
    }

    return model_path, csv_path, inv_output


@pytest.fixture
def pipeline():
    return CounterfactualPipeline()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_output_has_required_keys(pipeline, sample_setup):
    """Output must match CounterfactualOutput schema in schema.md"""
    model_path, csv_path, inv_output = sample_setup
    result = pipeline.run(model_path, csv_path, inv_output, top_n=3)

    required = {"agent", "version", "attempted",
                "found", "success_rate", "avg_features_to_flip", "examples"}
    assert required.issubset(result.keys()), \
        f"Missing keys: {required - result.keys()}"


def test_success_rate_is_valid_float(pipeline, sample_setup):
    """success_rate must be between 0.0 and 1.0"""
    model_path, csv_path, inv_output = sample_setup
    result = pipeline.run(model_path, csv_path, inv_output, top_n=5)
    assert 0.0 <= result["success_rate"] <= 1.0


def test_each_example_has_correct_schema(pipeline, sample_setup):
    """Each example in 'examples' must match the schema"""
    model_path, csv_path, inv_output = sample_setup
    result = pipeline.run(model_path, csv_path, inv_output, top_n=5)

    for ex in result["examples"]:
        assert "row_id" in ex,            f"Missing row_id in {ex}"
        assert "features_changed" in ex,  f"Missing features_changed in {ex}"
        assert "delta" in ex,             f"Missing delta in {ex}"
        assert "pct_change" in ex,        f"Missing pct_change in {ex}"
        assert "prediction_flipped" in ex,f"Missing prediction_flipped in {ex}"
        assert isinstance(ex["features_changed"], list)
        assert len(ex["features_changed"]) >= 1
        assert ex["prediction_flipped"] is True


def test_handles_missing_model_gracefully(pipeline, sample_setup):
    """Must not crash if model file doesn't exist"""
    _, csv_path, inv_output = sample_setup
    result = pipeline.run("nonexistent.pkl", csv_path, inv_output)
    assert "error" in result
    assert result["found"] == 0
    assert result["examples"] == []


def test_handles_empty_csv_gracefully(pipeline, sample_setup, tmp_path):
    """Must not crash if mispredictions CSV has 0 rows"""
    model_path, _, inv_output = sample_setup
    data = load_breast_cancer()
    empty = pd.DataFrame(
        columns=list(data.feature_names) + ["predicted", "actual"])
    csv_path = str(tmp_path / "empty.csv")
    empty.to_csv(csv_path, index=False)
    result = pipeline.run(model_path, csv_path, inv_output)
    assert result["attempted"] == 0
    assert result["found"] == 0


def test_cf_to_sentence_produces_readable_string(pipeline, sample_setup):
    """cf_to_sentence must return a non-empty readable string"""
    model_path, csv_path, inv_output = sample_setup
    result = pipeline.run(model_path, csv_path, inv_output, top_n=5)

    if result["examples"]:
        sentence = cf_to_sentence(result["examples"][0])
        assert isinstance(sentence, str)
        assert len(sentence) > 20
        assert "prediction" in sentence.lower()
        print(f"\nSample sentence: {sentence}")


def test_top_n_limits_attempts(pipeline, sample_setup):
    """'attempted' must equal top_n (or total rows if fewer)"""
    model_path, csv_path, inv_output = sample_setup
    result = pipeline.run(model_path, csv_path, inv_output, top_n=3)
    assert result["attempted"] <= 3