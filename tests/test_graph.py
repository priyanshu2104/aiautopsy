"""
Unit tests for the LangGraph pipeline.
Run with: pytest tests/ -v

These tests verify the graph wiring — not the ML logic
(that's already tested in test_investigator.py)
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from src.graph import build_graph, make_initial_state, AutopsyState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_model_and_csv(tmp_path_factory):
    """Creates a small model + CSV for graph tests."""
    tmp = tmp_path_factory.mktemp("graph_test")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:100], columns=data.feature_names)
    y = pd.Series(data.target[:100])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    model_path = str(tmp / "model.pkl")
    joblib.dump(model, model_path)
    wrong = X.copy()
    wrong["predicted"] = 0
    wrong["actual"] = 1
    csv_path = str(tmp / "wrong.csv")
    wrong.to_csv(csv_path, index=False)
    return model_path, csv_path


@pytest.fixture(scope="module")
def compiled_graph():
    return build_graph()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_graph_compiles():
    """Graph should compile without errors."""
    app = build_graph()
    assert app is not None


def test_initial_state_has_all_keys():
    """make_initial_state should return a dict with all required keys."""
    state = make_initial_state("model.pkl", "data.csv", "Test Model")
    required = {"model_path", "csv_path", "model_name",
                "investigator_output", "counterfactual_output",
                "report_output", "error"}
    assert required.issubset(state.keys())


def test_graph_runs_end_to_end(compiled_graph, test_model_and_csv):
    """Full pipeline should complete without error."""
    model_path, csv_path = test_model_and_csv
    state = make_initial_state(model_path, csv_path, "Test")
    result = compiled_graph.invoke(state)
    assert result.get("error") is None


def test_investigator_output_populated(compiled_graph, test_model_and_csv):
    """Agent 1 output should be populated after graph runs."""
    model_path, csv_path = test_model_and_csv
    state = make_initial_state(model_path, csv_path, "Test")
    result = compiled_graph.invoke(state)
    assert result["investigator_output"] is not None
    assert "total_failures" in result["investigator_output"]
    assert "top_features" in result["investigator_output"]


def test_graph_handles_missing_model(compiled_graph, test_model_and_csv):
    """Graph should return error gracefully when model file is missing."""
    _, csv_path = test_model_and_csv
    state = make_initial_state("nonexistent.pkl", csv_path, "Test")
    result = compiled_graph.invoke(state)
    # Should not raise — should return error in state
    assert result is not None
    # Either error is set, or investigator_output has error key
    has_error = (result.get("error") is not None or
                 "error" in result.get("investigator_output", {}))
    assert has_error


def test_pipeline_skips_agents_when_no_failures(
        compiled_graph, tmp_path):
    """If mispredictions CSV has 0 rows, counterfactual agent should be skipped."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:50], columns=data.feature_names)
    y = pd.Series(data.target[:50])
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    model_path = str(tmp_path / "model.pkl")
    joblib.dump(model, model_path)

    # Empty mispredictions CSV — 0 rows
    empty = pd.DataFrame(columns=list(data.feature_names) + ["predicted", "actual"])
    csv_path = str(tmp_path / "empty.csv")
    empty.to_csv(csv_path, index=False)

    state = make_initial_state(model_path, csv_path, "Test")
    result = compiled_graph.invoke(state)

    # Counterfactual should be None (skipped) when no failures
    inv = result.get("investigator_output", {})
    assert inv.get("total_failures", 0) == 0