"""
Integration tests — verify the full pipeline end-to-end.
These tests use REAL model files (not test fixtures).
They take longer (~10-20s each) but catch real-world failures.

Run with: pytest tests/test_integration.py -v
Or with all tests: pytest tests/ -v
"""
import pytest
import time
import os
from src.graph import build_graph, make_initial_state


# Skip these tests if real model files don't exist
# (e.g. on CI without the Kaggle data)
FRAUD_MODEL   = "models/fraud_rf.pkl"
CHURN_MODEL   = "models/churn_rf.pkl"
LOAN_MODEL    = "models/loan_rf.pkl"
FRAUD_CSV     = "data/mispredictions/fraud_wrong.csv"
CHURN_CSV     = "data/mispredictions/churn_wrong.csv"
LOAN_CSV      = "data/mispredictions/loan_wrong.csv"

skip_if_no_models = pytest.mark.skipif(
    not os.path.exists(FRAUD_MODEL),
    reason="Real Kaggle models not available — run scripts/train_models.py first"
)


@pytest.fixture(scope="module")
def compiled_graph():
    return build_graph()


# ── Test 1: Full pipeline completes ──────────────────────────────────────────

@skip_if_no_models
def test_full_pipeline_fraud(compiled_graph):
    """
    Full Agent 1 → Agent 2 → Agent 3 on the fraud model.
    Verifies: no crash, all outputs populated, CF found.
    """
    state  = make_initial_state(FRAUD_MODEL, FRAUD_CSV,
                                "Credit Card Fraud Detector")
    result = compiled_graph.invoke(state)

    # No pipeline error
    assert result.get("error") is None, \
        f"Pipeline error: {result.get('error')}"

    # Agent 1 output
    inv = result["investigator_output"]
    assert inv is not None
    assert inv["total_failures"] > 0
    assert len(inv["top_features"]) >= 1
    assert inv["top_features"][0]["feature"] == "V14", \
        "Expected V14 as top feature for fraud model"

    # Agent 2 output
    cf = result["counterfactual_output"]
    assert cf is not None
    assert cf["attempted"] == 10
    assert cf["found"] >= 5, \
        f"Expected ≥5 CFs found, got {cf['found']}"
    assert cf["success_rate"] >= 0.5

    # Agent 3 output (placeholder for now)
    rep = result["report_output"]
    assert rep is not None


@skip_if_no_models
def test_full_pipeline_churn(compiled_graph):
    """Full pipeline on churn model."""
    state  = make_initial_state(CHURN_MODEL, CHURN_CSV,
                                "Telco Churn Predictor")
    result = compiled_graph.invoke(state)

    assert result.get("error") is None
    inv = result["investigator_output"]
    assert inv["total_failures"] > 0
    # Churn top feature should be Contract based on our model
    top_feat = inv["top_features"][0]["feature"]
    assert top_feat in ("Contract", "tenure", "OnlineSecurity",
                        "MonthlyCharges"), \
        f"Unexpected top feature: {top_feat}"

    cf = result["counterfactual_output"]
    assert cf["found"] >= 4


@skip_if_no_models
def test_pipeline_completes_within_time_limit(compiled_graph):
    """
    Full pipeline on churn must complete in under 30 seconds.
    If this fails, the batch prediction optimization isn't working.
    """
    start  = time.time()
    state  = make_initial_state(CHURN_MODEL, CHURN_CSV, "Churn")
    result = compiled_graph.invoke(state)
    elapsed = time.time() - start

    assert result.get("error") is None
    assert elapsed < 30, \
        (f"Pipeline took {elapsed:.1f}s — should be under 30s. "
         f"Check that _find_counterfactual_batch() is being used "
         f"in src/counterfactual.py")


@skip_if_no_models
def test_pipeline_graceful_on_empty_csv(compiled_graph, tmp_path):
    """
    Pipeline should return cleanly when mispredictions CSV has 0 rows.
    Should NOT error — should just report 0 failures.
    """
    import pandas as pd
    import joblib
    import numpy as np

    # Load churn model to get column names
    model = joblib.load(CHURN_MODEL)
    original = pd.read_csv(CHURN_CSV)
    feature_cols = [c for c in original.columns
                    if c not in ("predicted", "actual")]

    # Create empty CSV with correct columns
    empty = pd.DataFrame(columns=feature_cols + ["predicted", "actual"])
    empty_path = str(tmp_path / "empty.csv")
    empty.to_csv(empty_path, index=False)

    state  = make_initial_state(CHURN_MODEL, empty_path, "Churn Empty")
    result = compiled_graph.invoke(state)

    # Should not error
    assert result.get("error") is None
    inv = result["investigator_output"]
    assert inv["total_failures"] == 0
    # CF and reporter should be skipped
    assert result.get("counterfactual_output") is None or \
           result["counterfactual_output"].get("attempted", 0) == 0


@skip_if_no_models
def test_timing_tracked_in_state(compiled_graph):
    """Timing dict should be populated after pipeline runs."""
    state  = make_initial_state(CHURN_MODEL, CHURN_CSV, "Churn")
    result = compiled_graph.invoke(state)

    timing = result.get("timing", {})
    assert "agent1_s" in timing, "Agent 1 timing not tracked"
    assert "agent2_s" in timing, "Agent 2 timing not tracked"
    assert timing["agent1_s"] > 0
    assert timing["agent2_s"] > 0
    print(f"\nTiming: Agent1={timing['agent1_s']}s, "
          f"Agent2={timing['agent2_s']}s")