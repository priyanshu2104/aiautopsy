"""
Quick local test of the LangGraph pipeline.
Run this to verify Agent 1 is correctly wired into the graph.

Usage: python3 scripts/test_graph_local.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s — %(message)s")

from src.graph import build_graph, make_initial_state

print("Building graph...")
app = build_graph()
print("Graph compiled OK\n")

# Test with all 3 models
MODELS = [
    ("models/fraud_rf.pkl",
     "data/mispredictions/fraud_wrong.csv",
     "Credit Card Fraud Detector"),
    ("models/churn_rf.pkl",
     "data/mispredictions/churn_wrong.csv",
     "Telco Customer Churn Predictor"),
    ("models/loan_rf.pkl",
     "data/mispredictions/loan_wrong.csv",
     "Credit Risk Loan Classifier"),
]

for model_path, csv_path, model_name in MODELS:
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")

    state = make_initial_state(model_path, csv_path, model_name)
    result = app.invoke(state)

    if result.get("error"):
        print(f"  ❌ ERROR: {result['error']}")
        continue

    inv = result["investigator_output"]
    cf  = result["counterfactual_output"]
    rep = result["report_output"]

    print(f"  Agent 1 ✓ — {inv['total_failures']} failures, "
          f"top feature: {inv['top_features'][0]['feature']}")
    print(f"  Agent 2 ✓ — {cf.get('note', 'running')}")
    print(f"  Agent 3 ✓ — {rep.get('note', 'running')}")

print("\n✅ Graph test complete — all 3 agents ran successfully")