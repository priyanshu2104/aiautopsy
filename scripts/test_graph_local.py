"""
Quick local test of the full LangGraph pipeline (updated for Week 3).
Run from project root: python3 scripts/test_graph_local.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

from src.graph import build_graph, make_initial_state
from src.counterfactual import cf_to_sentence

print("Building graph...")
app = build_graph()
print("Graph compiled OK\n")

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

all_ok = True

for model_path, csv_path, model_name in MODELS:
    print(f"\n{'='*55}")
    print(f"Testing: {model_name}")

    start = time.time()
    result = app.invoke(make_initial_state(model_path, csv_path, model_name))
    elapsed = time.time() - start

    if result.get("error"):
        print(f"  ❌ Pipeline error: {result['error']}")
        all_ok = False
        continue

    inv = result["investigator_output"]
    cf  = result["counterfactual_output"]
    rep = result["report_output"]

    print(f"  Agent 1 ✓ — {inv['total_failures']} failures, "
          f"top: {inv['top_features'][0]['feature']}")
    print(f"  Agent 2 ✓ — {cf.get('found', 0)}/{cf.get('attempted', 0)} "
          f"CFs found ({cf.get('success_rate', 0)*100:.0f}% success)")

    if cf.get("examples"):
        print(f"  Example: {cf_to_sentence(cf['examples'][0])}")

    print(f"  Agent 3 ✓ — {rep.get('note', 'ok')}")
    print(f"  Time: {elapsed:.1f}s")

print()
if all_ok:
    print("✅ All 3 models passed — Agent 1 → Agent 2 → Agent 3 working")
else:
    print("❌ Some models failed — check errors above")