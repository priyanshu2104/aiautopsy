"""
Runs the pipeline 10 times in a row, logs every failure.
Target: 9/10 or better success rate.

Run from project root: python3 scripts/stress_test.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import random
from src.graph import build_graph, make_initial_state

app = build_graph()

MODELS = [
    ("models/fraud_rf.pkl",
     "data/mispredictions/fraud_wrong.csv",
     "Credit Card Fraud Detector"),
    ("models/churn_rf.pkl",
     "data/mispredictions/churn_wrong.csv",
     "Telco Churn Predictor"),
    ("models/loan_rf.pkl",
     "data/mispredictions/loan_wrong.csv",
     "Credit Risk Loan Classifier"),
]

successes    = 0
failures     = 0
failure_log  = []
total_start  = time.time()

print(f"Running 10 pipeline tests...\n")

for i in range(10):
    model_path, csv_path, model_name = random.choice(MODELS)
    short_name = model_name.split()[0]

    try:
        start  = time.time()
        result = app.invoke(make_initial_state(
            model_path, csv_path, model_name))
        elapsed = round(time.time() - start, 1)

        if result.get("error"):
            raise Exception(result["error"])

        inv = result.get("investigator_output", {})
        cf  = result.get("counterfactual_output", {})

        print(f"  Run {i+1:2d}: ✓ {short_name:<8} "
              f"{inv.get('total_failures',0):4d} failures, "
              f"{cf.get('found',0)}/{cf.get('attempted',0)} CFs "
              f"({elapsed}s)")
        successes += 1

    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"  Run {i+1:2d}: ❌ {short_name:<8} FAILED ({elapsed}s): {e}")
        failure_log.append({"run": i + 1, "model": model_name, "error": str(e)})
        failures += 1

total_elapsed = round(time.time() - total_start, 1)
print(f"\n{'='*50}")
print(f"Results: {successes}/10 succeeded ({failures} failed)")
print(f"Total time: {total_elapsed}s (avg {total_elapsed/10:.1f}s per run)")

if failure_log:
    print(f"\nFailures:")
    for f in failure_log:
        print(f"  Run {f['run']}: {f['model']} — {f['error']}")

if successes >= 9:
    print(f"\n✅ Stress test passed ({successes}/10 ≥ 9)")
else:
    print(f"\n❌ Stress test failed ({successes}/10 < 9) — fix errors above")