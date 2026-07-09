"""
Standalone test of CounterfactualPipeline on all 3 models.
Run from project root: python3 scripts/run_counterfactual.py

Prints:
  - success rate per model
  - average pct change needed to flip
  - human-readable sentences for each example
  - timing per model
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from src.counterfactual import CounterfactualPipeline, cf_to_sentence

pipeline = CounterfactualPipeline()

MODELS = [
    (
        "fraud",
        "models/fraud_rf.pkl",
        "data/mispredictions/fraud_wrong.csv",
        "Credit Card Fraud Detector",
        {
            "top_features": [
                {"feature": "V14", "mean_abs_shap": 0.0638},
                {"feature": "V4",  "mean_abs_shap": 0.0366},
                {"feature": "V17", "mean_abs_shap": 0.0278},
            ]
        },
    ),
    (
        "churn",
        "models/churn_rf.pkl",
        "data/mispredictions/churn_wrong.csv",
        "Telco Customer Churn Predictor",
        {
            "top_features": [
                {"feature": "Contract",       "mean_abs_shap": 0.0816},
                {"feature": "tenure",         "mean_abs_shap": 0.0687},
                {"feature": "OnlineSecurity", "mean_abs_shap": 0.0483},
            ]
        },
    ),
    (
        "loan",
        "models/loan_rf.pkl",
        "data/mispredictions/loan_wrong.csv",
        "Credit Risk Loan Classifier",
        {
            "top_features": [
                {"feature": "loan_grade",           "mean_abs_shap": 0.0790},
                {"feature": "loan_percent_income",  "mean_abs_shap": 0.0727},
                {"feature": "person_home_ownership","mean_abs_shap": 0.0580},
            ]
        },
    ),
]

all_ok = True

for name, model_path, csv_path, display_name, inv_output in MODELS:
    print(f"\n{'='*55}")
    print(f"Model: {display_name}")

    start = time.time()
    result = pipeline.run(
        model_path=model_path,
        mispredictions_path=csv_path,
        investigator_output=inv_output,
        top_n=10,
    )
    elapsed = time.time() - start

    if "error" in result:
        print(f"  ❌ ERROR: {result['error']}")
        all_ok = False
        continue

    print(f"  Attempted : {result['attempted']}")
    print(f"  Found     : {result['found']}")
    print(f"  Success   : {result['success_rate']*100:.0f}%")
    print(f"  Avg feats : {result['avg_features_to_flip']}")
    print(f"  Time      : {elapsed:.1f}s")

    if result["examples"]:
        print(f"\n  Example counterfactuals:")
        for ex in result["examples"][:3]:
            print(f"    • {cf_to_sentence(ex)}")

    # Save for use by Agent 3 later
    out_path = f"output/counterfactual_{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")

print()
if all_ok:
    print("✅ All 3 models processed successfully")
else:
    print("❌ Some models had errors — check above")