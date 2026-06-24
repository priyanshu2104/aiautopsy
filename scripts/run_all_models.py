"""
Runs InvestigatorPipeline on all 3 Kaggle models and prints a summary.
Use this to manually verify output quality before handing off to Member 2.

Usage: python scripts/run_all_models.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from src.investigator import InvestigatorPipeline

pipeline = InvestigatorPipeline()

MODELS = [
    ("fraud", "models/fraud_rf.pkl",
     "data/mispredictions/fraud_wrong.csv",
     "Credit Card Fraud Detector"),
    ("churn", "models/churn_rf.pkl",
     "data/mispredictions/churn_wrong.csv",
     "Telco Customer Churn Predictor"),
    ("loan",  "models/loan_rf.pkl",
     "data/mispredictions/loan_wrong.csv",
     "Credit Risk Loan Classifier"),
]

for name, model_path, csv_path, display_name in MODELS:
    print(f"\n{'='*50}")
    print(f"Model: {display_name}")
    result = pipeline.run(model_path, csv_path, model_name=display_name)

    if "error" in result:
        print(f"  ❌ ERROR: {result['error']}")
        continue

    print(f"  Total failures: {result['total_failures']}")
    print(f"  Top 3 failure-driving features:")
    for feat in result["top_features"]:
        print(f"    {feat['feature']}: SHAP={feat['mean_abs_shap']:.4f}")
    print(f"  Top failure pattern:")
    if result["failure_patterns"]:
        p = result["failure_patterns"][0]
        print(f"    {p['feature']} in {p['range']}: "
              f"{p['failure_count']} failures ({p['failure_rate']*100:.1f}%)")

    # Save result for Member 2 to use in Agent wiring
    out_path = f"output/investigator_{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

print("\n✅ All models processed. Check output/ folder.")