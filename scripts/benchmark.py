"""
Benchmark script — measures timing of each agent across all 3 models.
Run from project root: python3 scripts/benchmark.py

Produces a timing table you can paste into your README.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from src.investigator import InvestigatorPipeline
from src.counterfactual import CounterfactualPipeline

MODELS = [
    (
        "Credit Card Fraud Detector",
        "models/fraud_rf.pkl",
        "data/mispredictions/fraud_wrong.csv",
        {
            "top_features": [
                {"feature": "V14", "mean_abs_shap": 0.0638},
                {"feature": "V4",  "mean_abs_shap": 0.0366},
                {"feature": "V17", "mean_abs_shap": 0.0278},
            ]
        },
    ),
    (
        "Telco Churn Predictor",
        "models/churn_rf.pkl",
        "data/mispredictions/churn_wrong.csv",
        {
            "top_features": [
                {"feature": "Contract",       "mean_abs_shap": 0.0816},
                {"feature": "tenure",         "mean_abs_shap": 0.0687},
                {"feature": "OnlineSecurity", "mean_abs_shap": 0.0483},
            ]
        },
    ),
    (
        "Credit Risk Loan Classifier",
        "models/loan_rf.pkl",
        "data/mispredictions/loan_wrong.csv",
        {
            "top_features": [
                {"feature": "loan_grade",           "mean_abs_shap": 0.0790},
                {"feature": "loan_percent_income",  "mean_abs_shap": 0.0727},
                {"feature": "person_home_ownership","mean_abs_shap": 0.0580},
            ]
        },
    ),
]

inv_pipeline = InvestigatorPipeline()
cf_pipeline  = CounterfactualPipeline()

print(f"\n{'Model':<35} {'Agent1':>8} {'Agent2':>8} {'Total':>8} {'Failures':>10} {'CF%':>6}")
print("-" * 80)

results = []
for name, model_path, csv_path, inv_output in MODELS:

    # Time Agent 1
    t0  = time.time()
    inv = inv_pipeline.run(model_path, csv_path, model_name=name)
    t1  = time.time()
    agent1_time = t1 - t0

    # Time Agent 2
    t0 = time.time()
    cf = cf_pipeline.run(model_path, csv_path,
                         investigator_output=inv, top_n=10)
    t1 = time.time()
    agent2_time = t1 - t0

    total_time  = agent1_time + agent2_time
    cf_pct      = cf.get("success_rate", 0) * 100
    failures    = inv.get("total_failures", 0)

    print(f"{name:<35} {agent1_time:>7.1f}s {agent2_time:>7.1f}s "
          f"{total_time:>7.1f}s {failures:>10} {cf_pct:>5.0f}%")

    results.append({
        "model": name,
        "agent1_time_s": round(agent1_time, 2),
        "agent2_time_s": round(agent2_time, 2),
        "total_time_s":  round(total_time, 2),
        "total_failures": failures,
        "cf_success_rate": round(cf_pct, 1),
    })

# Save for README
with open("output/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Benchmark complete — saved to output/benchmark_results.json")
print("\nPaste this into your README eval table:")
print("\n| Model | Failures | Agent 1 | Agent 2 | Total | CF Success |")
print("|---|---|---|---|---|---|")
for r in results:
    print(f"| {r['model']} | {r['total_failures']} | "
          f"{r['agent1_time_s']}s | {r['agent2_time_s']}s | "
          f"{r['total_time_s']}s | {r['cf_success_rate']}% |")