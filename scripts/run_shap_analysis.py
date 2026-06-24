"""
Runs SHAP analysis on all 3 models' mispredictions.
Saves per-row top-3 features to output/shap_{name}.json

Usage: python scripts/run_shap_analysis.py
"""
import shap
import pandas as pd
import numpy as np
import joblib
import json
import os

os.makedirs("output", exist_ok=True)

MODELS = [
    ("fraud", "models/fraud_rf.pkl",  "data/mispredictions/fraud_wrong.csv"),
    ("churn", "models/churn_rf.pkl",  "data/mispredictions/churn_wrong.csv"),
    ("loan",  "models/loan_rf.pkl",   "data/mispredictions/loan_wrong.csv"),
]


def analyse_model(name, model_path, mispred_path):
    print(f"\n{'='*40}")
    print(f"Analysing: {name}")

    model = joblib.load(model_path)
    wrong = pd.read_csv(mispred_path)

    # Separate feature columns from metadata columns
    feature_cols = [c for c in wrong.columns
                    if c not in ("predicted", "actual")]
    X_wrong = wrong[feature_cols]

    print(f"  Mispredictions: {len(X_wrong)} rows")
    print(f"  Features: {len(feature_cols)} columns")

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_wrong)

    # Handle both binary and multiclass output
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 (positive class)
    else:
        sv = shap_values

    # Per-row: top 3 features by absolute SHAP value
    per_row_results = []
    for i in range(len(X_wrong)):
        row_sv = sv[i]
        top3_idx = np.argsort(np.abs(row_sv))[::-1][:3]
        per_row_results.append({
            "row_id": int(wrong.index[i]),
            "predicted": int(wrong["predicted"].iloc[i]),
            "actual": int(wrong["actual"].iloc[i]),
            "top_features": [
                {
                    "feature": str(feature_cols[j]),
                    "shap_value": round(float(row_sv[j]), 4),
                    "feature_value": round(float(X_wrong.iloc[i, j]), 4)
                }
                for j in top3_idx
            ]
        })

    # Global: mean absolute SHAP across all mispredictions
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_global_idx = np.argsort(mean_abs_shap)[::-1][:5]
    global_top = [
        {
            "feature": str(feature_cols[j]),
            "mean_abs_shap": round(float(mean_abs_shap[j]), 4)
        }
        for j in top_global_idx
    ]

    # Failure clustering: for each top feature, which value range fails most?
    failure_patterns = []
    for feat_info in global_top[:3]:  # cluster by top 3 global features
        feat = feat_info["feature"]
        if feat in X_wrong.columns:
            try:
                bins = pd.cut(X_wrong[feat], bins=5)
                counts = wrong.groupby(bins, observed=True).size()
                counts = counts.sort_values(ascending=False)
                for b, count in counts.items():
                    if count > 2:
                        failure_patterns.append({
                            "feature": feat,
                            "range": str(b),
                            "failure_count": int(count),
                            "failure_rate": round(
                                float(count) / len(X_wrong), 3)
                        })
            except Exception:
                pass

    output = {
        "model": name,
        "total_failures": len(X_wrong),
        "global_top_features": global_top,
        "failure_patterns": failure_patterns[:5],  # top 5 patterns
        "per_row_shap": per_row_results[:20]        # first 20 rows
    }

    out_path = f"output/shap_{name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Top global failure features:")
    for feat in global_top[:3]:
        print(f"    {feat['feature']}: {feat['mean_abs_shap']:.4f}")
    print(f"  Saved to: {out_path}")
    return output


all_results = {}
for name, model_path, mispred_path in MODELS:
    try:
        result = analyse_model(name, model_path, mispred_path)
        all_results[name] = result
    except Exception as e:
        print(f"  ERROR on {name}: {e}")

print("\n✅ SHAP analysis complete!")
print("Results saved to:")
for name in all_results:
    print(f"  output/shap_{name}.json")