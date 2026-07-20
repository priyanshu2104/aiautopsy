# aiautopsy
A 3-agent system that autonomously investigates ML model failures using SHAP, counterfactual generation, and LLM-generated reports.

eval table:
| Model | Failures | Agent 1 | Agent 2 | Total | CF Success |
|---|---|---|---|---|---|
| Credit Card Fraud Detector | 18 | 0.05s | 0.86s | 0.91s | 80.0% |
| Telco Churn Predictor | 310 | 5.14s | 0.32s | 5.46s | 60.0% |
| Credit Risk Loan Classifier | 422 | 10.74s | 0.3s | 11.04s | 80.0% |