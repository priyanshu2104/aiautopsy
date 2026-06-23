"""
Run this once to confirm SHAP is installed correctly.
Usage: python scripts/verify_shap.py
Expected output: a matplotlib plot window opens showing SHAP summary
"""
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

print("Importing libraries... OK")

# Use sklearn's built-in dataset — no download needed
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("Model trained... OK")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("SHAP values computed... OK")

# This should open a plot window
shap.summary_plot(shap_values[1], X_test, show=False)
import matplotlib.pyplot as plt
plt.savefig("output/shap_verify.png", bbox_inches="tight", dpi=100)
plt.close()

print("\n✅ SHAP is working correctly!")
print("Check output/shap_verify.png to see your first SHAP plot.")