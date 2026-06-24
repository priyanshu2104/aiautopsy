"""
Run this once to confirm SHAP is installed correctly.
Usage: python3 scripts/verify_shap.py
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

os.makedirs("output", exist_ok=True)

print("Step 1: Importing libraries... OK")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("Step 2: Model trained... OK")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("Step 3: SHAP values computed... OK")

# Fix: handle both old SHAP (list of arrays) and new SHAP (single 3D array)
if isinstance(shap_values, list):
    # Old SHAP style: shap_values is [class_0_array, class_1_array]
    sv_to_plot = shap_values[1]
else:
    # New SHAP style: shap_values is a 3D array (rows, features, classes)
    sv_to_plot = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

print(f"Step 4: SHAP array shape = {np.array(sv_to_plot).shape}, X_test shape = {X_test.shape}")

shap.summary_plot(sv_to_plot, X_test, show=False)
plt.savefig("output/shap_verify.png", bbox_inches="tight", dpi=100)
plt.close()
print("Step 5: Plot saved to output/shap_verify.png... OK")

print("\n✅ SHAP is working correctly!")