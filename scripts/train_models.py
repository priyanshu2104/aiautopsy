"""
Trains 3 baseline RandomForest classifiers and saves:
  - models/{name}_rf.pkl        trained model
  - data/mispredictions/{name}_wrong.csv   all rows the model got wrong

Usage: python scripts/train_models.py
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("data/mispredictions", exist_ok=True)


def train_and_save(name, X, y, verbose=True):
    """Train a RandomForest, save the model, save mispredictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Save model
    model_path = f"models/{name}_rf.pkl"
    joblib.dump(model, model_path)

    # Get predictions
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    if verbose:
        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Test size: {len(X_test)} rows")
        print(classification_report(y_test, y_pred, zero_division=0))

    # Extract mispredictions
    wrong_mask = y_pred != y_test
    wrong_X = X_test[wrong_mask].copy()
    wrong_X["predicted"] = y_pred[wrong_mask]
    wrong_X["actual"] = y_test[wrong_mask].values
    wrong_path = f"data/mispredictions/{name}_wrong.csv"
    wrong_X.to_csv(wrong_path, index=False)

    print(f"Mispredictions: {wrong_mask.sum()} rows saved to {wrong_path}")
    print(f"Model saved to: {model_path}")
    return model


# ── 1. CREDIT CARD FRAUD ─────────────────────────────────────────────────────
print("\nLoading Credit Card Fraud dataset...")
fraud_df = pd.read_csv("data/raw/creditcard.csv")

# Use a balanced sample (full dataset is 99.8% non-fraud which makes training slow)
fraud_pos = fraud_df[fraud_df["Class"] == 1]          # 492 fraud cases
fraud_neg = fraud_df[fraud_df["Class"] == 0].sample(
    n=5000, random_state=42)                           # 5000 non-fraud
fraud_balanced = pd.concat([fraud_pos, fraud_neg]).sample(frac=1, random_state=42)

X_fraud = fraud_balanced.drop("Class", axis=1)
y_fraud = fraud_balanced["Class"]
fraud_model = train_and_save("fraud", X_fraud, y_fraud)


# ── 2. TELCO CUSTOMER CHURN ───────────────────────────────────────────────────
print("\nLoading Telco Churn dataset...")

# Try both possible filenames
import glob
churn_files = glob.glob("data/raw/*Telco*") + glob.glob("data/raw/*churn*") + glob.glob("data/raw/*Churn*")
if not churn_files:
    raise FileNotFoundError("Churn CSV not found in data/raw/ — check the filename")
churn_df = pd.read_csv(churn_files[0])

# Clean up
churn_df = churn_df.dropna()
churn_df["TotalCharges"] = pd.to_numeric(
    churn_df["TotalCharges"], errors="coerce")
churn_df = churn_df.dropna()

# Encode all categorical columns
le = LabelEncoder()
churn_encoded = churn_df.copy()
for col in churn_encoded.select_dtypes(include="object").columns:
    churn_encoded[col] = le.fit_transform(churn_encoded[col].astype(str))

X_churn = churn_encoded.drop(["customerID", "Churn"], axis=1,
                               errors="ignore")
# If customerID column has different name, just drop all string columns
X_churn = X_churn.select_dtypes(include=[np.number])
y_churn = churn_encoded["Churn"]
churn_model = train_and_save("churn", X_churn, y_churn)


# ── 3. CREDIT RISK / LOAN DEFAULT ────────────────────────────────────────────
print("\nLoading Credit Risk dataset...")

loan_files = glob.glob("data/raw/*credit_risk*") + glob.glob("data/raw/*loan*") + glob.glob("data/raw/*credit*")
if not loan_files:
    raise FileNotFoundError("Loan CSV not found in data/raw/ — check the filename")
loan_df = pd.read_csv(loan_files[0])
loan_df = loan_df.dropna()

# Encode categoricals
loan_encoded = loan_df.copy()
for col in loan_encoded.select_dtypes(include="object").columns:
    loan_encoded[col] = le.fit_transform(loan_encoded[col].astype(str))

# Target column is usually 'loan_status'
target_col = "loan_status" if "loan_status" in loan_encoded.columns else loan_encoded.columns[-1]
X_loan = loan_encoded.drop(target_col, axis=1).select_dtypes(include=[np.number])
y_loan = loan_encoded[target_col]
loan_model = train_and_save("loan", X_loan, y_loan)


print("\n" + "="*40)
print("✅ All 3 models trained and saved!")
print("Models:         ", [f for f in os.listdir("models") if f.endswith(".pkl")])
print("Mispredictions: ", os.listdir("data/mispredictions"))