from src.load_data import load_data
from src.split_data import split_holdout
from src.train_model import train_logistic, train_xgboost
from src.predict import predict_logistic, predict_xgboost
from src.explain import explain_logistic, explain_xgboost
import pandas as pd
import numpy as np
import os


print("\nSTEP 1: Load data")
df = load_data()

print("\nSTEP 2: Split holdout rows")
train_df, holdout_df = split_holdout(df, n_holdout=5, seed=None)

print("\nHOLDOUT ROWS (UNSEEN INPUTS):")
print(holdout_df[["id", "diagnosis"]])

# Save for auditability
os.makedirs("data/samples", exist_ok=True)
holdout_df.to_csv("data/samples/holdout_used_this_run.csv", index=False)

print("\nSTEP 3: Prepare training data")
X_train = train_df.drop(columns=["id", "diagnosis"])
y_train = train_df["diagnosis"].map({"M": 1, "B": 0})

X_holdout = holdout_df.drop(columns=["id", "diagnosis"])

print("\nSTEP 4: Train Logistic Regression")
train_logistic(X_train, y_train)

print("\nSTEP 5: Train XGBoost")
train_xgboost(X_train, y_train)

print("\nSTEP 6: Predict")
log_probs = predict_logistic(X_holdout)
xgb_probs = predict_xgboost(X_holdout)

results = pd.DataFrame({
    "ID": holdout_df["id"].values,
    "True_Label": holdout_df["diagnosis"].values,
    "Logistic_Risk": log_probs,
    "XGBoost_Risk": xgb_probs,
})

results["Confidence"] = np.where(
    abs(results["Logistic_Risk"] - results["XGBoost_Risk"]) < 0.15,
    "HIGH",
    "LOW"
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


print("\n=== FINAL RISK REPORT ===")
print(results.round(4))

print("\nSTEP 7: EXPLANATION FOR FIRST HOLDOUT ROW")
X_one = X_holdout.iloc[[0]]

print("\nLogistic explanation (top drivers):")
print(explain_logistic(X_one, X_train).head(5))

print("\nXGBoost explanation (top drivers):")
print(explain_xgboost(X_one).head(5))




