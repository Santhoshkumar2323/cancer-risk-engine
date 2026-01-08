import joblib
import pandas as pd

def predict_logistic(X):
    model = joblib.load("models/logistic_model.pkl")
    return model.predict_proba(X)[:, 1]

def predict_xgboost(X):
    model = joblib.load("models/xgboost_model.pkl")
    return model.predict_proba(X)[:, 1]
