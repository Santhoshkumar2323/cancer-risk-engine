import shap
import joblib
import pandas as pd


def explain_logistic(X_sample, X_train):
    """
    SHAP explanation for Logistic Regression inside a pipeline.
    Uses scaled training data as background.
    """
    pipeline = joblib.load("models/logistic_model.pkl")

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    # Scale background and sample
    X_train_scaled = scaler.transform(X_train)
    X_sample_scaled = scaler.transform(X_sample)

    explainer = shap.LinearExplainer(
        model,
        X_train_scaled
    )

    shap_values = explainer.shap_values(X_sample_scaled)

    return pd.Series(
        shap_values[0],
        index=X_sample.columns
    ).sort_values(key=abs, ascending=False)


def explain_xgboost(X_sample):
    """
    SHAP explanation for XGBoost model.
    """
    model = joblib.load("models/xgboost_model.pkl")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return pd.Series(
        shap_values[0],
        index=X_sample.columns
    ).sort_values(key=abs, ascending=False)
