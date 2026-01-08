from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

def train_logistic(X, y):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, "models/logistic_model.pkl")
    return pipeline

def train_xgboost(X, y):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, "models/xgboost_model.pkl")
    return model
