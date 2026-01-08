Cancer Risk Prediction Engine

This project implements a verified cancer risk prediction system using classical machine learning on the Breast Cancer Wisconsin (Diagnostic) dataset.

The system predicts malignancy risk probabilities from 30 numeric tumor measurements, using:

Logistic Regression (linear baseline)
XGBoost (non-linear model)

The focus is correctness and interpretability, not accuracy chasing.


Key design choices

Strict train–inference separation
Holdout rows are removed before training and treated as unseen inputs.

Model agreement for confidence
Risk confidence is based on agreement between Logistic and XGBoost.

Explainability by design
SHAP explanations are generated per prediction.

Behavioral verification
The model is tested by:

Increasing malignant features → risk increases
Neutralizing features → confidence drops

This verifies learning without blind trust.