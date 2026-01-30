Cancer Risk Engine:

What this project is?

This is an exploratory cancer risk estimation system.

It is not a medical device, diagnostic tool, or clinical decision system.
It is a controlled modeling and explanation pipeline designed for learning, analysis, and auditability.

What problem it explores

In sensitive domains like healthcare, model accuracy alone is insufficient.

This project explores three hard questions:

How stable are predictions across different model families?

How should disagreement between models be interpreted?

How can individual predictions be explained transparently?

The system is intentionally designed to surface uncertainty

System overview:

The pipeline is executed end-to-end via a single orchestration script.

1. Data loading:

Loads the Wisconsin Diagnostic Breast Cancer (WDBC) dataset

Applies column naming

Keeps the dataset structure unchanged

2. Holdout-first evaluation:

A small set of rows is held out before training

These rows are treated as unseen inputs

The exact holdout used in each run is saved to disk for auditability

This prevents accidental “peeking” during modeling.

3. Model training (two independent models)

Two fundamentally different models are trained:

Logistic Regression

Standardized inputs

Linear, interpretable.

XGBoost

Non-linear decision tree

Captures interaction effects

Both models are trained on the same data and saved.

4. Risk prediction

For each unseen holdout sample:

Both models produce a probability estimate

Predictions are compared side-by-side

The system does not force a single answer.

5. Confidence via disagreement

Confidence is derived from model agreement, not probability magnitude.

If Logistic and XGBoost predictions differ by less than a fixed margin → HIGH confidence

Otherwise → LOW confidence

This treats disagreement as a first-class signal of uncertainty.

6. Explainability (local, per-sample)

For individual predictions:

Logistic Regression

SHAP Linear Explainer

Uses scaled training data as background

XGBoost

SHAP Tree Explainer

Non-linear feature contributions

Top contributing features are displayed for inspection.

Execution flow:

Running run_pipeline.py performs:

Load dataset

Split holdout rows

Train both models

Generate predictions on unseen samples

Compute confidence from disagreement

Produce feature-level explanations

All steps are deterministic except the optional random seed for holdout selection.

Key design decisions:

Holdout-first workflow to preserve evaluation integrity

Model diversity to expose instability

Disagreement-based confidence, not probability

What this system does NOT do:

No medical diagnosis

No clinical validation

No treatment recommendations

Predictions are analytical signals only.

Limitations:

Trained on a small, clean academic dataset

Confidence metric is heuristic, not statistical

SHAP explanations are descriptive

Ethical note:

-This project is designed to demonstrate modeling discipline, not to be used in real medical decision-making.

Any real-world use would require:

clinical validation

regulatory approval

domain expert oversight








