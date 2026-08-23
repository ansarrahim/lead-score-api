"""
Trains a lead-conversion classifier on the public "X Education" lead scoring
dataset (~9,200 real leads from an online course marketing campaign).

Source: https://github.com/drajesh-tech/Logistic-Regression-Lead-Scoring-Case-Study
This is a well-known, publicly shared real-world lead scoring dataset used
widely for ML case studies. Columns filled in by sales reps *after* contact
(Tags, Lead Quality, Last Notable Activity) are deliberately excluded to
avoid leakage — the model only sees signals available at lead-scoring time.
"""

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]
CATEGORICAL_FEATURES = [
    "Lead Origin",
    "Lead Source",
    "Do Not Email",
    "Last Activity",
    "Specialization",
    "What is your current occupation",
]
TARGET = "Converted"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # "Select" is the dataset's literal placeholder for an unanswered dropdown —
    # it's not a real category, treat it as missing.
    df = df.replace("Select", pd.NA)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = GradientBoostingClassifier(random_state=42)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    df = load_and_clean("data/Leads.csv")
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "conversion_rate": round(y.mean(), 4),
    }
    print(json.dumps(metrics, indent=2))

    # Also fit a logistic regression as an interpretable baseline for comparison
    baseline = build_pipeline()
    baseline.set_params(model=LogisticRegression(max_iter=1000))
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = {
        "accuracy": round(accuracy_score(y_test, baseline_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1]), 4),
    }
    print("Logistic regression baseline:", json.dumps(baseline_metrics, indent=2))

    joblib.dump(pipeline, "model/lead_score_model.joblib")
    with open("model/metrics.json", "w") as f:
        json.dump({"gradient_boosting": metrics, "logistic_regression_baseline": baseline_metrics}, f, indent=2)
    print("\nSaved model to model/lead_score_model.joblib")


if __name__ == "__main__":
    main()
