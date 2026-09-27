"""
Trains a Linear Regression baseline and a Random Forest Regressor on
data/service_history.csv, evaluates both on a held-out validation split,
and persists whichever model has the lower validation MAE to
model/service_duration_model.joblib.

Run: python training/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import app.*` when run as a script

from app.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DEFAULT_REFERENCE_YEAR,
    NUMERIC_FEATURES,
    build_feature_frame,
)
from training.evaluate import evaluate_model  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "service_history.csv"
MODEL_PATH = REPO_ROOT / "model" / "service_duration_model.joblib"
METRICS_PATH = REPO_ROOT / "model" / "metrics.json"

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)

    y = df["durationMinutes"]
    X = build_feature_frame(df, reference_year=DEFAULT_REFERENCE_YEAR)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=None,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_val, y_val)
        results[name] = metrics
        print(f"{name}: {metrics}")

    # Lower MAE wins.
    best_name = min(results, key=lambda name: results[name]["mae"])
    best_pipeline = candidates[best_name]
    best_pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "reference_year": DEFAULT_REFERENCE_YEAR,
        },
        MODEL_PATH,
    )

    METRICS_PATH.write_text(
        json.dumps({"selected_model": best_name, "results": results}, indent=2)
    )

    print(f"\nSelected model: {best_name}")
    print(f"Persisted to: {MODEL_PATH}")
    print(f"Metrics written to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
