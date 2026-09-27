"""
Feature engineering shared by training/train.py and app/predictor.py.
Keeping this in one place guarantees train-time and inference-time features
are computed identically.
"""

import pandas as pd

CATEGORICAL_FEATURES = ["serviceType", "vehicleMake", "vehicleModel", "dayOfWeek"]
NUMERIC_FEATURES = ["vehicleAge"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
DEFAULT_REFERENCE_YEAR = 2026
MIN_VEHICLE_YEAR = 2008
MAX_VEHICLE_YEAR = 2026

VALID_DAYS_OF_WEEK = {
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
    "FRIDAY", "SATURDAY", "SUNDAY",
}


def build_feature_frame(raw: pd.DataFrame, reference_year: int = DEFAULT_REFERENCE_YEAR) -> pd.DataFrame:
    """
    Takes a DataFrame with raw columns
    [serviceType, vehicleMake, vehicleModel, vehicleYear, dayOfWeek]
    and returns a DataFrame with the exact columns/order the model pipeline
    was trained on: categoricals + engineered vehicleAge.
    """
    df = raw.copy()
    df["vehicleAge"] = reference_year - df["vehicleYear"]
    # Guard against future-dated years producing a negative age.
    df["vehicleAge"] = df["vehicleAge"].clip(lower=0)
    return df[FEATURE_COLUMNS]
