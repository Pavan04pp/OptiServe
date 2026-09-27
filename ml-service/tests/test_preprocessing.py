import pandas as pd

from app.features import DEFAULT_REFERENCE_YEAR, FEATURE_COLUMNS, build_feature_frame


def test_build_feature_frame_computes_vehicle_age():
    raw = pd.DataFrame(
        [
            {
                "serviceType": "Engine Service",
                "vehicleMake": "Hyundai",
                "vehicleModel": "i20",
                "vehicleYear": DEFAULT_REFERENCE_YEAR - 4,
                "dayOfWeek": "MONDAY",
            }
        ]
    )

    features = build_feature_frame(raw)

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "vehicleAge"] == 4


def test_build_feature_frame_clips_negative_age_to_zero():
    raw = pd.DataFrame(
        [
            {
                "serviceType": "Oil Change",
                "vehicleMake": "Tata",
                "vehicleModel": "Nexon",
                "vehicleYear": DEFAULT_REFERENCE_YEAR + 5,  # bogus future year
                "dayOfWeek": "FRIDAY",
            }
        ]
    )

    features = build_feature_frame(raw)

    assert features.loc[0, "vehicleAge"] == 0


def test_build_feature_frame_preserves_categorical_values():
    raw = pd.DataFrame(
        [
            {
                "serviceType": "Brake Service",
                "vehicleMake": "Honda",
                "vehicleModel": "City",
                "vehicleYear": 2020,
                "dayOfWeek": "SUNDAY",
            }
        ]
    )

    features = build_feature_frame(raw)

    assert features.loc[0, "serviceType"] == "Brake Service"
    assert features.loc[0, "vehicleMake"] == "Honda"
    assert features.loc[0, "vehicleModel"] == "City"
    assert features.loc[0, "dayOfWeek"] == "SUNDAY"
