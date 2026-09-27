import pandas as pd

from app.features import build_feature_frame
from app.model_loader import get_model, get_reference_year
from app.schemas import PredictionRequest

# Predictions are advisory estimates, not ground truth — never let a noisy
# model output a non-physical (<=0) or unreasonably tiny duration.
MIN_DURATION_MINUTES = 5


def predict_duration(request: PredictionRequest) -> int:
    raw = pd.DataFrame(
        [
            {
                "serviceType": request.serviceType,
                "vehicleMake": request.vehicleMake,
                "vehicleModel": request.vehicleModel,
                "vehicleYear": request.vehicleYear,
                "dayOfWeek": request.dayOfWeek,
            }
        ]
    )
    features = build_feature_frame(raw, reference_year=get_reference_year())

    model = get_model()
    prediction = float(model.predict(features)[0])

    duration = max(MIN_DURATION_MINUTES, round(prediction))
    return duration
