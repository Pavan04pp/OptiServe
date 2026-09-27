from fastapi.testclient import TestClient

from app.features import MAX_VEHICLE_YEAR, MIN_VEHICLE_YEAR
from app.main import app
from app.model_loader import get_model_artifact

client = TestClient(app)

VALID_PAYLOAD = {
    "serviceType": "Engine Service",
    "vehicleMake": "Hyundai",
    "vehicleModel": "i20",
    "vehicleYear": 2022,
    "dayOfWeek": "MONDAY",
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_check_remains_200_when_model_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("app.model_loader.MODEL_PATH", tmp_path / "nonexistent.joblib")
    get_model_artifact.cache_clear()
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    finally:
        get_model_artifact.cache_clear()


def test_ready_returns_200_when_model_is_available():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_model_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("app.model_loader.MODEL_PATH", tmp_path / "nonexistent.joblib")
    get_model_artifact.cache_clear()
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert "detail" in body
        assert "No trained model found" in body["detail"]
    finally:
        get_model_artifact.cache_clear()


def test_ready_returns_503_when_model_is_corrupt(monkeypatch, tmp_path):
    corrupt_file = tmp_path / "corrupt_model.joblib"
    corrupt_file.write_text("not a valid joblib file")
    monkeypatch.setattr("app.model_loader.MODEL_PATH", corrupt_file)
    get_model_artifact.cache_clear()
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert "Failed to load model" in body["detail"]
    finally:
        get_model_artifact.cache_clear()


def test_predict_duration_returns_503_when_model_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("app.model_loader.MODEL_PATH", tmp_path / "nonexistent.joblib")
    get_model_artifact.cache_clear()
    try:
        response = client.post("/predict-duration", json=VALID_PAYLOAD)
        assert response.status_code == 503
        body = response.json()
        assert "detail" in body
        assert "No trained model found" in body["detail"]
    finally:
        get_model_artifact.cache_clear()


def test_predict_duration_returns_positive_integer():
    response = client.post("/predict-duration", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert "predictedDurationMinutes" in body
    assert isinstance(body["predictedDurationMinutes"], int)
    assert body["predictedDurationMinutes"] > 0


def test_predict_duration_is_deterministic():
    first = client.post("/predict-duration", json=VALID_PAYLOAD).json()
    second = client.post("/predict-duration", json=VALID_PAYLOAD).json()
    assert first == second


def test_predict_duration_accepts_valid_vehicle_years():
    for year in (MIN_VEHICLE_YEAR, 2022, MAX_VEHICLE_YEAR):
        payload = {**VALID_PAYLOAD, "vehicleYear": year}
        response = client.post("/predict-duration", json=payload)
        assert response.status_code == 200
        assert response.json()["predictedDurationMinutes"] > 0


def test_predict_duration_rejects_vehicle_year_below_range():
    for year in (MIN_VEHICLE_YEAR - 1, 1800):
        payload = {**VALID_PAYLOAD, "vehicleYear": year}
        response = client.post("/predict-duration", json=payload)
        assert response.status_code == 422
        assert f"between {MIN_VEHICLE_YEAR} and {MAX_VEHICLE_YEAR}" in response.text


def test_predict_duration_rejects_vehicle_year_above_range():
    for year in (MAX_VEHICLE_YEAR + 1, 2050):
        payload = {**VALID_PAYLOAD, "vehicleYear": year}
        response = client.post("/predict-duration", json=payload)
        assert response.status_code == 422
        assert f"between {MIN_VEHICLE_YEAR} and {MAX_VEHICLE_YEAR}" in response.text


def test_predict_duration_rejects_invalid_day_of_week():
    payload = {**VALID_PAYLOAD, "dayOfWeek": "FUNDAY"}
    response = client.post("/predict-duration", json=payload)
    assert response.status_code == 422


def test_predict_duration_rejects_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "vehicleMake"}
    response = client.post("/predict-duration", json=payload)
    assert response.status_code == 422


def test_predict_duration_rejects_unreasonable_year():
    payload = {**VALID_PAYLOAD, "vehicleYear": 1800}
    response = client.post("/predict-duration", json=payload)
    assert response.status_code == 422


def test_predict_duration_handles_unseen_categorical_gracefully():
    # A make/model the model never saw during training — OneHotEncoder
    # with handle_unknown="ignore" should still produce a valid prediction
    # rather than raising.
    payload = {**VALID_PAYLOAD, "vehicleMake": "Skoda", "vehicleModel": "Kushaq"}
    response = client.post("/predict-duration", json=payload)
    assert response.status_code == 200
    assert response.json()["predictedDurationMinutes"] > 0
