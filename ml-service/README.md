# OptiServe — Duration Prediction ML Service

An independent Python microservice that predicts the **expected service
duration (in minutes)** for an OptiServe service stage.

This service is deliberately narrow in scope:

- ✅ Predicts service duration.
- ❌ Does **not** calculate queue waiting time.
- ❌ Does **not** determine scheduling priority.
- ❌ Does **not** access the OptiServe PostgreSQL database.
- ❌ Does **not** modify any queue, resource, or assignment state.

Predictions are **advisory**. The Spring Boot backend (`ServiceDurationPredictor`
abstraction) remains the authoritative system for all operational decisions.

---

## 1. Dataset

`data/service_history.csv` is currently a **synthetic** dataset
(`training/generate_synthetic_data.py`), generated because OptiServe does not
yet have enough real completed-assignment history to train on.

Each row simulates one completed service:

| Column             | Description                              |
| ------------------ | ----------------------------------------- |
| `serviceType`       | e.g. "Engine Service", "Oil Change"       |
| `vehicleMake`        | e.g. "Hyundai"                            |
| `vehicleModel`        | e.g. "i20"                                |
| `vehicleYear`         | Manufacturing year                        |
| `dayOfWeek`           | MONDAY–SUNDAY                             |
| `durationMinutes`     | Target — actual/simulated service duration |

Generation logic: a base duration per `serviceType`, adjusted by vehicle age
(older vehicles take slightly longer) and day-of-week load (Fri/Sat/Sun busier),
plus Gaussian noise so the relationship isn't perfectly linear.

**Migration plan:** this is intentionally temporary. Once the Spring Boot
backend has accumulated a meaningful volume of real `assignments` rows with
`actual_service_duration` filled in, export that data into the same
`serviceType,vehicleMake,vehicleModel,vehicleYear,dayOfWeek,durationMinutes`
shape, drop it into `data/service_history.csv`, and re-run `training/train.py`.
No code changes required — the schema and pipeline stay the same, only the
source of the rows changes. Real usage data will give sharper, more accurate
predictions than the synthetic approximation used for these first few months.

## 2. Feature engineering

`app/features.py` is the single source of truth for feature construction,
shared by both training and inference so they can never drift apart:

- `serviceType`, `vehicleMake`, `vehicleModel`, `dayOfWeek` → one-hot encoded
  (`handle_unknown="ignore"`, so an unseen make/model at inference time
  doesn't crash the request).
- `vehicleYear` → converted to `vehicleAge` (current year − `vehicleYear`,
  clipped at 0), which is a more directly useful signal than the raw year.

## 3. Training

```bash
pip install -r requirements.txt
python training/generate_synthetic_data.py   # (re)generate data/service_history.csv
python training/train.py
```

`train.py`:
1. Loads `data/service_history.csv`.
2. Builds features via `app/features.py`.
3. 80/20 train/validation split (`random_state=42` for reproducibility).
4. Trains two pipelines (preprocessing + model), each wrapped in a single
   sklearn `Pipeline` so preprocessing and the model are always persisted
   and loaded together:
   - **Linear Regression** (baseline)
   - **Random Forest Regressor** (200 trees)
5. Evaluates both on the validation split with MAE, RMSE, and R².
6. Selects the model with the **lower validation MAE**.
7. Refits the selected pipeline on the full labeled dataset.
8. Persists the winner and its fixed `reference_year` to
   `model/service_duration_model.joblib` via `joblib`.
9. Writes both models' validation metrics to `model/metrics.json` for reference.

## 4. Evaluation

`training/evaluate.py` holds the shared `evaluate_model()` used by `train.py`,
and can also be run standalone as a sanity check against the persisted model:

```bash
python training/evaluate.py
```

Metrics reported: **MAE** (mean absolute error, minutes), **RMSE** (root mean
squared error, minutes), **R²** (variance explained).

## 5. Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

### `POST /predict-duration`

Request:

```json
{
  "serviceType": "Engine Service",
  "vehicleMake": "Hyundai",
  "vehicleModel": "i20",
  "vehicleYear": 2022,
  "dayOfWeek": "MONDAY"
}
```

Response:

```json
{
  "predictedDurationMinutes": 97
}
```

Invalid input (bad `dayOfWeek`, missing field, out-of-range `vehicleYear` outside
the supported 2008–2026 range) returns `422`. If the model is missing or cannot be loaded,
the API returns `503` with a descriptive error message.

### `GET /health`

Liveness probe — returns `{"status": "ok"}` when the FastAPI process is running.
It does not check or depend on ML model availability.

### `GET /ready`

Readiness probe — checks whether the trained ML model is available and can be loaded.
Returns `{"status": "ready"}` with HTTP 200 when ready to serve predictions, or HTTP 503
when the model is missing, unavailable, corrupt, or cannot be loaded.

## 6. Tests

```bash
pytest tests/ -v
```

- `tests/test_preprocessing.py` — feature engineering (vehicle age
  computation, negative-age clipping, categorical pass-through).
- `tests/test_predictor.py` — end-to-end API tests via FastAPI's
  `TestClient`: liveness (`/health`), readiness (`/ready`) when model is available
  vs unavailable/corrupt (HTTP 503), prediction availability and 503 handling,
  supported `vehicleYear` range acceptance (2008–2026) and boundary rejections (HTTP 422),
  positive integer predictions, deterministic outputs, invalid field handling,
  and graceful unseen categorical values.

## 7. Project structure

```
ml-service/
├── app/
│   ├── main.py           # FastAPI app + /predict-duration, /health, /ready
│   ├── schemas.py         # Pydantic request/response models + validation
│   ├── features.py         # Shared feature engineering + vehicle year range
│   ├── predictor.py         # Builds features, calls model, clamps output
│   └── model_loader.py       # Loads/caches the persisted joblib model
├── model/
│   ├── service_duration_model.joblib
│   └── metrics.json        # MAE/RMSE/R² for both candidate models
├── training/
│   ├── generate_synthetic_data.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   └── service_history.csv
├── tests/
│   ├── test_preprocessing.py
│   └── test_predictor.py
├── pytest.ini
├── requirements.txt
└── README.md
```


## 8. Explicitly out of scope for this phase

- No database access of any kind.
- No queue/resource/assignment mutation.
- No authentication.
- No notifications/WhatsApp.
- No changes to the Spring Boot backend (integration is a separate,
  future issue — this service is self-contained and independently runnable).
