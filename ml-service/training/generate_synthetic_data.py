"""
Generates a synthetic service_history.csv for training the duration model.

This is a stand-in for real OptiServe assignment history. Once the backend
has accumulated enough completed assignments (service_type, vehicle info,
day of week, actual_service_duration), swap this file out for a real export
and re-run training/train.py — nothing else needs to change.

Run: python training/generate_synthetic_data.py
"""

import csv
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.features import DEFAULT_REFERENCE_YEAR, MAX_VEHICLE_YEAR, MIN_VEHICLE_YEAR

random.seed(42)

# Base duration (minutes) per service type — this is the dominant signal.
SERVICE_TYPE_BASE_DURATION = {
    "Oil Change": 30,
    "General Inspection": 40,
    "Battery Replacement": 35,
    "Tire Replacement": 45,
    "Wheel Alignment": 50,
    "Brake Service": 60,
    "AC Repair": 70,
    "Diagnostics": 55,
    "Engine Service": 90,
    "Transmission Service": 120,
}

MAKE_MODELS = {
    "Hyundai": ["i20", "Creta", "Venue", "Verna"],
    "Maruti Suzuki": ["Swift", "Baleno", "Ertiga", "Brezza"],
    "Tata": ["Nexon", "Altroz", "Harrier", "Punch"],
    "Honda": ["City", "Amaze", "Elevate"],
    "Toyota": ["Innova", "Glanza", "Fortuner"],
    "Mahindra": ["XUV700", "Scorpio", "Bolero"],
    "Kia": ["Seltos", "Sonet"],
    "Ford": ["EcoSport", "Figo"],
}

DAYS_OF_WEEK = [
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
    "FRIDAY", "SATURDAY", "SUNDAY",
]

# Weekends run slightly busier/slower at a service center; small multiplier.
DAY_LOAD_FACTOR = {
    "MONDAY": 1.00, "TUESDAY": 1.00, "WEDNESDAY": 1.00,
    "THURSDAY": 1.00, "FRIDAY": 1.05,
    "SATURDAY": 1.15, "SUNDAY": 1.10,
}


def sample_row():
    service_type = random.choice(list(SERVICE_TYPE_BASE_DURATION.keys()))
    make = random.choice(list(MAKE_MODELS.keys()))
    model = random.choice(MAKE_MODELS[make])
    year = random.randint(MIN_VEHICLE_YEAR, MAX_VEHICLE_YEAR)
    day = random.choice(DAYS_OF_WEEK)

    base = SERVICE_TYPE_BASE_DURATION[service_type]
    vehicle_age = DEFAULT_REFERENCE_YEAR - year

    # Older vehicles take a bit longer (worn parts, more diagnostics).
    age_factor = 1.0 + min(vehicle_age, 15) * 0.012

    # Weekend/Friday load bump.
    day_factor = DAY_LOAD_FACTOR[day]

    # Gaussian noise to keep it realistic (not a perfectly learnable line).
    noise = random.gauss(0, base * 0.10)

    duration = base * age_factor * day_factor + noise
    duration = max(10, round(duration))  # floor at 10 minutes

    return {
        "serviceType": service_type,
        "vehicleMake": make,
        "vehicleModel": model,
        "vehicleYear": year,
        "dayOfWeek": day,
        "durationMinutes": duration,
    }


def main(n_rows: int = 4000):
    out_path = Path(__file__).resolve().parent.parent / "data" / "service_history.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [sample_row() for _ in range(n_rows)]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
