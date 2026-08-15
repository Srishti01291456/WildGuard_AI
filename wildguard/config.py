"""Central configuration: paths, domain vocabulary and model settings."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

EVENTS_CSV = DATA_DIR / "field_events.csv"
PATROLS_CSV = DATA_DIR / "patrols.csv"

RISK_MODEL_PATH = MODEL_DIR / "poaching_risk_model.joblib"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"

RANDOM_SEED = 42

# --- Domain vocabulary -------------------------------------------------------

ZONES = [
    "Northern Corridor",
    "Riverine Belt",
    "Eastern Grassland",
    "Southern Buffer",
    "Core Sanctuary",
    "Western Foothills",
]

# Approximate reserve geography (synthetic, centred on a fictional reserve).
ZONE_CENTRES = {
    "Northern Corridor": (-1.180, 34.905),
    "Riverine Belt": (-1.245, 34.860),
    "Eastern Grassland": (-1.210, 35.010),
    "Southern Buffer": (-1.340, 34.945),
    "Core Sanctuary": (-1.265, 34.945),
    "Western Foothills": (-1.290, 34.830),
}

SENSOR_TYPES = ["camera_trap", "acoustic_sensor", "gps_collar", "drone_patrol", "ranger_report"]

WILDLIFE_SPECIES = ["elephant", "rhino", "tiger", "leopard", "zebra", "deer", "pangolin"]
ENDANGERED_SPECIES = {"rhino", "tiger", "pangolin", "elephant"}

HUMAN_SIGNALS = ["human_presence", "vehicle", "gunshot", "chainsaw", "campfire", "snare_trap"]
THREAT_SIGNALS = {"gunshot", "chainsaw", "snare_trap", "vehicle"}

SIGNALS = WILDLIFE_SPECIES + HUMAN_SIGNALS

WEATHER = ["clear", "cloudy", "rain", "fog"]

# --- Risk bands --------------------------------------------------------------

RISK_BANDS = [(0.75, "Critical"), (0.50, "High"), (0.25, "Medium"), (0.0, "Low")]
BAND_COLORS = {
    "Critical": "#ff4b4b",
    "High": "#ffa421",
    "Medium": "#00c0f2",
    "Low": "#21c354",
}


def risk_band(probability: float) -> str:
    """Map a 0-1 poaching probability onto a human readable band."""
    for threshold, label in RISK_BANDS:
        if probability >= threshold:
            return label
    return "Low"
