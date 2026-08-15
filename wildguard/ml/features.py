"""Feature engineering shared by every model in the project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "confidence",
    "detected_count",
    "distance_to_boundary_km",
    "distance_to_road_km",
    "distance_to_village_km",
    "patrol_coverage",
    "hours_since_patrol",
    "temperature_c",
    "is_night",
    "hour_sin",
    "hour_cos",
    "boundary_proximity",
    "road_proximity",
]

CATEGORICAL_FEATURES = ["zone", "sensor_type", "signal", "signal_class", "weather"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "is_poaching_incident"

FEATURE_LABELS = {
    "confidence": "Detector confidence",
    "detected_count": "Objects detected",
    "distance_to_boundary_km": "Distance to reserve boundary",
    "distance_to_road_km": "Distance to road",
    "distance_to_village_km": "Distance to village",
    "patrol_coverage": "Ranger patrol coverage",
    "hours_since_patrol": "Hours since last patrol",
    "temperature_c": "Temperature",
    "is_night": "Night-time detection",
    "hour_sin": "Time of day (cyclical)",
    "hour_cos": "Time of day (cyclical)",
    "boundary_proximity": "Boundary proximity",
    "road_proximity": "Road proximity",
    "zone": "Reserve zone",
    "sensor_type": "Sensor type",
    "signal": "Detected signal",
    "signal_class": "Signal category",
    "weather": "Weather",
}


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Derive the model feature matrix (no target) from raw events."""
    df = events.copy()
    hour = df["timestamp"].dt.hour if "hour" not in df.columns else df["hour"]
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["boundary_proximity"] = 1.0 / (1.0 + df["distance_to_boundary_km"])
    df["road_proximity"] = 1.0 / (1.0 + df["distance_to_road_km"])

    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "unknown"
        df[column] = df[column].astype(str)

    return df[FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical ones."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def encoded_feature_names(pipeline: Pipeline) -> list[str]:
    """Readable names for the columns produced by the preprocessor."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def base_feature_of(encoded_name: str) -> str:
    """Map an encoded column (e.g. ``cat__zone_Core Sanctuary``) to its source feature."""
    name = encoded_name.split("__", 1)[-1]
    for feature in sorted(FEATURE_COLUMNS, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def pretty(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())
