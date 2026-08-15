"""CSV ingestion with validation, defaults and derived columns."""

from __future__ import annotations

from typing import IO

import numpy as np
import pandas as pd

from wildguard.config import DATA_DIR, EVENTS_CSV, PATROLS_CSV, RANDOM_SEED, ZONE_CENTRES
from wildguard.data.generator import generate_dataset, signal_class

REQUIRED_COLUMNS = ["timestamp", "zone", "signal"]

NUMERIC_DEFAULTS: dict[str, float] = {
    "confidence": 0.8,
    "detected_count": 1,
    "distance_to_boundary_km": 5.0,
    "distance_to_road_km": 5.0,
    "distance_to_village_km": 8.0,
    "patrol_coverage": 0.5,
    "hours_since_patrol": 24.0,
    "temperature_c": 24.0,
}


class DataValidationError(ValueError):
    """Raised when an uploaded file cannot be used as an event log."""


def bootstrap_dataset(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the sample dataset, generating it on first run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENTS_CSV.exists() or not PATROLS_CSV.exists():
        events, patrols = generate_dataset(seed=seed)
        events.to_csv(EVENTS_CSV, index=False)
        patrols.to_csv(PATROLS_CSV, index=False)
    return load_events(EVENTS_CSV), load_patrols(PATROLS_CSV)


def load_events(source: str | IO[bytes]) -> pd.DataFrame:
    """Read an event log and normalise it into the canonical schema."""
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - pandas raises many parser errors
        raise DataValidationError(f"Could not read the CSV file: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Event log is missing required column(s): {', '.join(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise DataValidationError("No rows left after parsing timestamps - check the timestamp format.")

    df["zone"] = df["zone"].astype(str).str.strip()
    df["signal"] = df["signal"].astype(str).str.strip().str.lower()
    if "signal_class" not in df.columns:
        df["signal_class"] = pd.Series(index=df.index, dtype=object)
    df["signal_class"] = df["signal_class"].fillna(df["signal"].map(signal_class)).astype(str)

    for column, default in NUMERIC_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    df["confidence"] = df["confidence"].clip(0, 1)
    df["patrol_coverage"] = df["patrol_coverage"].clip(0, 1)
    df["detected_count"] = df["detected_count"].clip(lower=1).astype(int)

    for column, default in (("sensor_type", "unknown"), ("sensor_id", "unknown"), ("weather", "clear")):
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default).astype(str)

    if "event_id" not in df.columns:
        df["event_id"] = [f"EVT-{i:06d}" for i in range(len(df))]

    df["latitude"] = _coords(df, "latitude", 0)
    df["longitude"] = _coords(df, "longitude", 1)

    df["hour"] = df["timestamp"].dt.hour
    if "is_night" in df.columns:
        df["is_night"] = pd.to_numeric(df["is_night"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_night"] = ((df["hour"] >= 19) | (df["hour"] <= 5)).astype(int)

    if "is_poaching_incident" in df.columns:
        df["is_poaching_incident"] = (
            pd.to_numeric(df["is_poaching_incident"], errors="coerce").fillna(0).clip(0, 1).astype(int)
        )

    df["date"] = df["timestamp"].dt.date
    return df.sort_values("timestamp").reset_index(drop=True)


def load_patrols(source: str | IO[bytes]) -> pd.DataFrame:
    df = pd.read_csv(source)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def _coords(df: pd.DataFrame, column: str, index: int) -> pd.Series:
    """Fill missing coordinates with the centre of the reported zone."""
    fallback = df["zone"].map(lambda z: ZONE_CENTRES.get(z, (np.nan, np.nan))[index])
    if column not in df.columns:
        return fallback
    return pd.to_numeric(df[column], errors="coerce").fillna(fallback)
