"""Synthetic but realistic field-telemetry generator for the reserve.

The generator produces two tables:

* ``events``  - one row per sensor detection (camera trap, acoustic sensor,
  GPS collar, drone or ranger report) with the contextual features the ML
  models learn from, plus a ``is_poaching_incident`` label.
* ``patrols`` - daily ranger patrol effort per zone, used for context and for
  the patrol-coverage feature.

Labels are drawn from a latent logistic process, so the supervised model has a
learnable - but noisy - signal rather than a leaky rule.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from wildguard.config import (
    ENDANGERED_SPECIES,
    HUMAN_SIGNALS,
    RANDOM_SEED,
    SENSOR_TYPES,
    THREAT_SIGNALS,
    WEATHER,
    WILDLIFE_SPECIES,
    ZONE_CENTRES,
    ZONES,
)

EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "zone",
    "latitude",
    "longitude",
    "sensor_id",
    "sensor_type",
    "signal",
    "signal_class",
    "confidence",
    "detected_count",
    "distance_to_boundary_km",
    "distance_to_road_km",
    "distance_to_village_km",
    "patrol_coverage",
    "hours_since_patrol",
    "is_night",
    "weather",
    "temperature_c",
    "is_poaching_incident",
]

SIGNAL_WEIGHTS = {
    "gunshot": 3.6,
    "chainsaw": 2.4,
    "snare_trap": 2.7,
    "vehicle": 1.7,
    "human_presence": 1.4,
    "campfire": 1.2,
}

# Zones closer to the reserve edge carry a higher baseline pressure.
ZONE_PRESSURE = {
    "Northern Corridor": 0.55,
    "Riverine Belt": 0.30,
    "Eastern Grassland": 0.15,
    "Southern Buffer": 0.65,
    "Core Sanctuary": -0.35,
    "Western Foothills": 0.20,
}


def signal_class(signal: str) -> str:
    """Group a raw detection label into wildlife / human / threat."""
    if signal in THREAT_SIGNALS:
        return "threat"
    if signal in HUMAN_SIGNALS:
        return "human"
    return "wildlife"


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(
    n_events: int = 6000,
    days: int = 90,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the ``(events, patrols)`` dataset."""
    rng = np.random.default_rng(seed)
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    zones = rng.choice(ZONES, size=n_events, p=_zone_mix())
    offsets = rng.random(n_events) * days * 24 * 3600
    timestamps = pd.to_datetime([start + timedelta(seconds=float(o)) for o in offsets])

    # Poaching pressure peaks at night, so night events are over-sampled.
    hours = timestamps.hour.to_numpy()
    is_night = ((hours >= 19) | (hours <= 5)).astype(int)

    sensor_type = rng.choice(SENSOR_TYPES, size=n_events, p=[0.4, 0.2, 0.15, 0.1, 0.15])
    signals = np.array([_draw_signal(rng, s, bool(n)) for s, n in zip(sensor_type, is_night, strict=True)])
    classes = np.array([signal_class(s) for s in signals])

    confidence = np.clip(rng.beta(6, 2, n_events), 0.35, 0.999).round(3)
    detected_count = np.clip(rng.poisson(1.4, n_events) + 1, 1, 12)

    dist_boundary = np.clip(rng.gamma(2.0, 1.6, n_events), 0.05, 18).round(2)
    dist_road = np.clip(rng.gamma(1.8, 1.4, n_events), 0.05, 15).round(2)
    dist_village = np.clip(rng.gamma(2.4, 2.0, n_events), 0.1, 25).round(2)

    patrol_coverage = np.clip(rng.beta(2.6, 2.2, n_events), 0.02, 0.99).round(3)
    hours_since_patrol = np.clip(rng.exponential(14, n_events) * (1.4 - patrol_coverage), 0.2, 160).round(1)

    weather = rng.choice(WEATHER, size=n_events, p=[0.55, 0.25, 0.14, 0.06])
    temperature = np.where(
        is_night == 1,
        rng.normal(17, 3.2, n_events),
        rng.normal(29, 4.0, n_events),
    ).round(1)

    zone_pressure = np.array([ZONE_PRESSURE[z] for z in zones])
    signal_score = np.array([SIGNAL_WEIGHTS.get(s, -0.6) for s in signals])
    endangered_bonus = np.array([0.35 if s in ENDANGERED_SPECIES else 0.0 for s in signals])

    logit = (
        -3.6
        + signal_score
        + endangered_bonus
        + 1.5 * is_night
        + 2.2 * (1 - patrol_coverage)
        + 1.6 / (1.0 + dist_boundary)
        + 1.2 / (1.0 + dist_road)
        + 0.7 / (1.0 + dist_village)
        + 1.2 * (confidence - 0.6)
        + 0.9 * np.log1p(hours_since_patrol) / 3
        + 1.4 * zone_pressure
        + rng.normal(0, 0.3, n_events)
    )
    label = (rng.random(n_events) < _sigmoid(logit)).astype(int)

    lat, lon = _jitter_coordinates(rng, zones)

    events = pd.DataFrame(
        {
            "event_id": [f"EVT-{i:06d}" for i in range(n_events)],
            "timestamp": timestamps,
            "zone": zones,
            "latitude": lat,
            "longitude": lon,
            "sensor_id": [f"{t[:3].upper()}-{rng.integers(1, 45):02d}" for t in sensor_type],
            "sensor_type": sensor_type,
            "signal": signals,
            "signal_class": classes,
            "confidence": confidence,
            "detected_count": detected_count,
            "distance_to_boundary_km": dist_boundary,
            "distance_to_road_km": dist_road,
            "distance_to_village_km": dist_village,
            "patrol_coverage": patrol_coverage,
            "hours_since_patrol": hours_since_patrol,
            "is_night": is_night,
            "weather": weather,
            "temperature_c": temperature,
            "is_poaching_incident": label,
        },
        columns=EVENT_COLUMNS,
    )

    events = pd.concat([events, _gunshot_cluster(rng, end)], ignore_index=True)
    events = events.sort_values("timestamp").reset_index(drop=True)
    return events, _generate_patrols(rng, start, days)


def _zone_mix() -> list[float]:
    weights = np.array([0.2, 0.16, 0.18, 0.16, 0.16, 0.14])
    return list(weights / weights.sum())


def _draw_signal(rng: np.random.Generator, sensor_type: str, night: bool) -> str:
    """Sensors see different things - collars only see wildlife, mics hear sounds."""
    if sensor_type == "gps_collar":
        return str(rng.choice(WILDLIFE_SPECIES))
    if sensor_type == "acoustic_sensor":
        return str(rng.choice(["gunshot", "chainsaw", "vehicle", "human_presence", "elephant"],
                              p=[0.16, 0.14, 0.2, 0.25, 0.25]))
    human_prob = 0.32 if night else 0.18
    if rng.random() < human_prob:
        return str(rng.choice(HUMAN_SIGNALS, p=[0.34, 0.22, 0.08, 0.08, 0.13, 0.15]))
    return str(rng.choice(WILDLIFE_SPECIES, p=[0.18, 0.08, 0.07, 0.09, 0.22, 0.28, 0.08]))


def _jitter_coordinates(rng: np.random.Generator, zones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres = np.array([ZONE_CENTRES[z] for z in zones])
    jitter = rng.normal(0, 0.018, size=centres.shape)
    coords = centres + jitter
    return coords[:, 0].round(5), coords[:, 1].round(5)


def _gunshot_cluster(rng: np.random.Generator, end: datetime) -> pd.DataFrame:
    """Plant a recent night-time incident so the alert engine has true positives."""
    base = end - timedelta(hours=8)
    lat0, lon0 = ZONE_CENTRES["Southern Buffer"]
    rows = []
    for i in range(14):
        signal = "gunshot" if i % 3 == 0 else str(rng.choice(["vehicle", "human_presence", "snare_trap"]))
        rows.append(
            {
                "event_id": f"EVT-INC-{i:03d}",
                "timestamp": base + timedelta(minutes=i * 9),
                "zone": "Southern Buffer",
                "latitude": round(lat0 + rng.normal(0, 0.004), 5),
                "longitude": round(lon0 + rng.normal(0, 0.004), 5),
                "sensor_id": "ACO-07",
                "sensor_type": "acoustic_sensor" if signal in {"gunshot", "vehicle"} else "camera_trap",
                "signal": signal,
                "signal_class": signal_class(signal),
                "confidence": round(float(rng.uniform(0.82, 0.98)), 3),
                "detected_count": int(rng.integers(1, 4)),
                "distance_to_boundary_km": 0.4,
                "distance_to_road_km": 0.6,
                "distance_to_village_km": 2.1,
                "patrol_coverage": 0.08,
                "hours_since_patrol": 71.0,
                "is_night": 1,
                "weather": "clear",
                "temperature_c": 16.4,
                "is_poaching_incident": 1,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _generate_patrols(rng: np.random.Generator, start: datetime, days: int) -> pd.DataFrame:
    rows = []
    for day in range(days):
        date = (start + timedelta(days=day)).date()
        for zone in ZONES:
            rows.append(
                {
                    "date": date,
                    "zone": zone,
                    "rangers_deployed": int(rng.integers(2, 9)),
                    "patrol_hours": round(float(rng.uniform(2, 14)), 1),
                    "km_covered": round(float(rng.uniform(4, 42)), 1),
                    "snares_removed": int(rng.poisson(0.7)),
                }
            )
    return pd.DataFrame(rows)
