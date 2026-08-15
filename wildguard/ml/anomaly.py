"""Unsupervised anomaly detection over zone/day activity profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wildguard.config import RANDOM_SEED

PROFILE_FEATURES = [
    "events",
    "threat_ratio",
    "night_ratio",
    "mean_confidence",
    "distinct_signals",
    "mean_patrol_coverage",
    "human_events",
]


def build_profiles(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events into one behavioural profile per zone and day."""
    if events.empty:
        return pd.DataFrame(columns=["zone", "date", *PROFILE_FEATURES])

    df = events.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    grouped = df.groupby(["zone", "date"])
    profiles = pd.DataFrame(
        {
            "events": grouped.size(),
            "threat_ratio": grouped["signal_class"].apply(lambda s: float((s == "threat").mean())),
            "night_ratio": grouped["is_night"].mean(),
            "mean_confidence": grouped["confidence"].mean(),
            "distinct_signals": grouped["signal"].nunique(),
            "mean_patrol_coverage": grouped["patrol_coverage"].mean(),
            "human_events": grouped["signal_class"].apply(lambda s: int((s != "wildlife").sum())),
        }
    ).reset_index()
    return profiles.round(4)


class ZoneAnomalyDetector:
    """Isolation Forest over zone/day profiles, exposed as a 0-100 score."""

    def __init__(self, contamination: float = 0.06, seed: int = RANDOM_SEED) -> None:
        self.contamination = contamination
        self.seed = seed
        self.pipeline: Pipeline | None = None

    def fit(self, events: pd.DataFrame) -> ZoneAnomalyDetector:
        profiles = build_profiles(events)
        if len(profiles) < 10:
            self.pipeline = None
            return self
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "forest",
                    IsolationForest(
                        n_estimators=250,
                        contamination=self.contamination,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(profiles[PROFILE_FEATURES])
        return self

    def score(self, events: pd.DataFrame) -> pd.DataFrame:
        """Score profiles; higher ``anomaly_score`` means more unusual."""
        profiles = build_profiles(events)
        if self.pipeline is None or profiles.empty:
            profiles["anomaly_score"] = 0.0
            profiles["is_anomaly"] = False
            return profiles

        x = profiles[PROFILE_FEATURES]
        raw = -self.pipeline.decision_function(x)  # higher = more anomalous
        span = raw.max() - raw.min()
        profiles["anomaly_score"] = np.round(100 * (raw - raw.min()) / span, 1) if span else 0.0
        profiles["is_anomaly"] = self.pipeline.predict(x) == -1
        return profiles.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
