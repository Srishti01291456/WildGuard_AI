"""Shared fixtures - a small dataset and models trained once per session."""

from __future__ import annotations

import pandas as pd
import pytest

from wildguard.data.generator import generate_dataset
from wildguard.ml.anomaly import ZoneAnomalyDetector
from wildguard.ml.risk_model import PoachingRiskModel


@pytest.fixture(scope="session")
def sample_events() -> pd.DataFrame:
    events, _ = generate_dataset(n_events=1500, days=30, seed=11)
    events["hour"] = events["timestamp"].dt.hour
    return events


@pytest.fixture(scope="session")
def risk_model(sample_events: pd.DataFrame) -> PoachingRiskModel:
    return PoachingRiskModel(seed=11).fit(sample_events)


@pytest.fixture(scope="session")
def scored_events(sample_events: pd.DataFrame, risk_model: PoachingRiskModel) -> pd.DataFrame:
    return risk_model.score_events(sample_events)


@pytest.fixture(scope="session")
def anomaly_profiles(sample_events: pd.DataFrame) -> pd.DataFrame:
    return ZoneAnomalyDetector(seed=11).fit(sample_events).score(sample_events)
