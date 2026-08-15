"""Dataset generation and ingestion tests."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from wildguard.data.generator import generate_dataset
from wildguard.data.loader import DataValidationError, load_events


def test_generate_dataset_shape(sample_events: pd.DataFrame) -> None:
    assert len(sample_events) > 500
    assert sample_events["is_poaching_incident"].between(0, 1).all()
    # the label must be learnable but not degenerate
    assert 0.02 < sample_events["is_poaching_incident"].mean() < 0.6


def test_generated_signals_are_categorised() -> None:
    events, patrols = generate_dataset(n_events=400, days=20, seed=7)
    assert set(events["signal_class"]) <= {"wildlife", "human", "threat"}
    assert not patrols.empty
    assert events["timestamp"].is_monotonic_increasing


def test_load_events_fills_defaults() -> None:
    csv = io.BytesIO(
        b"timestamp,zone,signal\n"
        b"2024-05-01 22:15:00,Core Sanctuary,gunshot\n"
        b"2024-05-02 03:10:00,Riverine Belt,elephant\n"
    )
    df = load_events(csv)
    assert len(df) == 2
    assert df["confidence"].notna().all()
    assert df["signal_class"].tolist() == ["threat", "wildlife"]
    assert df["is_night"].tolist() == [1, 1]
    assert df["latitude"].notna().all()  # filled from the zone centroid


def test_load_events_rejects_missing_columns() -> None:
    with pytest.raises(DataValidationError, match="missing required column"):
        load_events(io.BytesIO(b"timestamp,zone\n2024-05-01,Core Sanctuary\n"))


def test_load_events_drops_unparseable_timestamps() -> None:
    csv = io.BytesIO(b"timestamp,zone,signal\nnot-a-date,Core Sanctuary,rhino\n2024-05-01,Core Sanctuary,rhino\n")
    assert len(load_events(csv)) == 1
