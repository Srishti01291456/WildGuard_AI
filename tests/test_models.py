"""Model behaviour tests: training, scoring, anomalies and hotspots."""

from __future__ import annotations

import pandas as pd
import pytest

from wildguard.ml.features import FEATURE_COLUMNS, build_features
from wildguard.ml.hotspots import cluster_hotspots
from wildguard.ml.risk_model import PoachingRiskModel, zone_threat_profile


def test_build_features_columns(sample_events: pd.DataFrame) -> None:
    features = build_features(sample_events)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.notna().all().all()


def test_model_learns_signal(risk_model: PoachingRiskModel) -> None:
    metrics = risk_model.metrics
    assert metrics is not None
    assert metrics.roc_auc > 0.7, "the risk model should clearly beat random"
    assert 0.0 <= metrics.brier <= 0.4
    assert sum(sum(row) for row in metrics.confusion) == metrics.support


def test_importances_are_ranked(risk_model: PoachingRiskModel) -> None:
    importances = risk_model.importances
    assert not importances.empty
    assert importances["importance"].is_monotonic_decreasing
    assert importances["share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_scores_are_bounded_and_banded(scored_events: pd.DataFrame) -> None:
    assert scored_events["risk_score"].between(0, 100).all()
    assert set(scored_events["risk_band"]) <= {"Low", "Medium", "High", "Critical"}


def test_gunshots_score_higher_than_herbivores(scored_events: pd.DataFrame) -> None:
    gunshots = scored_events[scored_events["signal"] == "gunshot"]["risk_score"].mean()
    deer = scored_events[scored_events["signal"] == "deer"]["risk_score"].mean()
    assert gunshots > deer


def test_untrained_model_raises() -> None:
    with pytest.raises(RuntimeError):
        PoachingRiskModel().predict_proba(pd.DataFrame())


def test_training_requires_labels(sample_events: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="is_poaching_incident"):
        PoachingRiskModel().fit(sample_events.drop(columns=["is_poaching_incident"]))


def test_zone_profile(scored_events: pd.DataFrame) -> None:
    profile = zone_threat_profile(scored_events)
    assert not profile.empty
    assert profile["mean_risk"].is_monotonic_decreasing
    assert profile["events"].sum() == len(scored_events)


def test_anomaly_scores(anomaly_profiles: pd.DataFrame) -> None:
    assert not anomaly_profiles.empty
    assert anomaly_profiles["anomaly_score"].between(0, 100).all()
    assert anomaly_profiles["is_anomaly"].sum() >= 1


def test_hotspots(scored_events: pd.DataFrame) -> None:
    hotspots = cluster_hotspots(scored_events, n_clusters=4)
    assert len(hotspots) == 4
    assert hotspots["events"].sum() == len(scored_events)
    assert hotspots["mean_risk"].is_monotonic_decreasing


def test_hotspots_on_empty_frame() -> None:
    assert cluster_hotspots(pd.DataFrame()).empty
