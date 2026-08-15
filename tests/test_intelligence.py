"""Alert engine and analytics tests."""

from __future__ import annotations

import pandas as pd

from wildguard.intelligence.alerts import ALERT_COLUMNS, AlertEngine, AlertThresholds, alert_summary
from wildguard.intelligence.analytics import (
    activity_timeline,
    kpis,
    sensor_health,
    shannon_diversity,
    zone_biodiversity,
)


def test_alert_engine_produces_ranked_alerts(scored_events: pd.DataFrame, anomaly_profiles: pd.DataFrame) -> None:
    alerts = AlertEngine().run(scored_events, anomaly_profiles)
    assert not alerts.empty
    assert list(alerts.columns) == ALERT_COLUMNS
    order = alerts["severity"].map({"Critical": 0, "High": 1, "Medium": 2, "Low": 3})
    assert order.is_monotonic_increasing
    assert alerts["recommended_action"].str.len().gt(10).all()


def test_alerts_deduplicate(scored_events: pd.DataFrame) -> None:
    alerts = AlertEngine().run(scored_events)
    assert not alerts.duplicated(subset=["rule", "entity"]).any()


def test_thresholds_change_alert_volume(scored_events: pd.DataFrame) -> None:
    strict = AlertEngine(AlertThresholds(risk_threshold=95, min_cluster_events=9)).run(scored_events)
    loose = AlertEngine(AlertThresholds(risk_threshold=25, min_cluster_events=1)).run(scored_events)
    assert len(loose) >= len(strict)


def test_alert_engine_on_empty_input() -> None:
    alerts = AlertEngine().run(pd.DataFrame())
    assert alerts.empty
    assert alert_summary(alerts) == {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}


def test_kpis(scored_events: pd.DataFrame) -> None:
    alerts = AlertEngine().run(scored_events)
    values = kpis(scored_events, alerts)
    assert values["Detections"] == f"{len(scored_events):,}"
    assert values["Species observed"] != "0"


def test_shannon_diversity_bounds() -> None:
    assert shannon_diversity(pd.Series([10, 0, 0])) == 0.0
    assert shannon_diversity(pd.Series([5, 5])) > shannon_diversity(pd.Series([9, 1]))


def test_biodiversity_and_sensor_health(scored_events: pd.DataFrame) -> None:
    diversity = zone_biodiversity(scored_events)
    assert not diversity.empty
    assert diversity["diversity_index"].ge(0).all()

    health = sensor_health(scored_events)
    assert set(health["status"]) <= {"Healthy", "Degraded", "Offline"}
    assert health["detections"].sum() == len(scored_events)


def test_figures_render(scored_events: pd.DataFrame) -> None:
    figure = activity_timeline(scored_events)
    assert figure.data
    assert activity_timeline(scored_events.head(0)).layout.annotations
