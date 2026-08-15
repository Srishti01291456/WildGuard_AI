import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import alerts, analytics, data_loader, threat_intel


@pytest.fixture(scope="module")
def dataset():
    events, ioc = data_loader.bootstrap_sample_data()
    return threat_intel.enrich_with_ioc(events, ioc), ioc


def test_sample_data_loads(dataset):
    enriched, ioc = dataset
    assert len(enriched) > 1000
    assert not ioc.empty
    assert enriched["timestamp"].is_monotonic_increasing


def test_enrichment_flags_known_bad(dataset):
    enriched, _ = dataset
    assert enriched["is_malicious"].sum() > 0
    assert enriched.loc[enriched["is_malicious"], "ioc_confidence"].min() > 0


def test_risk_scores_are_bounded(dataset):
    enriched, _ = dataset
    scores = threat_intel.score_sources(enriched)
    assert scores["risk_score"].between(0, 100).all()
    assert scores["risk_score"].is_monotonic_decreasing
    assert set(scores["risk_band"]) <= {"Critical", "High", "Medium", "Low"}


def test_brute_force_is_detected(dataset):
    enriched, _ = dataset
    found = alerts.run_rules(enriched)
    assert "Credential brute force" in set(found["rule"])
    assert "Possible data exfiltration" in set(found["rule"])


def test_alert_thresholds_are_respected(dataset):
    enriched, _ = dataset
    strict = alerts.run_rules(enriched, brute_force_threshold=50, exfil_mb=500)
    lenient = alerts.run_rules(enriched, brute_force_threshold=5, exfil_mb=10)
    assert len(lenient) >= len(strict)


def test_empty_input_is_handled():
    empty = pd.DataFrame(columns=["timestamp", "source_ip", "event_type", "severity", "bytes_out"])
    assert alerts.run_rules(empty).empty
    assert threat_intel.score_sources(empty).empty
    analytics.timeline(empty)  # must not raise


def test_load_events_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing required column"):
        data_loader.load_events(bad)


def test_load_events_coerces_types(tmp_path):
    csv = tmp_path / "events.csv"
    now = datetime.utcnow()
    rows = [
        f"{(now - timedelta(minutes=i)).isoformat()},1.2.3.4,login,,,\n" for i in range(3)
    ]
    csv.write_text("timestamp,source_ip,event_type,severity,bytes_out,user\n" + "".join(rows))
    df = data_loader.load_events(csv)
    assert len(df) == 3
    assert df["severity"].eq(3).all()
    assert df["bytes_out"].eq(0).all()
    assert df["user"].eq("unknown").all()
