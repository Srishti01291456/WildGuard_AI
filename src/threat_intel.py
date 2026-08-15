"""Feature 1 - Threat Intelligence: IOC correlation and risk scoring."""

from __future__ import annotations

import ipaddress

import pandas as pd

RISK_BANDS = [(80, "Critical"), (60, "High"), (35, "Medium"), (0, "Low")]

# Weights used by the transparent risk model (documented in the UI).
W_IOC = 45
W_SEVERITY = 20
W_BLOCKED = 10
W_VOLUME = 15
W_SPREAD = 10


def enrich_with_ioc(events: pd.DataFrame, ioc: pd.DataFrame) -> pd.DataFrame:
    """Left-join events against the IOC feed on source IP."""
    cols = ["indicator", "threat_type", "confidence", "source"]
    feed = ioc[cols].rename(
        columns={"indicator": "source_ip", "source": "intel_source", "confidence": "ioc_confidence"}
    )
    enriched = events.merge(feed, on="source_ip", how="left")
    enriched["is_malicious"] = enriched["ioc_confidence"].notna()
    enriched["threat_type"] = enriched["threat_type"].fillna("none")
    enriched["intel_source"] = enriched["intel_source"].fillna("-")
    enriched["ioc_confidence"] = enriched["ioc_confidence"].fillna(0).astype(int)
    enriched["is_private_ip"] = enriched["source_ip"].map(_is_private)
    return enriched


def score_sources(enriched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per source IP and compute a 0-100 risk score."""
    if enriched.empty:
        return pd.DataFrame(
            columns=[
                "source_ip", "events", "avg_severity", "max_severity", "blocked_ratio",
                "bytes_out", "targets", "threat_type", "intel_source", "ioc_confidence",
                "risk_score", "risk_band", "last_seen",
            ]
        )

    grp = enriched.groupby("source_ip")
    agg = pd.DataFrame(
        {
            "events": grp.size(),
            "avg_severity": grp["severity"].mean().round(2),
            "max_severity": grp["severity"].max(),
            "blocked_ratio": grp["action"].apply(lambda s: (s == "blocked").mean()).round(2),
            "bytes_out": grp["bytes_out"].sum(),
            "targets": grp["dest_ip"].nunique(),
            "ioc_confidence": grp["ioc_confidence"].max(),
            "threat_type": grp["threat_type"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "none"),
            "intel_source": grp["intel_source"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "-"),
            "last_seen": grp["timestamp"].max(),
        }
    ).reset_index()

    ioc_part = W_IOC * (agg["ioc_confidence"] / 100)
    sev_part = W_SEVERITY * ((agg["avg_severity"] - 1) / 4)
    blocked_part = W_BLOCKED * agg["blocked_ratio"]
    volume_part = W_VOLUME * _normalise(agg["events"])
    spread_part = W_SPREAD * _normalise(agg["targets"])

    agg["risk_score"] = (ioc_part + sev_part + blocked_part + volume_part + spread_part).round(1)
    agg["risk_band"] = agg["risk_score"].map(risk_band)
    return agg.sort_values("risk_score", ascending=False).reset_index(drop=True)


def risk_band(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "Low"


def score_breakdown(row: pd.Series) -> dict[str, float]:
    """Per-component contribution, used to explain a score in the UI."""
    return {
        "IOC match": round(W_IOC * (row["ioc_confidence"] / 100), 1),
        "Severity": round(W_SEVERITY * ((row["avg_severity"] - 1) / 4), 1),
        "Blocked ratio": round(W_BLOCKED * row["blocked_ratio"], 1),
        "Event volume": round(row["risk_score"] - sum(
            [
                W_IOC * (row["ioc_confidence"] / 100),
                W_SEVERITY * ((row["avg_severity"] - 1) / 4),
                W_BLOCKED * row["blocked_ratio"],
            ]
        ), 1),
    }


def _normalise(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / span


def _is_private(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False
