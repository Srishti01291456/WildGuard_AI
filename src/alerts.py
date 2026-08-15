"""Feature 2 - Smart Alerts: rule + statistics based detection engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


@dataclass
class Alert:
    rule: str
    severity: str
    entity: str
    detail: str
    events: int
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    mitre: str


def run_rules(
    enriched: pd.DataFrame,
    *,
    brute_force_threshold: int = 15,
    brute_force_window_min: int = 10,
    exfil_mb: float = 50.0,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """Run every detection rule and return a de-duplicated alert table."""
    if enriched.empty:
        return pd.DataFrame(columns=[f.name for f in Alert.__dataclass_fields__.values()])

    alerts: list[Alert] = []
    alerts += _brute_force(enriched, brute_force_threshold, brute_force_window_min)
    alerts += _malicious_ioc_allowed(enriched)
    alerts += _data_exfiltration(enriched, exfil_mb)
    alerts += _port_scan(enriched)
    alerts += _off_hours_admin(enriched)
    alerts += _volume_anomaly(enriched, zscore_threshold)

    df = pd.DataFrame([asdict(a) for a in alerts])
    if df.empty:
        return df
    df["_order"] = df["severity"].map(SEVERITY_ORDER).fillna(9)
    return df.sort_values(["_order", "events"], ascending=[True, False]).drop(columns="_order").reset_index(drop=True)


def _brute_force(df: pd.DataFrame, threshold: int, window_min: int) -> list[Alert]:
    logins = df[(df["event_type"] == "login") & (df["action"] == "blocked")]
    out: list[Alert] = []
    for ip, grp in logins.groupby("source_ip"):
        grp = grp.sort_values("timestamp")
        counts = grp.set_index("timestamp")["event_id"].rolling(f"{window_min}min").count()
        peak = int(counts.max()) if len(counts) else 0
        if peak >= threshold:
            out.append(
                Alert(
                    rule="Credential brute force",
                    severity="Critical",
                    entity=ip,
                    detail=f"{peak} failed logins in {window_min} min (users: {grp['user'].nunique()})",
                    events=len(grp),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                    mitre="T1110 Brute Force",
                )
            )
    return out


def _malicious_ioc_allowed(df: pd.DataFrame, min_events: int = 3) -> list[Alert]:
    """Known-bad source IPs whose traffic was not blocked (noise-gated)."""
    hits = df[df["is_malicious"] & (df["action"] == "allowed")]
    out: list[Alert] = []
    for ip, grp in hits.groupby("source_ip"):
        if len(grp) < min_events:
            continue
        confidence = int(grp["ioc_confidence"].max())
        out.append(
            Alert(
                rule="Allowed traffic from known-bad IOC",
                severity="Critical" if confidence >= 80 else "High",
                entity=ip,
                detail=f"{grp['threat_type'].mode().iat[0]} indicator, confidence {confidence}%",
                events=len(grp),
                first_seen=grp["timestamp"].min(),
                last_seen=grp["timestamp"].max(),
                mitre="T1071 Application Layer Protocol",
            )
        )
    return out


def _data_exfiltration(df: pd.DataFrame, exfil_mb: float) -> list[Alert]:
    transfers = df[df["event_type"] == "data_transfer"]
    if transfers.empty:
        return []
    totals = transfers.groupby(["source_ip", "user"])["bytes_out"].sum() / 1_000_000
    out: list[Alert] = []
    for (ip, user), mb in totals[totals >= exfil_mb].items():
        grp = transfers[(transfers["source_ip"] == ip) & (transfers["user"] == user)]
        out.append(
            Alert(
                rule="Possible data exfiltration",
                severity="High",
                entity=f"{user}@{ip}",
                detail=f"{mb:,.1f} MB transferred outbound",
                events=len(grp),
                first_seen=grp["timestamp"].min(),
                last_seen=grp["timestamp"].max(),
                mitre="T1048 Exfiltration Over Alternative Protocol",
            )
        )
    return out


def _port_scan(df: pd.DataFrame) -> list[Alert]:
    scans = df[df["event_type"] == "port_scan"]
    out: list[Alert] = []
    for ip, grp in scans.groupby("source_ip"):
        if grp["dest_ip"].nunique() >= 5:
            out.append(
                Alert(
                    rule="Network reconnaissance",
                    severity="Medium",
                    entity=ip,
                    detail=f"scanned {grp['dest_ip'].nunique()} internal hosts",
                    events=len(grp),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                    mitre="T1046 Network Service Discovery",
                )
            )
    return out


def _off_hours_admin(df: pd.DataFrame) -> list[Alert]:
    mask = df["event_type"].isin(["privilege_escalation", "login"]) & (
        df["hour"].isin(list(range(0, 5)) + [23])
    ) & (df["user"].isin(["root", "svc_backup"]))
    hits = df[mask]
    out: list[Alert] = []
    for user, grp in hits.groupby("user"):
        if len(grp) >= 5:
            out.append(
                Alert(
                    rule="Off-hours privileged activity",
                    severity="Medium",
                    entity=str(user),
                    detail=f"{len(grp)} events between 23:00-05:00 UTC",
                    events=len(grp),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                    mitre="T1078 Valid Accounts",
                )
            )
    return out


def _volume_anomaly(df: pd.DataFrame, z: float) -> list[Alert]:
    """Statistical spike detection on hourly event volume (z-score)."""
    hourly = df.set_index("timestamp").resample("1h").size()
    if len(hourly) < 12 or hourly.std(ddof=0) == 0:
        return []
    scores = (hourly - hourly.mean()) / hourly.std(ddof=0)
    spikes = scores[scores >= z].sort_values(ascending=False).head(5)
    out: list[Alert] = []
    for ts, score in spikes.items():
        out.append(
            Alert(
                rule="Event volume anomaly",
                severity="High" if score >= z + 1 else "Medium",
                entity=ts.strftime("%Y-%m-%d %H:00"),
                detail=f"{int(hourly[ts])} events (z={score:.1f} vs {hourly.mean():.0f} avg)",
                events=int(hourly[ts]),
                first_seen=ts,
                last_seen=ts + pd.Timedelta(hours=1),
                mitre="TA0040 Impact",
            )
        )
    return out


def alert_summary(alerts: pd.DataFrame) -> dict[str, int]:
    base = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    if alerts.empty:
        return base
    base.update(alerts["severity"].value_counts().to_dict())
    return base
