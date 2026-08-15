"""Feature 2 - Smart Alert System.

Fuses three signals into one ranked, de-duplicated alert queue:

* the supervised risk model (``risk_score`` per detection),
* the unsupervised zone anomaly detector,
* domain rules written with rangers in mind (gunshots, snares, patrol gaps).

Every alert carries a severity, the evidence behind it and a recommended
response action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from wildguard.config import ENDANGERED_SPECIES

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
ALERT_COLUMNS = [
    "rule",
    "severity",
    "zone",
    "entity",
    "detail",
    "recommended_action",
    "events",
    "risk_score",
    "first_seen",
    "last_seen",
]


@dataclass
class AlertThresholds:
    """Live-tunable detection thresholds (wired to the sidebar)."""

    risk_threshold: float = 60.0
    min_cluster_events: int = 3
    gunshot_confidence: float = 0.75
    patrol_gap_hours: float = 48.0
    anomaly_score: float = 70.0
    silent_sensor_hours: float = 72.0
    recent_window_hours: int = 48


@dataclass
class Alert:
    rule: str
    severity: str
    zone: str
    entity: str
    detail: str
    recommended_action: str
    events: int
    risk_score: float
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp


@dataclass
class AlertEngine:
    """Runs every detector and returns a ranked alert table."""

    thresholds: AlertThresholds = field(default_factory=AlertThresholds)

    def run(self, scored: pd.DataFrame, anomalies: pd.DataFrame | None = None) -> pd.DataFrame:
        if scored.empty:
            return pd.DataFrame(columns=ALERT_COLUMNS)

        alerts: list[Alert] = []
        alerts += self._gunshot(scored)
        alerts += self._high_risk_cluster(scored)
        alerts += self._night_intrusion(scored)
        alerts += self._snare_and_logging(scored)
        alerts += self._endangered_exposure(scored)
        alerts += self._patrol_gap(scored)
        alerts += self._silent_sensor(scored)
        if anomalies is not None:
            alerts += self._zone_anomaly(anomalies)

        if not alerts:
            return pd.DataFrame(columns=ALERT_COLUMNS)

        df = pd.DataFrame([asdict(a) for a in alerts])
        df = df.drop_duplicates(subset=["rule", "entity"], keep="first")
        df["_order"] = df["severity"].map(SEVERITY_ORDER).fillna(9)
        df = df.sort_values(["_order", "risk_score", "events"], ascending=[True, False, False])
        return df.drop(columns="_order").reset_index(drop=True)[ALERT_COLUMNS]

    # -- individual detectors -------------------------------------------------

    def _recent(self, scored: pd.DataFrame) -> pd.DataFrame:
        cutoff = scored["timestamp"].max() - pd.Timedelta(hours=self.thresholds.recent_window_hours)
        return scored[scored["timestamp"] >= cutoff]

    def _gunshot(self, scored: pd.DataFrame) -> list[Alert]:
        hits = scored[
            (scored["signal"] == "gunshot") & (scored["confidence"] >= self.thresholds.gunshot_confidence)
        ]
        out = []
        for zone, grp in hits.groupby("zone"):
            out.append(
                Alert(
                    rule="Gunshot detected",
                    severity="Critical",
                    zone=str(zone),
                    entity=f"{zone} / {grp['sensor_id'].mode().iat[0]}",
                    detail=(
                        f"{len(grp)} acoustic gunshot detections, "
                        f"mean confidence {grp['confidence'].mean():.0%}"
                    ),
                    recommended_action="Dispatch armed rapid-response unit and launch drone overwatch",
                    events=len(grp),
                    risk_score=round(float(grp["risk_score"].max()), 1),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                )
            )
        return out

    def _high_risk_cluster(self, scored: pd.DataFrame) -> list[Alert]:
        """Zones where the model flags several high-risk detections in the window."""
        recent = self._recent(scored)
        hot = recent[recent["risk_score"] >= self.thresholds.risk_threshold]
        out = []
        for zone, grp in hot.groupby("zone"):
            if len(grp) < self.thresholds.min_cluster_events:
                continue
            out.append(
                Alert(
                    rule="ML high-risk detection cluster",
                    severity="Critical" if grp["risk_score"].mean() >= 75 else "High",
                    zone=str(zone),
                    entity=str(zone),
                    detail=(
                        f"{len(grp)} detections scored ≥ {self.thresholds.risk_threshold:.0f} "
                        f"(mean {grp['risk_score'].mean():.0f}) in the last "
                        f"{self.thresholds.recent_window_hours}h"
                    ),
                    recommended_action="Re-task patrol route to this zone within the next shift",
                    events=len(grp),
                    risk_score=round(float(grp["risk_score"].mean()), 1),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                )
            )
        return out

    def _night_intrusion(self, scored: pd.DataFrame) -> list[Alert]:
        recent = self._recent(scored)
        mask = (
            (recent["is_night"] == 1)
            & recent["signal"].isin(["human_presence", "vehicle", "campfire"])
            & (recent["distance_to_boundary_km"] <= 3.0)
        )
        hits = recent[mask]
        out = []
        for zone, grp in hits.groupby("zone"):
            if len(grp) < self.thresholds.min_cluster_events:
                continue
            out.append(
                Alert(
                    rule="Night-time boundary intrusion",
                    severity="High",
                    zone=str(zone),
                    entity=str(zone),
                    detail=(
                        f"{len(grp)} night detections of {', '.join(sorted(grp['signal'].unique()))} "
                        f"within {grp['distance_to_boundary_km'].min():.1f} km of the boundary"
                    ),
                    recommended_action="Deploy night patrol to the boundary track and verify camera feeds",
                    events=len(grp),
                    risk_score=round(float(grp["risk_score"].mean()), 1),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                )
            )
        return out

    def _snare_and_logging(self, scored: pd.DataFrame) -> list[Alert]:
        hits = self._recent(scored)
        hits = hits[hits["signal"].isin(["snare_trap", "chainsaw"])]
        out = []
        for (zone, signal), grp in hits.groupby(["zone", "signal"]):
            label = "Snare line suspected" if signal == "snare_trap" else "Illegal logging activity"
            action = (
                "Send a snare-sweep team along the detection corridor"
                if signal == "snare_trap"
                else "Notify forest-crime unit and record GPS track of the activity"
            )
            out.append(
                Alert(
                    rule=label,
                    severity="High" if len(grp) >= 3 else "Medium",
                    zone=str(zone),
                    entity=f"{zone} / {signal}",
                    detail=(
                        f"{len(grp)} {signal.replace('_', ' ')} detections, "
                        f"mean risk {grp['risk_score'].mean():.0f}"
                    ),
                    recommended_action=action,
                    events=len(grp),
                    risk_score=round(float(grp["risk_score"].mean()), 1),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                )
            )
        return out

    def _endangered_exposure(self, scored: pd.DataFrame) -> list[Alert]:
        """Endangered species seen in zones the model considers dangerous."""
        recent = self._recent(scored)
        species = recent[recent["signal"].isin(ENDANGERED_SPECIES)]
        risky_zones = (
            recent.groupby("zone")["risk_score"].mean().pipe(lambda s: s[s >= self.thresholds.risk_threshold * 0.7])
        )
        out = []
        for zone, grp in species.groupby("zone"):
            if zone not in risky_zones.index:
                continue
            out.append(
                Alert(
                    rule="Endangered species exposed",
                    severity="High",
                    zone=str(zone),
                    entity=f"{zone} / {', '.join(sorted(grp['signal'].unique()))}",
                    detail=(
                        f"{len(grp)} sightings of protected species in a zone with mean risk "
                        f"{risky_zones[zone]:.0f}"
                    ),
                    recommended_action="Assign a dedicated monitoring team to shadow the herd",
                    events=len(grp),
                    risk_score=round(float(risky_zones[zone]), 1),
                    first_seen=grp["timestamp"].min(),
                    last_seen=grp["timestamp"].max(),
                )
            )
        return out

    def _patrol_gap(self, scored: pd.DataFrame) -> list[Alert]:
        recent = self._recent(scored)
        gaps = recent.groupby("zone").agg(
            hours=("hours_since_patrol", "max"),
            coverage=("patrol_coverage", "mean"),
            risk=("risk_score", "mean"),
            events=("event_id", "count"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        gaps = gaps[(gaps["hours"] >= self.thresholds.patrol_gap_hours) & (gaps["risk"] >= 35)]
        return [
            Alert(
                rule="Patrol coverage gap",
                severity="Medium",
                zone=str(zone),
                entity=str(zone),
                detail=f"{row.hours:.0f}h since last patrol, coverage {row.coverage:.0%}, mean risk {row.risk:.0f}",
                recommended_action="Schedule a patrol sweep and reposition a camera trap",
                events=int(row.events),
                risk_score=round(float(row.risk), 1),
                first_seen=row.first_seen,
                last_seen=row.last_seen,
            )
            for zone, row in gaps.iterrows()
        ]

    def _silent_sensor(self, scored: pd.DataFrame) -> list[Alert]:
        """Sensors that stopped reporting - tampering or battery failure."""
        latest = scored["timestamp"].max()
        last_seen = scored.groupby(["sensor_id", "zone"])["timestamp"].max()
        silent = last_seen[(latest - last_seen) > pd.Timedelta(hours=self.thresholds.silent_sensor_hours)]
        return [
            Alert(
                rule="Sensor offline",
                severity="Low",
                zone=str(zone),
                entity=str(sensor),
                detail=f"no data for {(latest - ts).total_seconds() / 3600:.0f}h",
                recommended_action="Check battery/SD card on the next patrol, inspect for tampering",
                events=0,
                risk_score=0.0,
                first_seen=ts,
                last_seen=ts,
            )
            for (sensor, zone), ts in silent.items()
        ]

    def _zone_anomaly(self, anomalies: pd.DataFrame) -> list[Alert]:
        if anomalies.empty or "anomaly_score" not in anomalies.columns:
            return []
        hits = anomalies[anomalies["anomaly_score"] >= self.thresholds.anomaly_score].head(6)
        out = []
        for _, row in hits.iterrows():
            ts = pd.Timestamp(row["date"])
            out.append(
                Alert(
                    rule="Behavioural anomaly (Isolation Forest)",
                    severity="High" if row["anomaly_score"] >= 85 else "Medium",
                    zone=str(row["zone"]),
                    entity=f"{row['zone']} / {row['date']}",
                    detail=(
                        f"anomaly score {row['anomaly_score']:.0f} - {int(row['events'])} events, "
                        f"{row['threat_ratio']:.0%} threat signals, {row['night_ratio']:.0%} at night"
                    ),
                    recommended_action="Review the day's footage and compare with the zone baseline",
                    events=int(row["events"]),
                    risk_score=round(float(row["anomaly_score"]), 1),
                    first_seen=ts,
                    last_seen=ts,
                )
            )
        return out


def alert_summary(alerts: pd.DataFrame) -> dict[str, int]:
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    if not alerts.empty:
        counts.update(alerts["severity"].value_counts().to_dict())
    return counts
