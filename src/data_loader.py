"""Data generation and loading for the Threat Intelligence dashboard."""

from __future__ import annotations

import ipaddress
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVENTS_CSV = DATA_DIR / "events.csv"
IOC_CSV = DATA_DIR / "ioc_feed.csv"

EVENT_COLUMNS = [
    "timestamp",
    "event_id",
    "source_ip",
    "dest_ip",
    "user",
    "country",
    "event_type",
    "action",
    "severity",
    "bytes_out",
]

EVENT_TYPES = [
    "login",
    "file_access",
    "dns_query",
    "port_scan",
    "malware_detected",
    "data_transfer",
    "privilege_escalation",
]

COUNTRIES = ["US", "IN", "DE", "BR", "RU", "CN", "NL", "GB", "NG", "KP"]
USERS = ["a.sharma", "j.doe", "m.patel", "svc_backup", "root", "k.lee", "s.gupta"]


def _random_ip(rng: np.random.Generator, private: bool = False) -> str:
    if private:
        return f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    return ".".join(str(rng.integers(1, 254)) for _ in range(4))


def generate_ioc_feed(rng: np.random.Generator, n: int = 60) -> pd.DataFrame:
    """Simulated threat-intel feed (stand-in for AbuseIPDB / OTX / MISP)."""
    rows = []
    for _ in range(n):
        rows.append(
            {
                "indicator": _random_ip(rng),
                "type": "ipv4",
                "threat_type": rng.choice(
                    ["botnet", "scanner", "c2", "phishing", "ransomware", "tor-exit"]
                ),
                "confidence": int(rng.integers(50, 100)),
                "source": rng.choice(["AbuseIPDB", "AlienVault OTX", "MISP", "Internal"]),
                "last_seen": (
                    datetime.utcnow() - timedelta(days=int(rng.integers(0, 30)))
                ).strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset="indicator").reset_index(drop=True)


def generate_events(
    rng: np.random.Generator, ioc: pd.DataFrame, n: int = 4000, days: int = 14
) -> pd.DataFrame:
    """Synthetic SIEM-style event log with planted attack patterns."""
    start = datetime.utcnow() - timedelta(days=days)
    malicious_ips = ioc["indicator"].tolist()

    rows = []
    for i in range(n):
        ts = start + timedelta(seconds=int(rng.integers(0, days * 24 * 3600)))
        # 12% of traffic comes from known-bad indicators
        if rng.random() < 0.12:
            src = str(rng.choice(malicious_ips))
        else:
            src = _random_ip(rng)
        event_type = str(rng.choice(EVENT_TYPES, p=[0.35, 0.2, 0.15, 0.1, 0.05, 0.1, 0.05]))
        action = "blocked" if rng.random() < 0.25 else "allowed"
        severity = int(np.clip(rng.normal(3, 1.4), 1, 5).round())
        rows.append(
            {
                "timestamp": ts,
                "event_id": f"EVT-{i:06d}",
                "source_ip": src,
                "dest_ip": _random_ip(rng, private=True),
                "user": str(rng.choice(USERS)),
                "country": str(rng.choice(COUNTRIES)),
                "event_type": event_type,
                "action": action,
                "severity": severity,
                "bytes_out": int(abs(rng.normal(50_000, 120_000))),
            }
        )

    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df = pd.concat([df, _brute_force_burst(rng, malicious_ips), _exfil_burst(rng)], ignore_index=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _brute_force_burst(rng: np.random.Generator, malicious_ips: list[str]) -> pd.DataFrame:
    """Plant a credential brute-force so the alert engine has a true positive."""
    attacker = str(rng.choice(malicious_ips))
    base = datetime.utcnow() - timedelta(hours=6)
    rows = []
    for i in range(45):
        rows.append(
            {
                "timestamp": base + timedelta(seconds=i * 20),
                "event_id": f"EVT-BF-{i:04d}",
                "source_ip": attacker,
                "dest_ip": "10.0.0.15",
                "user": "root",
                "country": "RU",
                "event_type": "login",
                "action": "blocked",
                "severity": 4,
                "bytes_out": int(rng.integers(200, 2000)),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _exfil_burst(rng: np.random.Generator) -> pd.DataFrame:
    """Plant a large data-transfer spike (exfiltration)."""
    base = datetime.utcnow() - timedelta(hours=2)
    rows = []
    for i in range(12):
        rows.append(
            {
                "timestamp": base + timedelta(minutes=i * 3),
                "event_id": f"EVT-EX-{i:04d}",
                "source_ip": "10.0.4.77",
                "dest_ip": "185.220.101.9",
                "user": "svc_backup",
                "country": "NL",
                "event_type": "data_transfer",
                "action": "allowed",
                "severity": 4,
                "bytes_out": int(rng.integers(8_000_000, 20_000_000)),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def bootstrap_sample_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the CSV sample dataset if it does not exist yet, then load it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENTS_CSV.exists() or not IOC_CSV.exists():
        rng = np.random.default_rng(seed)
        ioc = generate_ioc_feed(rng)
        events = generate_events(rng, ioc)
        ioc.to_csv(IOC_CSV, index=False)
        events.to_csv(EVENTS_CSV, index=False)
    return load_events(EVENTS_CSV), load_ioc(IOC_CSV)


def load_events(source) -> pd.DataFrame:
    """Load and validate an event log from a path or file-like object."""
    df = pd.read_csv(source)
    missing = [c for c in ("timestamp", "source_ip", "event_type") if c not in df.columns]
    if missing:
        raise ValueError(f"Event log is missing required column(s): {', '.join(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for col, default in (
        ("dest_ip", "-"),
        ("user", "unknown"),
        ("country", "??"),
        ("action", "allowed"),
    ):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)

    df["severity"] = pd.to_numeric(df.get("severity", 3), errors="coerce").fillna(3).clip(1, 5).astype(int)
    df["bytes_out"] = pd.to_numeric(df.get("bytes_out", 0), errors="coerce").fillna(0).astype(int)
    if "event_id" not in df.columns:
        df["event_id"] = [f"EVT-{i:06d}" for i in range(len(df))]

    df["source_ip"] = df["source_ip"].astype(str).str.strip()
    df["is_valid_ip"] = df["source_ip"].map(_is_ip)
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    return df.sort_values("timestamp").reset_index(drop=True)


def load_ioc(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    if "indicator" not in df.columns:
        raise ValueError("IOC feed must contain an 'indicator' column")
    df["indicator"] = df["indicator"].astype(str).str.strip()
    if "confidence" not in df.columns:
        df["confidence"] = 75
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(75).clip(0, 100).astype(int)
    for col, default in (("threat_type", "unknown"), ("source", "unknown"), ("type", "ipv4")):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)
    return df.drop_duplicates(subset="indicator").reset_index(drop=True)


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False
