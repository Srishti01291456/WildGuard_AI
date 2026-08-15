"""Tiny model registry: train once, persist to ``models/`` and reuse."""

from __future__ import annotations

import hashlib

import joblib
import pandas as pd

from wildguard.config import ANOMALY_MODEL_PATH, MODEL_DIR, RANDOM_SEED, RISK_MODEL_PATH
from wildguard.ml.anomaly import ZoneAnomalyDetector
from wildguard.ml.features import TARGET
from wildguard.ml.risk_model import PoachingRiskModel


def dataset_fingerprint(events: pd.DataFrame) -> str:
    """Stable hash of the dataset so stale artefacts are retrained."""
    head = pd.util.hash_pandas_object(events.head(500), index=False).values.tobytes()
    payload = f"{len(events)}|{sorted(events.columns)}".encode() + head
    return hashlib.sha256(payload).hexdigest()[:16]


def train(events: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[PoachingRiskModel, ZoneAnomalyDetector]:
    risk = PoachingRiskModel(seed=seed).fit(events)
    anomaly = ZoneAnomalyDetector(seed=seed).fit(events)
    return risk, anomaly


def save(risk: PoachingRiskModel, anomaly: ZoneAnomalyDetector, fingerprint: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"fingerprint": fingerprint, "model": risk}, RISK_MODEL_PATH)
    joblib.dump({"fingerprint": fingerprint, "model": anomaly}, ANOMALY_MODEL_PATH)


def load_or_train(
    events: pd.DataFrame,
    *,
    force_retrain: bool = False,
    seed: int = RANDOM_SEED,
) -> tuple[PoachingRiskModel, ZoneAnomalyDetector, bool]:
    """Return ``(risk_model, anomaly_model, was_trained)``.

    Cached artefacts are reused only when they were fitted on the same data.
    Unlabelled datasets fall back to the cached model, since the supervised
    model cannot be refitted without a target column.
    """
    fingerprint = dataset_fingerprint(events)
    cached = _load_cached(fingerprint) if not force_retrain else None
    if cached is not None:
        return cached[0], cached[1], False

    if TARGET not in events.columns:
        fallback = _load_cached(fingerprint=None)
        if fallback is not None:
            return fallback[0], fallback[1], False
        raise ValueError(
            "No labelled data and no trained model available. "
            "Run `python scripts/train_models.py` on the sample dataset first."
        )

    risk, anomaly = train(events, seed=seed)
    save(risk, anomaly, fingerprint)
    return risk, anomaly, True


def _load_cached(fingerprint: str | None) -> tuple[PoachingRiskModel, ZoneAnomalyDetector] | None:
    if not (RISK_MODEL_PATH.exists() and ANOMALY_MODEL_PATH.exists()):
        return None
    try:
        risk_blob = joblib.load(RISK_MODEL_PATH)
        anomaly_blob = joblib.load(ANOMALY_MODEL_PATH)
    except Exception:
        return None
    if fingerprint is not None and risk_blob.get("fingerprint") != fingerprint:
        return None
    return risk_blob["model"], anomaly_blob["model"]
