"""Cached data/model loading and the shared sidebar controls."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from wildguard.data.loader import DataValidationError, bootstrap_dataset, load_events
from wildguard.intelligence.alerts import AlertThresholds
from wildguard.ml.anomaly import ZoneAnomalyDetector
from wildguard.ml.registry import dataset_fingerprint, load_or_train
from wildguard.ml.risk_model import PoachingRiskModel

UPLOAD_KEY = "uploaded_events_bytes"


@dataclass
class Context:
    """Everything a page needs: data, models and model output."""

    events: pd.DataFrame
    patrols: pd.DataFrame
    scored: pd.DataFrame
    anomalies: pd.DataFrame
    risk_model: PoachingRiskModel
    anomaly_model: ZoneAnomalyDetector
    trained_now: bool
    source_label: str


@st.cache_data(show_spinner="Loading field telemetry…")
def _load_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    return bootstrap_dataset()


@st.cache_data(show_spinner="Parsing uploaded event log…")
def _load_uploaded(payload: bytes) -> pd.DataFrame:
    return load_events(io.BytesIO(payload))


@st.cache_resource(show_spinner="Training models (first run only)…")
def _get_models(fingerprint: str, _events: pd.DataFrame, force: bool = False):
    return load_or_train(_events, force_retrain=force)


@st.cache_data(show_spinner="Scoring detections…")
def _score(fingerprint: str, events: pd.DataFrame, _model: PoachingRiskModel) -> pd.DataFrame:
    return _model.score_events(events)


@st.cache_data(show_spinner=False)
def _anomalies(fingerprint: str, events: pd.DataFrame, _model: ZoneAnomalyDetector) -> pd.DataFrame:
    return _model.score(events)


def get_context(force_retrain: bool = False) -> Context:
    """Load data, (re)train models if needed and score every detection."""
    force_retrain = force_retrain or bool(st.session_state.pop("force_retrain", False))
    sample_events, patrols = _load_sample()
    payload = st.session_state.get(UPLOAD_KEY)

    events, source_label = sample_events, "Sample reserve dataset"
    if payload:
        try:
            events = _load_uploaded(payload)
            source_label = "Uploaded event log"
        except DataValidationError as exc:
            st.sidebar.error(str(exc))

    fingerprint = dataset_fingerprint(events)
    if force_retrain:
        _get_models.clear()
    risk_model, anomaly_model, trained_now = _get_models(fingerprint, events, force_retrain)

    scored = _score(fingerprint, events, risk_model)
    anomalies = _anomalies(fingerprint, events, anomaly_model)
    return Context(
        events=events,
        patrols=patrols,
        scored=scored,
        anomalies=anomalies,
        risk_model=risk_model,
        anomaly_model=anomaly_model,
        trained_now=trained_now,
        source_label=source_label,
    )


def sidebar_data_source() -> None:
    """Upload / reset controls, rendered on every page."""
    with st.sidebar:
        st.markdown("### 📥 Data source")
        upload = st.file_uploader("Event log (CSV)", type="csv", key="event_upload")
        if upload is not None:
            st.session_state[UPLOAD_KEY] = upload.getvalue()
        if st.session_state.get(UPLOAD_KEY) and st.button("Reset to sample data", use_container_width=True):
            st.session_state.pop(UPLOAD_KEY, None)
            st.rerun()
        if st.button("🔁 Retrain models", use_container_width=True):
            st.session_state["force_retrain"] = True
            st.rerun()


def sidebar_filters(scored: pd.DataFrame) -> pd.DataFrame:
    """Shared filters; selections persist across pages via session state."""
    if scored.empty:
        return scored

    with st.sidebar:
        st.markdown("### 🔎 Filters")
        min_date: date = scored["timestamp"].min().date()
        max_date: date = scored["timestamp"].max().date()
        date_range = st.date_input(
            "Date range",
            value=st.session_state.get("filter_dates", (min_date, max_date)),
            min_value=min_date,
            max_value=max_date,
            key="filter_dates",
        )
        zones = st.multiselect(
            "Zones", sorted(scored["zone"].unique()),
            default=st.session_state.get("filter_zones", []), key="filter_zones",
            placeholder="All zones",
        )
        classes = st.multiselect(
            "Signal category", sorted(scored["signal_class"].unique()),
            default=st.session_state.get("filter_classes", []), key="filter_classes",
            placeholder="All categories",
        )
        sensors = st.multiselect(
            "Sensor type", sorted(scored["sensor_type"].unique()),
            default=st.session_state.get("filter_sensors", []), key="filter_sensors",
            placeholder="All sensors",
        )
        min_conf = st.slider("Minimum detector confidence", 0.0, 1.0,
                             float(st.session_state.get("filter_conf", 0.0)), 0.05, key="filter_conf")

    df = scored
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
        df = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]
    if zones:
        df = df[df["zone"].isin(zones)]
    if classes:
        df = df[df["signal_class"].isin(classes)]
    if sensors:
        df = df[df["sensor_type"].isin(sensors)]
    if min_conf:
        df = df[df["confidence"] >= min_conf]
    return df.reset_index(drop=True)


def sidebar_thresholds() -> AlertThresholds:
    """Alert tuning controls (Smart Alerts page)."""
    with st.sidebar:
        st.markdown("### 🎚️ Alert tuning")
        thresholds = AlertThresholds(
            risk_threshold=st.slider("Risk score threshold", 20.0, 95.0, 60.0, 5.0),
            min_cluster_events=st.slider("Minimum events per cluster", 1, 10, 3),
            gunshot_confidence=st.slider("Gunshot confidence", 0.5, 0.99, 0.75, 0.01),
            patrol_gap_hours=st.slider("Patrol gap (hours)", 12.0, 120.0, 48.0, 6.0),
            anomaly_score=st.slider("Anomaly score threshold", 40.0, 100.0, 70.0, 5.0),
            recent_window_hours=st.select_slider("Detection window (hours)", [24, 48, 72, 168], value=48),
        )
    return thresholds


def page_intro(context: Context) -> None:
    """Small status line shown under each feature page header."""
    st.caption(
        f"Source: **{context.source_label}** · {len(context.events):,} detections · "
        f"model trained on {context.risk_model.trained_rows:,} rows"
        + (" · retrained just now" if context.trained_now else " · loaded from model registry")
    )
