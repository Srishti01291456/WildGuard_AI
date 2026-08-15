"""Feature 1 - Threat Intelligence: risk model, hotspots and explainability."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wildguard.config import ZONES, risk_band
from wildguard.data.generator import signal_class
from wildguard.intelligence.analytics import hotspot_map
from wildguard.intelligence.modelviz import (
    confusion_chart,
    feature_importance_chart,
    risk_distribution,
    roc_chart,
    zone_risk_chart,
)
from wildguard.ml.hotspots import cluster_hotspots
from wildguard.ml.risk_model import zone_threat_profile
from wildguard.ui.components import download_csv, footer, hero, kpi_row, section
from wildguard.ui.state import get_context, page_intro, sidebar_data_source, sidebar_filters
from wildguard.ui.theme import setup_page

setup_page("Threat Intelligence", "🧠")
sidebar_data_source()

context = get_context()
filtered = sidebar_filters(context.scored)

hero(
    "🧠 Threat Intelligence",
    "A gradient-boosted classifier estimates the probability that each detection belongs to a "
    "poaching event, and the output is rolled up into zone profiles, geospatial hotspots and an "
    "explainable model card.",
    tags=["HistGradientBoostingClassifier", "Permutation importance", "KMeans hotspots"],
)
page_intro(context)

if filtered.empty:
    st.warning("No detections match the current filters.")
    st.stop()

high_risk = filtered[filtered["risk_band"].isin(["High", "Critical"])]
kpi_row(
    {
        "Detections scored": f"{len(filtered):,}",
        "Mean risk": f"{filtered['risk_score'].mean():.0f}/100",
        "High/Critical": f"{len(high_risk):,}",
        "Riskiest zone": zone_threat_profile(filtered).iloc[0]["zone"],
        "Model ROC AUC": f"{context.risk_model.metrics.roc_auc:.3f}" if context.risk_model.metrics else "-",
    }
)

tab_zones, tab_hotspots, tab_model, tab_whatif = st.tabs(
    ["🗺️ Zone threat profile", "📍 Hotspots", "🔬 Model card", "🧪 What-if scoring"]
)

with tab_zones:
    section("Predicted risk by zone", "Mean model probability per zone, with the evidence behind it.")
    profile = zone_threat_profile(filtered)
    st.plotly_chart(zone_risk_chart(profile), use_container_width=True)
    st.plotly_chart(risk_distribution(filtered), use_container_width=True)

    section("Highest-risk detections", "Individual detections ranked by model probability.")
    top = filtered.sort_values("risk_score", ascending=False).head(200)
    st.dataframe(
        top[["timestamp", "zone", "signal", "sensor_type", "confidence", "is_night",
             "patrol_coverage", "distance_to_boundary_km", "risk_score", "risk_band"]],
        use_container_width=True, hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
            "timestamp": st.column_config.DatetimeColumn("Time", format="MMM DD, HH:mm"),
        },
    )
    download_csv(top, "⬇️ Download scored detections (CSV)", "wildguard_risk_scores.csv", key="dl_scores")

with tab_hotspots:
    section("Poaching hotspots", "KMeans clusters over detection coordinates, weighted by predicted risk.")
    n_clusters = st.slider("Number of hotspots", 2, 8, 5)
    hotspots = cluster_hotspots(filtered, n_clusters=n_clusters)
    st.plotly_chart(hotspot_map(filtered, hotspots), use_container_width=True)
    st.dataframe(hotspots, use_container_width=True, hide_index=True)

with tab_model:
    metrics = context.risk_model.metrics
    section("Model card", "Everything about how the risk model was trained and how well it performs.")
    if metrics is None:
        st.info("Model metrics are unavailable — retrain from the sidebar.")
    else:
        kpi_row(metrics.as_dict())
        left, right = st.columns(2, gap="large")
        left.plotly_chart(roc_chart(metrics), use_container_width=True)
        right.plotly_chart(confusion_chart(metrics), use_container_width=True)

    st.plotly_chart(feature_importance_chart(context.risk_model.importances), use_container_width=True)
    st.markdown(
        """
**Algorithm** `HistGradientBoostingClassifier` (250 iterations, lr 0.08, depth 6, early stopping)
**Pipeline** `StandardScaler` on 13 numeric features + `OneHotEncoder` on 5 categorical features
**Split** stratified 75 / 25 train-test split, permutation importance computed on the holdout set
**Target** `is_poaching_incident` — confirmed poaching activity linked to the detection
        """
    )

with tab_whatif:
    section("What-if scoring", "Score a hypothetical detection with the trained model.")
    columns = st.columns(3)
    with columns[0]:
        zone = st.selectbox("Zone", ZONES)
        signal = st.selectbox(
            "Signal", ["gunshot", "chainsaw", "snare_trap", "vehicle", "human_presence", "elephant", "rhino"]
        )
        sensor_type = st.selectbox("Sensor", ["acoustic_sensor", "camera_trap", "drone_patrol",
                                              "gps_collar", "ranger_report"])
    with columns[1]:
        confidence = st.slider("Detector confidence", 0.3, 1.0, 0.9, 0.01)
        hour = st.slider("Hour of day", 0, 23, 2)
        patrol_coverage = st.slider("Patrol coverage", 0.0, 1.0, 0.2, 0.05)
    with columns[2]:
        boundary = st.slider("Distance to boundary (km)", 0.1, 18.0, 1.2, 0.1)
        road = st.slider("Distance to road (km)", 0.1, 15.0, 1.0, 0.1)
        hours_since_patrol = st.slider("Hours since last patrol", 0.0, 160.0, 60.0, 1.0)

    candidate = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=hour),
                "hour": hour,
                "zone": zone,
                "signal": signal,
                "signal_class": signal_class(signal),
                "sensor_type": sensor_type,
                "confidence": confidence,
                "detected_count": 2,
                "distance_to_boundary_km": boundary,
                "distance_to_road_km": road,
                "distance_to_village_km": 6.0,
                "patrol_coverage": patrol_coverage,
                "hours_since_patrol": hours_since_patrol,
                "temperature_c": 18.0 if hour < 6 or hour > 19 else 29.0,
                "is_night": int(hour >= 19 or hour <= 5),
                "weather": "clear",
            }
        ]
    )
    probability = float(context.risk_model.predict_proba(candidate)[0])
    band = risk_band(probability)
    st.metric("Predicted poaching probability", f"{probability:.1%}", band)
    st.progress(min(probability, 1.0))
    st.caption("The same pipeline that scores live telemetry — useful for sanity-checking the model's behaviour.")

footer()
