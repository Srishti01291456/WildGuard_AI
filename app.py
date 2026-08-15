"""WildGuard AI - Home.

Machine-learning command centre for wildlife protection. Three headline
features, each on its own page:

1. 🧠 Threat Intelligence        - supervised poaching-risk model + hotspots
2. 🚨 Smart Alert System         - anomaly detection + rule fusion
3. 📊 Wildlife Event Intelligence - species, sensor and biodiversity analytics
"""

from __future__ import annotations

import streamlit as st

from wildguard.intelligence.alerts import AlertEngine, alert_summary
from wildguard.intelligence.analytics import kpis
from wildguard.ml.risk_model import zone_threat_profile
from wildguard.ui.components import feature_cards, footer, hero, kpi_row, section
from wildguard.ui.state import get_context, sidebar_data_source, sidebar_filters
from wildguard.ui.theme import setup_page

setup_page("Home")
sidebar_data_source()

context = get_context()
filtered = sidebar_filters(context.scored)

hero(
    "🐘 WildGuard AI — Wildlife Protection Intelligence",
    "An end-to-end AI/ML platform that turns camera-trap, acoustic, collar and ranger telemetry "
    "into poaching-risk predictions, prioritised field alerts and conservation analytics.",
    tags=["scikit-learn", "Gradient boosting", "Isolation Forest", "KMeans hotspots", "Streamlit"],
)

section(
    "Unique features",
    "The three capabilities that make WildGuard AI more than a dashboard — each one is a full page.",
)
feature_cards()

alerts = AlertEngine().run(filtered, context.anomalies)

section("Reserve at a glance", "Live figures for the current filter selection.")
kpi_row(kpis(filtered, alerts))

left, right = st.columns([1.35, 1], gap="large")

with left:
    section("Zones by predicted risk", "Model output aggregated per reserve zone.")
    profile = zone_threat_profile(filtered)
    st.dataframe(
        profile[["zone", "risk_band", "mean_risk", "max_risk", "events", "high_risk_events",
                 "threat_signals", "patrol_coverage", "last_event"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "mean_risk": st.column_config.ProgressColumn("Mean risk", min_value=0, max_value=100, format="%.0f"),
            "patrol_coverage": st.column_config.NumberColumn("Patrol coverage", format="%.2f"),
            "last_event": st.column_config.DatetimeColumn("Last detection", format="MMM DD, HH:mm"),
        },
    )

with right:
    section("Model card", "Holdout performance of the poaching-risk classifier.")
    metrics = context.risk_model.metrics
    if metrics is not None:
        items = list(metrics.as_dict().items())
        for row_start in range(0, len(items), 2):
            columns = st.columns(2)
            for column, (label, value) in zip(columns, items[row_start:row_start + 2], strict=False):
                column.metric(label, value)
        st.caption(
            f"Evaluated on {metrics.support:,} held-out detections "
            f"({metrics.positives:,} confirmed incidents)."
        )

    summary = alert_summary(alerts)
    st.markdown("**Open alerts**")
    st.write(" · ".join(f"{severity}: **{count}**" for severity, count in summary.items()))

section("How the pipeline works", "")
st.markdown(
    """
| Stage | What happens | Module |
| --- | --- | --- |
| **1. Ingest** | Sensor telemetry is validated, defaults are filled and geo/time features derived | `wildguard/data` |
| **2. Features** | Cyclical time encoding, proximity features, scaling, one-hot | `wildguard/ml/features.py` |
| **3. Learn** | HistGradientBoosting risk classifier, Isolation Forest anomalies, KMeans hotspots | `wildguard/ml` |
| **4. Decide** | Model output fused with ranger rules into an alert queue | `wildguard/intelligence/alerts.py` |
| **5. Explain** | Model card, permutation importances, what-if scoring | `pages/1_🧠_Threat_Intelligence.py` |
"""
)

with st.expander("Preview the underlying dataset"):
    st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)

footer()
