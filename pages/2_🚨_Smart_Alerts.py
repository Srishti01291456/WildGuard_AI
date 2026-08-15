"""Feature 2 - Smart Alert System: fused ML + rule alerting for rangers."""

from __future__ import annotations

import streamlit as st

from wildguard.config import BAND_COLORS
from wildguard.intelligence.alerts import AlertEngine, alert_summary
from wildguard.intelligence.modelviz import alert_severity_chart, alerts_over_time
from wildguard.ui.components import download_csv, footer, hero, kpi_row, section
from wildguard.ui.state import (
    get_context,
    page_intro,
    sidebar_data_source,
    sidebar_filters,
    sidebar_thresholds,
)
from wildguard.ui.theme import setup_page

setup_page("Smart Alerts", "🚨")
sidebar_data_source()

context = get_context()
filtered = sidebar_filters(context.scored)
thresholds = sidebar_thresholds()

hero(
    "🚨 Smart Alert System",
    "Model probabilities, unsupervised anomalies and ranger-focused rules are fused into a single "
    "ranked queue. Every alert explains why it fired and what the team should do next.",
    tags=["Rule + ML fusion", "Isolation Forest", "Recommended actions", "Live thresholds"],
)
page_intro(context)

if filtered.empty:
    st.warning("No detections match the current filters.")
    st.stop()

anomalies = context.anomalies
if not anomalies.empty:
    anomalies = anomalies[anomalies["zone"].isin(filtered["zone"].unique())]

alerts = AlertEngine(thresholds).run(filtered, anomalies)
summary = alert_summary(alerts)

kpi_row(
    {
        "Total alerts": f"{len(alerts):,}",
        "Critical": f"{summary['Critical']}",
        "High": f"{summary['High']}",
        "Medium": f"{summary['Medium']}",
        "Detection rules": "9",
    }
)

left, right = st.columns([1, 1], gap="large")
left.plotly_chart(alert_severity_chart(summary), use_container_width=True)
right.plotly_chart(alerts_over_time(alerts), use_container_width=True)

section("Alert queue", "Ranked by severity, then by model risk. Filter by severity to triage.")

if alerts.empty:
    st.success("No alerts fired for the current thresholds — loosen them in the sidebar to see more.")
else:
    chosen = st.multiselect(
        "Severity", ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium"],
    )
    queue = alerts[alerts["severity"].isin(chosen)] if chosen else alerts

    st.dataframe(
        queue,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rule": st.column_config.TextColumn("Rule", width="medium"),
            "detail": st.column_config.TextColumn("Evidence", width="large"),
            "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
            "first_seen": st.column_config.DatetimeColumn("First seen", format="MMM DD, HH:mm"),
            "last_seen": st.column_config.DatetimeColumn("Last seen", format="MMM DD, HH:mm"),
        },
    )
    download_csv(queue, "⬇️ Download alert queue (CSV)", "wildguard_alerts.csv", key="dl_alerts")

    section("Priority briefings", "The top alerts written up for the duty ranger.")
    for _, alert in queue.head(5).iterrows():
        colour = BAND_COLORS.get(alert["severity"], "#8794ab")
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{colour};font-weight:600'>{alert['severity'].upper()}</span> · "
                f"**{alert['rule']}** — {alert['entity']}",
                unsafe_allow_html=True,
            )
            st.write(alert["detail"])
            st.caption(f"➡️ {alert['recommended_action']}")

section("Detection logic", "Each rule combines model output with field knowledge.")
st.markdown(
    """
| Rule | Trigger | Severity |
| --- | --- | --- |
| Gunshot detected | Acoustic gunshot above the confidence threshold | Critical |
| ML high-risk detection cluster | ≥ N detections above the risk threshold in one zone/window | Critical / High |
| Night-time boundary intrusion | Human, vehicle or campfire at night within 3 km of the boundary | High |
| Snare line suspected | Repeated snare-trap detections in a zone | High / Medium |
| Illegal logging activity | Chainsaw detections in a zone | High / Medium |
| Endangered species exposed | Protected species sighted in a high-risk zone | High |
| Patrol coverage gap | Long patrol gap combined with elevated model risk | Medium |
| Behavioural anomaly | Isolation Forest flags an unusual zone/day activity profile | High / Medium |
| Sensor offline | No telemetry from a sensor for longer than the silence window | Low |
"""
)

footer()
