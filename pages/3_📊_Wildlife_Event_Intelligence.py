"""Feature 3 - Wildlife Event Intelligence: species, sensor and habitat analytics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wildguard.intelligence.analytics import (
    activity_timeline,
    detection_map,
    diversity_chart,
    hourly_heatmap,
    sensor_contribution,
    sensor_health,
    signal_sunburst,
    species_breakdown,
    zone_biodiversity,
)
from wildguard.ui.components import download_csv, footer, hero, kpi_row, section
from wildguard.ui.state import get_context, page_intro, sidebar_data_source, sidebar_filters
from wildguard.ui.theme import setup_page

setup_page("Wildlife Event Intelligence", "📊")
sidebar_data_source()

context = get_context()
filtered = sidebar_filters(context.scored)

hero(
    "📊 Wildlife Event Intelligence",
    "Understand what the reserve's sensors are seeing: species activity, temporal patterns, "
    "biodiversity per zone, sensor health and how detections map onto the landscape.",
    tags=["Species analytics", "Shannon diversity", "Sensor health", "Geospatial view"],
)
page_intro(context)

if filtered.empty:
    st.warning("No detections match the current filters.")
    st.stop()

wildlife = filtered[filtered["signal_class"] == "wildlife"]
kpi_row(
    {
        "Detections": f"{len(filtered):,}",
        "Wildlife sightings": f"{len(wildlife):,}",
        "Species observed": f"{wildlife['signal'].nunique()}",
        "Active sensors": f"{filtered['sensor_id'].nunique()}",
        "Night activity": f"{filtered['is_night'].mean():.0%}",
    }
)

tab_activity, tab_species, tab_sensors, tab_map = st.tabs(
    ["📈 Activity", "🦏 Species & biodiversity", "🛰️ Sensor health", "🗺️ Map & export"]
)

with tab_activity:
    granularity = st.radio("Granularity", ["1D", "12h", "6h", "1h"], horizontal=True, index=0)
    st.plotly_chart(activity_timeline(filtered, granularity), use_container_width=True)
    left, right = st.columns(2, gap="large")
    left.plotly_chart(hourly_heatmap(filtered), use_container_width=True)
    right.plotly_chart(signal_sunburst(filtered), use_container_width=True)

with tab_species:
    left, right = st.columns([1.1, 1], gap="large")
    left.plotly_chart(species_breakdown(filtered), use_container_width=True)
    diversity = zone_biodiversity(filtered)
    right.plotly_chart(diversity_chart(diversity), use_container_width=True)

    section("Biodiversity by zone", "Shannon index over species sightings, with endangered-species counts.")
    st.dataframe(diversity, use_container_width=True, hide_index=True)

    section("Species activity pattern", "When each species is most active (share of its sightings by hour).")
    if not wildlife.empty:
        pattern = (
            pd.crosstab(wildlife["signal"], wildlife["hour"], normalize="index")
            .round(3)
            .reindex(columns=range(24), fill_value=0)
        )
        st.dataframe(
            pattern.style.background_gradient(cmap="YlGn", axis=1).format("{:.2f}"),
            use_container_width=True,
        )

with tab_sensors:
    health = sensor_health(filtered)
    offline = int((health["status"] == "Offline").sum())
    degraded = int((health["status"] == "Degraded").sum())
    kpi_row(
        {
            "Sensors reporting": f"{len(health)}",
            "Offline": f"{offline}",
            "Degraded": f"{degraded}",
            "Mean confidence": f"{filtered['confidence'].mean():.2f}",
        }
    )
    st.plotly_chart(sensor_contribution(filtered), use_container_width=True)
    section("Sensor fleet", "Sorted by time since last transmission — the top rows need a field visit.")
    st.dataframe(
        health,
        use_container_width=True,
        hide_index=True,
        column_config={
            "last_seen": st.column_config.DatetimeColumn("Last seen", format="MMM DD, HH:mm"),
            "hours_silent": st.column_config.NumberColumn("Hours silent", format="%.1f"),
        },
    )

with tab_map:
    st.plotly_chart(detection_map(filtered), use_container_width=True)
    section("Filtered event log", "The exact rows behind every chart on this page.")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    download_csv(filtered, "⬇️ Download filtered events (CSV)", "wildguard_events.csv", key="dl_events")

footer()
