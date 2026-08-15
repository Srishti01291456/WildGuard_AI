"""Feature 3 - Event Analytics: aggregations and Plotly figures."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"
PALETTE = ["#ff4b4b", "#ffa421", "#00c0f2", "#21c354", "#a56eff", "#f5f5f5"]


def timeline(df: pd.DataFrame, freq: str = "1h") -> go.Figure:
    if df.empty:
        return _empty("No events in range")
    series = (
        df.set_index("timestamp")
        .groupby("action")
        .resample(freq)
        .size()
        .rename("events")
        .reset_index()
    )
    fig = px.area(
        series, x="timestamp", y="events", color="action",
        template=TEMPLATE, color_discrete_sequence=PALETTE,
        title="Event volume over time",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def by_event_type(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty("No events in range")
    counts = df["event_type"].value_counts().reset_index()
    counts.columns = ["event_type", "events"]
    fig = px.bar(
        counts, x="events", y="event_type", orientation="h",
        template=TEMPLATE, color="events", color_continuous_scale="Reds",
        title="Events by type",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False,
                      yaxis=dict(categoryorder="total ascending"))
    return fig


def severity_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty("No events in range")
    pivot = (
        df.pivot_table(index="event_type", columns="hour", values="event_id", aggfunc="count")
        .reindex(columns=range(24))
        .fillna(0)
    )
    fig = px.imshow(
        pivot, template=TEMPLATE, color_continuous_scale="Inferno", aspect="auto",
        labels=dict(x="Hour of day (UTC)", y="", color="events"),
        title="Activity heatmap by hour",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def top_countries(df: pd.DataFrame, n: int = 10) -> go.Figure:
    if df.empty:
        return _empty("No events in range")
    counts = df["country"].value_counts().head(n).reset_index()
    counts.columns = ["country", "events"]
    fig = px.pie(
        counts, names="country", values="events", hole=0.55,
        template=TEMPLATE, color_discrete_sequence=PALETTE,
        title="Top source countries",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def risk_scatter(scores: pd.DataFrame) -> go.Figure:
    if scores.empty:
        return _empty("No sources in range")
    fig = px.scatter(
        scores.head(150), x="events", y="risk_score", size="bytes_out", color="risk_band",
        hover_name="source_ip", hover_data=["threat_type", "targets", "avg_severity"],
        template=TEMPLATE, size_max=40,
        color_discrete_map={"Critical": "#ff4b4b", "High": "#ffa421", "Medium": "#00c0f2", "Low": "#21c354"},
        title="Risk score vs activity per source IP",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def kpis(df: pd.DataFrame, scores: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, str]:
    malicious = int(df["is_malicious"].sum()) if "is_malicious" in df else 0
    critical = int((alerts["severity"] == "Critical").sum()) if not alerts.empty else 0
    return {
        "Total events": f"{len(df):,}",
        "Unique source IPs": f"{df['source_ip'].nunique():,}" if not df.empty else "0",
        "Malicious IOC hits": f"{malicious:,}",
        "Critical alerts": f"{critical:,}",
        "Data out (GB)": f"{df['bytes_out'].sum() / 1e9:,.2f}" if not df.empty else "0.00",
    }


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#888"))
    fig.update_layout(template=TEMPLATE, xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig
