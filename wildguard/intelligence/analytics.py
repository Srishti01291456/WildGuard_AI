"""Feature 3 - Wildlife Event Intelligence: aggregations and Plotly figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wildguard.config import BAND_COLORS, ENDANGERED_SPECIES

TEMPLATE = "plotly_dark"
PALETTE = ["#21c354", "#00c0f2", "#ffa421", "#ff4b4b", "#a56eff", "#f5f5f5", "#ff8fab"]


# -- KPIs ---------------------------------------------------------------------


def kpis(scored: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, str]:
    if scored.empty:
        return {"Detections": "0", "Mean risk": "0", "High-risk zones": "0",
                "Critical alerts": "0", "Species observed": "0"}
    high_zones = scored.groupby("zone")["risk_score"].mean().pipe(lambda s: int((s >= 50).sum()))
    critical = int((alerts["severity"] == "Critical").sum()) if not alerts.empty else 0
    species = scored[scored["signal_class"] == "wildlife"]["signal"].nunique()
    return {
        "Detections": f"{len(scored):,}",
        "Mean risk": f"{scored['risk_score'].mean():.0f}/100",
        "High-risk zones": f"{high_zones}",
        "Critical alerts": f"{critical}",
        "Species observed": f"{species}",
    }


def shannon_diversity(counts: pd.Series) -> float:
    """Shannon-Wiener biodiversity index of a species count distribution."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def zone_biodiversity(events: pd.DataFrame) -> pd.DataFrame:
    wildlife = events[events["signal_class"] == "wildlife"]
    if wildlife.empty:
        return pd.DataFrame(columns=["zone", "species", "sightings", "diversity_index", "endangered_sightings"])
    rows = []
    for zone, grp in wildlife.groupby("zone"):
        counts = grp["signal"].value_counts()
        rows.append(
            {
                "zone": zone,
                "species": int(counts.size),
                "sightings": int(counts.sum()),
                "diversity_index": round(shannon_diversity(counts), 3),
                "endangered_sightings": int(grp["signal"].isin(ENDANGERED_SPECIES).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("diversity_index", ascending=False).reset_index(drop=True)


def sensor_health(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["sensor_id", "sensor_type", "zone", "detections",
                                     "mean_confidence", "last_seen", "hours_silent", "status"])
    latest = events["timestamp"].max()
    grouped = events.groupby(["sensor_id", "sensor_type", "zone"])
    table = pd.DataFrame(
        {
            "detections": grouped.size(),
            "mean_confidence": grouped["confidence"].mean().round(2),
            "last_seen": grouped["timestamp"].max(),
        }
    ).reset_index()
    table["hours_silent"] = ((latest - table["last_seen"]).dt.total_seconds() / 3600).round(1)
    table["status"] = np.where(table["hours_silent"] > 72, "Offline",
                               np.where(table["hours_silent"] > 24, "Degraded", "Healthy"))
    return table.sort_values("hours_silent", ascending=False).reset_index(drop=True)


# -- Figures ------------------------------------------------------------------


def activity_timeline(events: pd.DataFrame, freq: str = "1D") -> go.Figure:
    if events.empty:
        return empty_figure("No detections in range")
    series = (
        events.set_index("timestamp")
        .groupby("signal_class")["event_id"]
        .resample(freq)
        .size()
        .rename("detections")
        .reset_index()
    )
    fig = px.area(
        series, x="timestamp", y="detections", color="signal_class",
        template=TEMPLATE, color_discrete_sequence=PALETTE,
        title="Detections over time by signal category",
    )
    return style_figure(fig)


def species_breakdown(events: pd.DataFrame, top_n: int = 10) -> go.Figure:
    wildlife = events[events["signal_class"] == "wildlife"]
    if wildlife.empty:
        return empty_figure("No wildlife detections in range")
    counts = wildlife["signal"].value_counts().head(top_n).reset_index()
    counts.columns = ["species", "sightings"]
    fig = px.bar(
        counts, x="sightings", y="species", orientation="h", template=TEMPLATE,
        color="sightings", color_continuous_scale="Greens", title="Species sightings",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
    return style_figure(fig)


def hourly_heatmap(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return empty_figure("No detections in range")
    pivot = (
        events.pivot_table(index="zone", columns="hour", values="event_id", aggfunc="count")
        .reindex(columns=range(24)).fillna(0)
    )
    fig = px.imshow(
        pivot, template=TEMPLATE, color_continuous_scale="Inferno", aspect="auto",
        labels=dict(x="Hour of day", y="", color="detections"),
        title="Activity by hour and zone",
    )
    return style_figure(fig)


def signal_sunburst(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return empty_figure("No detections in range")
    counts = events.groupby(["signal_class", "signal"]).size().reset_index(name="detections")
    fig = px.sunburst(
        counts, path=["signal_class", "signal"], values="detections",
        template=TEMPLATE, color="signal_class", color_discrete_sequence=PALETTE,
        title="Signal mix: wildlife vs human vs threat",
    )
    return style_figure(fig)


def sensor_contribution(events: pd.DataFrame) -> go.Figure:
    if events.empty:
        return empty_figure("No detections in range")
    counts = events["sensor_type"].value_counts().reset_index()
    counts.columns = ["sensor_type", "detections"]
    fig = px.pie(
        counts, names="sensor_type", values="detections", hole=0.55,
        template=TEMPLATE, color_discrete_sequence=PALETTE, title="Detections by sensor type",
    )
    return style_figure(fig)


def diversity_chart(diversity: pd.DataFrame) -> go.Figure:
    if diversity.empty:
        return empty_figure("No wildlife detections in range")
    fig = px.bar(
        diversity, x="zone", y="diversity_index", template=TEMPLATE,
        color="endangered_sightings", color_continuous_scale="Teal",
        title="Shannon biodiversity index by zone",
        hover_data=["species", "sightings", "endangered_sightings"],
    )
    return style_figure(fig)


def detection_map(scored: pd.DataFrame, max_points: int = 1200) -> go.Figure:
    if scored.empty or scored[["latitude", "longitude"]].dropna().empty:
        return empty_figure("No geolocated detections in range")
    df = scored.dropna(subset=["latitude", "longitude"]).tail(max_points)
    fig = px.scatter(
        df, x="longitude", y="latitude", color="risk_band", size="risk_score",
        hover_name="signal", hover_data=["zone", "sensor_type", "timestamp", "risk_score"],
        template=TEMPLATE, size_max=18, color_discrete_map=BAND_COLORS,
        title="Detection map (risk weighted)",
    )
    fig.update_layout(legend_title_text="", xaxis_title="Longitude", yaxis_title="Latitude")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return style_figure(fig)


def hotspot_map(scored: pd.DataFrame, hotspots: pd.DataFrame) -> go.Figure:
    fig = detection_map(scored)
    if hotspots.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=hotspots["longitude"], y=hotspots["latitude"], mode="markers+text",
            marker=dict(symbol="x", size=16, color="#ffffff", line=dict(width=1, color="#000")),
            text=hotspots["hotspot"], textposition="top center", name="Hotspot centroid",
            hovertext=[
                f"{r.hotspot}: {r.events} detections, mean risk {r.mean_risk} ({r.top_zone})"
                for r in hotspots.itertuples()
            ],
            hoverinfo="text",
        )
    )
    fig.update_layout(title="Poaching hotspots (KMeans on risk-weighted detections)")
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=15, color="#8892a8"))
    fig.update_layout(template=TEMPLATE, xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10), height=320)
    return fig


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(margin=dict(l=10, r=10, t=52, b=10), legend_title_text="",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
