"""Plotly figures that explain the models (model card, ROC, importances)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wildguard.config import BAND_COLORS
from wildguard.intelligence.analytics import TEMPLATE, empty_figure, style_figure
from wildguard.ml.risk_model import ModelMetrics


def feature_importance_chart(importances: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if importances.empty:
        return empty_figure("Model has no importances yet")
    top = importances.head(top_n).sort_values("importance")
    fig = px.bar(
        top, x="importance", y="label", orientation="h", template=TEMPLATE,
        color="importance", color_continuous_scale="Oranges",
        title="What drives the risk model (permutation importance, holdout set)",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Drop in ROC AUC when shuffled")
    return style_figure(fig)


def roc_chart(metrics: ModelMetrics) -> go.Figure:
    curve = metrics.roc_curve or {}
    if not curve.get("fpr"):
        return empty_figure("ROC curve unavailable (single-class holdout)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines",
                             name=f"ROC (AUC={metrics.roc_auc:.3f})", line=dict(color="#ffa421", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#5a6480", dash="dash")))
    fig.update_layout(template=TEMPLATE, title="ROC curve - holdout set",
                      xaxis_title="False positive rate", yaxis_title="True positive rate")
    return style_figure(fig)


def confusion_chart(metrics: ModelMetrics) -> go.Figure:
    if not metrics.confusion:
        return empty_figure("No confusion matrix available")
    fig = px.imshow(
        metrics.confusion, text_auto=True, template=TEMPLATE, color_continuous_scale="Blues",
        x=["Predicted: safe", "Predicted: poaching"], y=["Actual: safe", "Actual: poaching"],
        title="Confusion matrix @ threshold 0.5",
    )
    fig.update_layout(coloraxis_showscale=False)
    return style_figure(fig)


def risk_distribution(scored: pd.DataFrame) -> go.Figure:
    if scored.empty:
        return empty_figure("No detections in range")
    fig = px.histogram(
        scored, x="risk_score", color="risk_band", nbins=40, template=TEMPLATE,
        color_discrete_map=BAND_COLORS, title="Distribution of predicted risk scores",
    )
    fig.update_layout(xaxis_title="Risk score (0-100)", yaxis_title="Detections")
    return style_figure(fig)


def zone_risk_chart(profile: pd.DataFrame) -> go.Figure:
    if profile.empty:
        return empty_figure("No zones in range")
    fig = px.bar(
        profile.sort_values("mean_risk"), x="mean_risk", y="zone", orientation="h",
        template=TEMPLATE, color="risk_band", color_discrete_map=BAND_COLORS,
        hover_data=["events", "high_risk_events", "threat_signals", "patrol_coverage"],
        title="Mean predicted risk by zone",
    )
    fig.update_layout(xaxis_title="Mean risk score", yaxis_title="")
    return style_figure(fig)


def alert_severity_chart(summary: dict[str, int]) -> go.Figure:
    data = pd.DataFrame({"severity": list(summary), "alerts": list(summary.values())})
    if data["alerts"].sum() == 0:
        return empty_figure("No alerts for the current thresholds")
    fig = px.bar(
        data, x="severity", y="alerts", template=TEMPLATE, color="severity",
        color_discrete_map=BAND_COLORS, title="Alerts by severity",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    return style_figure(fig)


def alerts_over_time(alerts: pd.DataFrame) -> go.Figure:
    if alerts.empty:
        return empty_figure("No alerts for the current thresholds")
    df = alerts.copy()
    df["day"] = pd.to_datetime(df["last_seen"]).dt.date
    counts = df.groupby(["day", "severity"]).size().reset_index(name="alerts")
    fig = px.bar(counts, x="day", y="alerts", color="severity", template=TEMPLATE,
                 color_discrete_map=BAND_COLORS, title="Alert volume by day")
    fig.update_layout(xaxis_title="", yaxis_title="")
    return style_figure(fig)
