"""Reusable Streamlit building blocks (hero, feature cards, KPI row…)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

FEATURES = [
    {
        "css": "",
        "icon": "🧠",
        "title": "Threat Intelligence",
        "body": (
            "A gradient-boosted classifier scores every detection with a 0-100 poaching-risk "
            "probability, then rolls it up into zone threat profiles, KMeans hotspots and a "
            "full model card with ROC, confusion matrix and permutation importances."
        ),
        "chip": "HistGradientBoosting · KMeans",
        "page": "pages/1_🧠_Threat_Intelligence.py",
    },
    {
        "css": "b",
        "icon": "🚨",
        "title": "Smart Alert System",
        "body": (
            "Model scores, an Isolation Forest anomaly detector and nine ranger-focused rules "
            "are fused into one ranked alert queue - each alert carries its evidence, severity "
            "and a recommended field response, with live-tunable thresholds."
        ),
        "chip": "Isolation Forest · rule fusion",
        "page": "pages/2_🚨_Smart_Alerts.py",
    },
    {
        "css": "c",
        "icon": "📊",
        "title": "Wildlife Event Intelligence",
        "body": (
            "Interactive analytics over the event stream: species activity, hour-by-zone "
            "heatmaps, Shannon biodiversity per zone, signal mix, sensor health and a "
            "risk-weighted detection map - all filterable and exportable."
        ),
        "chip": "Plotly · biodiversity metrics",
        "page": "pages/3_📊_Wildlife_Event_Intelligence.py",
    },
]


def hero(title: str, subtitle: str, tags: list[str] | None = None) -> None:
    tag_html = "".join(f'<span class="wg-tag">{t}</span>' for t in (tags or []))
    st.markdown(
        f"""
        <div class="wg-hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div class="wg-tagline">{tag_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_cards(with_links: bool = True) -> None:
    """The three headline features, shown at the top of the home page."""
    columns = st.columns(3, gap="medium")
    for column, feature in zip(columns, FEATURES, strict=False):
        with column:
            st.markdown(
                f"""
                <div class="wg-card {feature['css']}">
                  <h4>{feature['icon']} {feature['title']}</h4>
                  <p>{feature['body']}</p>
                  <span class="wg-chip">{feature['chip']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if with_links:
                st.page_link(feature["page"], label=f"Open {feature['title']}", icon="➡️")


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="wg-section"><h3>{title}</h3><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_row(metrics: dict[str, str]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items(), strict=False):
        column.metric(label, value)


def download_csv(df: pd.DataFrame, label: str, filename: str, key: str | None = None) -> None:
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="wg-footer">WildGuard AI · synthetic demo data · '
        "models retrain automatically when the dataset changes</div>",
        unsafe_allow_html=True,
    )
