"""Page configuration and the shared CSS skin."""

from __future__ import annotations

import streamlit as st

APP_TITLE = "WildGuard AI"
APP_ICON = "🐘"

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}

.wg-hero {
  background: linear-gradient(120deg, #0f2c25 0%, #123049 55%, #2a1a3d 100%);
  border: 1px solid #23405a; border-radius: 16px;
  padding: 1.5rem 1.8rem; margin-bottom: 1.3rem;
}
.wg-hero h1 {margin: 0; font-size: 2.05rem; letter-spacing: -.5px;}
.wg-hero p {margin: .45rem 0 0; color: #b7c6d8; font-size: .97rem; max-width: 80ch;}
.wg-tagline {margin-top: .8rem;}
.wg-tag {
  display: inline-block; padding: .18rem .62rem; border-radius: 999px; margin-right: .4rem;
  font-size: .74rem; background: #16283a; color: #9fd6ff; border: 1px solid #23485f;
}

.wg-card {
  background: #121722; border: 1px solid #263042; border-left: 4px solid #21c354;
  border-radius: 14px; padding: 1.05rem 1.15rem; height: 100%;
}
.wg-card h4 {margin: 0 0 .4rem 0; font-size: 1.04rem;}
.wg-card p {margin: 0; color: #97a3b8; font-size: .86rem; line-height: 1.38rem;}
.wg-card .wg-chip {
  display: inline-block; margin-top: .65rem; padding: .12rem .5rem; border-radius: 6px;
  font-size: .71rem; background: #1b2433; color: #c6d2e6; border: 1px solid #2b3448;
}
.wg-card.b {border-left-color: #ffa421;}
.wg-card.c {border-left-color: #00c0f2;}

.wg-section {margin: 1.5rem 0 .3rem;}
.wg-section h3 {margin: 0; font-size: 1.25rem;}
.wg-section p {margin: .2rem 0 0; color: #8794ab; font-size: .87rem;}

div[data-testid="stMetric"] {
  background: #121722; border: 1px solid #263042; border-radius: 12px; padding: .75rem .9rem;
}
div[data-testid="stMetricLabel"] p {color: #8794ab; font-size: .8rem;}

.wg-footer {margin-top: 2.5rem; color: #66718a; font-size: .8rem; text-align: center;}
</style>
"""


def setup_page(page_title: str, page_icon: str = APP_ICON, layout: str = "wide") -> None:
    """Standard page setup - call once at the top of every page."""
    st.set_page_config(
        page_title=f"{APP_TITLE} · {page_title}",
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
