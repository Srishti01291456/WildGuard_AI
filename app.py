"""WildGuard AI - Threat Intelligence Dashboard (Streamlit).

Three headline features:
  1. Threat Intelligence  - IOC correlation + explainable risk scoring
  2. Smart Alerts         - rule + z-score detection engine mapped to MITRE ATT&CK
  3. Event Analytics      - interactive Plotly analytics over the event log
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src import alerts as alert_engine
from src import analytics, data_loader, threat_intel

st.set_page_config(
    page_title="WildGuard AI - Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 2rem;}
.hero {
  background: linear-gradient(120deg, #17203a 0%, #2b1533 100%);
  border: 1px solid #33405e; border-radius: 14px;
  padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.0rem;}
.hero p {margin: .35rem 0 0; color: #b9c2d8;}
.feature {
  background: #161b26; border: 1px solid #2b3348; border-left: 4px solid #ff4b4b;
  border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
}
.feature h4 {margin: 0 0 .35rem 0; font-size: 1.05rem;}
.feature p {margin: 0; color: #9aa4bd; font-size: .87rem; line-height: 1.35rem;}
.feature.b {border-left-color: #ffa421;}
.feature.c {border-left-color: #00c0f2;}
.pill {display:inline-block; padding:.12rem .5rem; border-radius:999px;
       font-size:.72rem; background:#242c3e; color:#c9d3ea; margin-top:.5rem;}
</style>
"""


@st.cache_data(show_spinner="Loading security telemetry…")
def load_default_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return data_loader.bootstrap_sample_data()


@st.cache_data(show_spinner=False)
def load_uploaded(events_bytes: bytes | None, ioc_bytes: bytes | None):
    events, ioc = load_default_data()
    if events_bytes:
        events = data_loader.load_events(io.BytesIO(events_bytes))
    if ioc_bytes:
        ioc = data_loader.load_ioc(io.BytesIO(ioc_bytes))
    return events, ioc


def hero() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>🛡️ WildGuard AI — Threat Intelligence Dashboard</h1>
          <p>SOC-style analytics over security event logs — IOC enrichment, automated detections
          and interactive analytics in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="feature"><h4>🛡️ Threat Intelligence</h4>
            <p>Every source IP is correlated against a threat-intel feed (AbuseIPDB / OTX / MISP style) and scored 0–100
            with an explainable model: IOC confidence, severity, block ratio, volume and target spread.</p>
            <span class="pill">IOC correlation · explainable risk score</span></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="feature b"><h4>🚨 Smart Alerts</h4>
            <p>A detection engine combining rules and statistics: brute force, allowed known-bad traffic, exfiltration,
            recon scans, off-hours privileged use and z-score volume anomalies — each mapped to MITRE ATT&CK.</p>
            <span class="pill">6 detections · MITRE ATT&CK mapped</span></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="feature c"><h4>📊 Event Analytics</h4>
            <p>Interactive Plotly analytics: volume timeline, event-type breakdown, hour-of-day heatmap, geo split and a
            risk-vs-activity scatter — all responding to the sidebar filters and exportable to CSV.</p>
            <span class="pill">5 interactive views · CSV export</span></div>""",
            unsafe_allow_html=True,
        )
    st.markdown("")


def sidebar(events: pd.DataFrame):
    st.sidebar.header("⚙️ Data & filters")
    with st.sidebar.expander("Upload your own CSVs", expanded=False):
        st.caption("Events need: timestamp, source_ip, event_type. IOC feed needs: indicator.")
        events_file = st.file_uploader("Event log CSV", type="csv", key="events_csv")
        ioc_file = st.file_uploader("IOC feed CSV", type="csv", key="ioc_csv")

    min_d, max_d = events["timestamp"].min().date(), events["timestamp"].max().date()
    date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    event_types = st.sidebar.multiselect(
        "Event types", sorted(events["event_type"].unique()), default=sorted(events["event_type"].unique())
    )
    severities = st.sidebar.slider("Severity range", 1, 5, (1, 5))
    actions = st.sidebar.multiselect(
        "Action", sorted(events["action"].unique()), default=sorted(events["action"].unique())
    )
    only_malicious = st.sidebar.checkbox("Only IOC matches", value=False)

    st.sidebar.divider()
    st.sidebar.subheader("Detection tuning")
    bf = st.sidebar.slider("Brute-force: failed logins / 10 min", 5, 50, 15)
    exfil = st.sidebar.slider("Exfiltration threshold (MB)", 10, 500, 50, step=10)
    z = st.sidebar.slider("Volume anomaly z-score", 1.5, 5.0, 3.0, step=0.5)

    return dict(
        events_file=events_file, ioc_file=ioc_file, date_range=date_range,
        event_types=event_types, severities=severities, actions=actions,
        only_malicious=only_malicious, bf=bf, exfil=exfil, z=z,
    )


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df
    if isinstance(f["date_range"], (tuple, list)) and len(f["date_range"]) == 2:
        start, end = f["date_range"]
        out = out[(out["timestamp"].dt.date >= start) & (out["timestamp"].dt.date <= end)]
    if f["event_types"]:
        out = out[out["event_type"].isin(f["event_types"])]
    if f["actions"]:
        out = out[out["action"].isin(f["actions"])]
    lo, hi = f["severities"]
    out = out[out["severity"].between(lo, hi)]
    if f["only_malicious"]:
        out = out[out["is_malicious"]]
    return out


def tab_threat_intel(filtered: pd.DataFrame, scores: pd.DataFrame, ioc: pd.DataFrame) -> None:
    st.subheader("🛡️ Threat Intelligence")
    st.caption("Source IPs correlated with the threat-intel feed and ranked by an explainable risk score.")

    left, right = st.columns([2, 1])
    with left:
        band_filter = st.multiselect(
            "Risk band", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"]
        )
        table = scores[scores["risk_band"].isin(band_filter)] if band_filter else scores
        st.dataframe(
            table[
                ["source_ip", "risk_score", "risk_band", "threat_type", "ioc_confidence",
                 "events", "avg_severity", "targets", "intel_source", "last_seen"]
            ],
            use_container_width=True, hide_index=True, height=420,
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f"),
                "ioc_confidence": st.column_config.NumberColumn("IOC conf %"),
            },
        )
        st.download_button(
            "⬇️ Download risk report (CSV)", table.to_csv(index=False).encode(),
            file_name="risk_report.csv", mime="text/csv",
        )
    with right:
        if scores.empty:
            st.info("No sources match the current filters.")
            return
        options = scores["source_ip"].head(50).tolist()
        selected = st.selectbox("Investigate a source IP", options)
        row = scores[scores["source_ip"] == selected].iloc[0]
        st.metric("Risk score", f"{row['risk_score']:.1f}", row["risk_band"])
        st.write("**Score breakdown**")
        st.bar_chart(pd.Series(threat_intel.score_breakdown(row)))
        st.write("**Recent events**")
        st.dataframe(
            filtered[filtered["source_ip"] == selected]
            .sort_values("timestamp", ascending=False)
            .head(10)[["timestamp", "event_type", "user", "action", "severity"]],
            use_container_width=True, hide_index=True,
        )

    with st.expander("Threat-intel feed used for enrichment"):
        st.dataframe(ioc, use_container_width=True, hide_index=True)


def tab_alerts(alerts: pd.DataFrame) -> None:
    st.subheader("🚨 Smart Alerts")
    st.caption("Rule-based and statistical detections. Tune thresholds in the sidebar.")

    summary = alert_engine.alert_summary(alerts)
    cols = st.columns(4)
    for col, (label, count) in zip(cols, summary.items(), strict=False):
        col.metric(label, count)

    if alerts.empty:
        st.success("No alerts triggered for the current filters and thresholds.")
        return

    chosen = st.multiselect(
        "Severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium"]
    )
    view = alerts[alerts["severity"].isin(chosen)] if chosen else alerts

    for _, a in view.iterrows():
        icon = {"Critical": "🔴", "High": "🟠", "Medium": "🔵", "Low": "🟢"}.get(a["severity"], "⚪")
        with st.expander(f"{icon} **{a['rule']}** — `{a['entity']}` · {a['detail']}"):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Severity:** {a['severity']}")
            c2.write(f"**Events:** {a['events']}")
            c3.write(f"**MITRE:** {a['mitre']}")
            st.write(f"**Window:** {a['first_seen']:%Y-%m-%d %H:%M} → {a['last_seen']:%Y-%m-%d %H:%M} UTC")

    st.download_button(
        "⬇️ Download alerts (CSV)", view.to_csv(index=False).encode(),
        file_name="alerts.csv", mime="text/csv",
    )


def tab_analytics(filtered: pd.DataFrame, scores: pd.DataFrame) -> None:
    st.subheader("📊 Event Analytics")
    freq = st.radio("Granularity", ["30min", "1h", "6h", "1D"], index=1, horizontal=True)
    st.plotly_chart(analytics.timeline(filtered, freq), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(analytics.by_event_type(filtered), use_container_width=True)
    c2.plotly_chart(analytics.top_countries(filtered), use_container_width=True)

    st.plotly_chart(analytics.severity_heatmap(filtered), use_container_width=True)
    st.plotly_chart(analytics.risk_scatter(scores), use_container_width=True)

    with st.expander("Raw events"):
        st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download filtered events (CSV)", filtered.to_csv(index=False).encode(),
            file_name="events_filtered.csv", mime="text/csv",
        )


def main() -> None:
    hero()

    base_events, base_ioc = load_default_data()
    f = sidebar(base_events)

    events, ioc = load_uploaded(
        f["events_file"].getvalue() if f["events_file"] else None,
        f["ioc_file"].getvalue() if f["ioc_file"] else None,
    )

    enriched = threat_intel.enrich_with_ioc(events, ioc)
    filtered = apply_filters(enriched, f)
    scores = threat_intel.score_sources(filtered)
    alerts = alert_engine.run_rules(
        filtered, brute_force_threshold=f["bf"], exfil_mb=f["exfil"], zscore_threshold=f["z"]
    )

    kpi_values = analytics.kpis(filtered, scores, alerts)
    for col, (label, value) in zip(st.columns(len(kpi_values)), kpi_values.items(), strict=True):
        col.metric(label, value)
    st.divider()

    t1, t2, t3 = st.tabs(["🛡️ Threat Intelligence", "🚨 Smart Alerts", "📊 Event Analytics"])
    with t1:
        tab_threat_intel(filtered, scores, ioc)
    with t2:
        tab_alerts(alerts)
    with t3:
        tab_analytics(filtered, scores)

    st.caption("Data is synthetic and generated locally — safe for demos. Upload your own CSVs from the sidebar.")


if __name__ == "__main__":
    main()
