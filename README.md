# 🛡️ WildGuard AI — Threat Intelligence Dashboard

A SOC-style Streamlit dashboard that turns raw security event logs into intelligence, alerts and analytics.
Built to be demo-ready for internships and resumes: no external API keys, no database, one command to run.

## Highlight features

| Feature | What it does |
| --- | --- |
| 🛡️ **Threat Intelligence** | Correlates every source IP against a threat-intel feed (AbuseIPDB / OTX / MISP style) and computes an **explainable 0–100 risk score** from IOC confidence, event severity, block ratio, event volume and target spread. Each score can be broken down per component in the UI. |
| 🚨 **Smart Alerts** | Detection engine with 6 rules — credential brute force (rolling window), allowed traffic from known-bad IOCs, data exfiltration, network reconnaissance, off-hours privileged activity and a **z-score volume anomaly detector**. Every alert is mapped to a **MITRE ATT&CK** technique and thresholds are tunable live from the sidebar. |
| 📊 **Event Analytics** | Five interactive Plotly views — volume timeline by action, event-type breakdown, hour-of-day heatmap, source-country split and a risk-vs-activity bubble chart — all driven by the same filters, with CSV export. |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Sample data (~4,000 events + 60 IOCs) is generated on first run into `data/`.

## Use your own data

Upload CSVs from the sidebar.

* **Event log** — required: `timestamp`, `source_ip`, `event_type`. Optional: `dest_ip`, `user`, `country`, `action` (`allowed`/`blocked`), `severity` (1–5), `bytes_out`. Missing optional columns are filled with safe defaults, bad timestamps are dropped.
* **IOC feed** — required: `indicator`. Optional: `threat_type`, `confidence` (0–100), `source`.

## Project layout

```
WildGuard_AI/
├── app.py                     Streamlit UI: hero, KPIs, three feature tabs
├── requirements.txt
├── pyproject.toml             pytest + ruff configuration
├── .streamlit/config.toml     dark theme
├── .vscode/                   VS Code settings, launch/debug, tasks, extensions
│   ├── settings.json
│   ├── launch.json            "Streamlit: WildGuard AI dashboard" + "Pytest: full suite"
│   ├── tasks.json             install deps / run dashboard / run tests
│   └── extensions.json
├── .github/workflows/ci.yml   ruff + pytest on every push and PR
├── data/                      sample CSVs generated on first run (git-ignored)
├── src/
│   ├── data_loader.py         sample-data generation + CSV loading/validation
│   ├── threat_intel.py        IOC enrichment and the risk-scoring model
│   ├── alerts.py              detection rules and the alert engine
│   └── analytics.py           Plotly figures and KPI computation
└── tests/test_pipeline.py     pytest suite covering the whole pipeline
```

### Running it in VS Code

1. Open the folder, accept the recommended extensions.
2. `Ctrl+Shift+P` → *Python: Create Environment* → venv → `requirements.txt`.
3. Press `F5` and pick **Streamlit: WildGuard AI dashboard** (or run the *Run dashboard* task).

## Tests

```bash
pytest -q
```

## Risk model

```
risk = 45·(ioc_confidence/100) + 20·((avg_severity-1)/4) + 10·blocked_ratio
     + 15·norm(event_count) + 10·norm(distinct_targets)
```

Bands: Critical ≥ 80, High ≥ 60, Medium ≥ 35, otherwise Low.

## Resume bullets

* Built a Streamlit threat-intelligence dashboard that correlates 4k+ security events against an IOC feed and ranks source IPs with an explainable 0–100 risk model.
* Implemented a detection engine of 6 rule-based and statistical (z-score) detections mapped to MITRE ATT&CK, with live-tunable thresholds and CSV export.
* Delivered interactive Plotly analytics (timeline, heatmap, geo, risk scatter) with validated CSV ingestion and a pytest suite covering the ingestion → enrichment → detection pipeline.

Sample data is synthetic and generated locally, so the dashboard is safe to demo anywhere.
