# 🐘 WildGuard AI — Wildlife Protection Intelligence

An end-to-end **AI/ML project with a professional Streamlit dashboard**. WildGuard AI ingests
wildlife-reserve telemetry (camera traps, acoustic sensors, GPS collars, drones and ranger
reports), predicts the **probability that a detection belongs to a poaching event**, turns those
predictions into a **prioritised alert queue for rangers**, and explains what is happening in the
reserve through **wildlife event analytics**.

No API keys, no database, no cloud services — `streamlit run app.py` and everything works.

---

## The three headline features

Each feature is a **separate page** in the dashboard and is highlighted on the home page.

| Page | Feature | What it actually does | ML behind it |
| --- | --- | --- | --- |
| 1 | 🧠 **Threat Intelligence** | Scores every detection 0-100, aggregates into zone threat profiles, clusters geospatial poaching hotspots and ships a full model card (ROC, confusion matrix, permutation importances) plus a what-if scorer | `HistGradientBoostingClassifier`, `KMeans` |
| 2 | 🚨 **Smart Alert System** | Fuses model risk, unsupervised anomalies and 9 ranger rules into one ranked, de-duplicated alert queue — every alert carries its evidence and a recommended field action, with live-tunable thresholds | `IsolationForest` + rule fusion |
| 3 | 📊 **Wildlife Event Intelligence** | Species activity, hour-by-zone heatmaps, signal mix, Shannon biodiversity per zone, sensor-fleet health and a risk-weighted detection map, all filterable and exportable | Diversity metrics, aggregation |

## Quickstart

```bash
git clone https://github.com/Srishti01291456/WildGuard_AI.git
cd WildGuard_AI

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py               # http://localhost:8501
```

On first run the app generates ~6,000 synthetic detections into `data/`, trains both models and
caches them in `models/`. Subsequent runs load the cached models instantly.

Train from the command line instead (prints the metrics and hotspots):

```bash
python scripts/train_models.py            # train or reuse cached models
python scripts/train_models.py --force    # always retrain
```

## Project structure

```
WildGuard_AI/
├── app.py                                  Home page: hero, 3 feature cards, KPIs, model card
├── pages/
│   ├── 1_🧠_Threat_Intelligence.py         Feature 1 — risk model, hotspots, model card, what-if
│   ├── 2_🚨_Smart_Alerts.py                Feature 2 — fused alert queue + briefings
│   └── 3_📊_Wildlife_Event_Intelligence.py Feature 3 — species, biodiversity, sensors, map
├── wildguard/                              importable Python package (all the logic)
│   ├── config.py                           paths, domain vocabulary, risk bands
│   ├── data/
│   │   ├── generator.py                    synthetic reserve telemetry + labels
│   │   └── loader.py                       CSV validation, defaults, derived columns
│   ├── ml/
│   │   ├── features.py                     feature engineering + sklearn preprocessor
│   │   ├── risk_model.py                   poaching-risk classifier, metrics, importances
│   │   ├── anomaly.py                      Isolation Forest over zone/day profiles
│   │   ├── hotspots.py                     risk-weighted KMeans hotspots
│   │   └── registry.py                     train / cache / load model artefacts
│   ├── intelligence/
│   │   ├── alerts.py                       9-rule alert engine fused with model output
│   │   ├── analytics.py                    wildlife analytics + Plotly figures
│   │   └── modelviz.py                     ROC, confusion matrix, importance charts
│   └── ui/
│       ├── theme.py                        page config + CSS skin
│       ├── components.py                   hero, feature cards, KPI row, downloads
│       └── state.py                        cached loading, filters, sidebar controls
├── scripts/train_models.py                 CLI trainer
├── tests/                                  pytest suite (data, models, intelligence)
├── data/  models/                          generated CSVs and model artefacts (git-ignored)
├── .streamlit/config.toml                  dark theme
├── .vscode/                                settings, launch/debug, tasks, extensions
└── .github/workflows/ci.yml                ruff + pytest on every push and PR
```

## Running it in VS Code

1. Open the folder and accept the recommended extensions.
2. `Ctrl+Shift+P` → **Python: Create Environment** → `venv` → `requirements.txt`.
3. Press `F5` and choose **Streamlit: WildGuard AI dashboard** (or run the *Run dashboard* task
   with `Ctrl+Shift+B`).
4. Tests appear in the Testing panel automatically (`pytest`).

## The models

**1. Poaching-risk classifier** — `HistGradientBoostingClassifier` on 13 numeric + 5 categorical
features (detector confidence, night flag, cyclical hour, patrol coverage, hours since patrol,
proximity to boundary/road/village, zone, sensor type, signal, weather …). Stratified 75/25 split;
typical holdout performance on the bundled dataset:

| ROC AUC | PR AUC | Accuracy | Precision | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| ≈ 0.82 | ≈ 0.87 | ≈ 76% | ≈ 78% | ≈ 83% | ≈ 0.80 | ≈ 0.17 |

Explainability comes from **permutation importance on the holdout set**, shown in the UI.

**2. Zone anomaly detector** — `IsolationForest` over daily per-zone behaviour profiles
(event volume, threat ratio, night ratio, mean confidence, distinct signals, patrol coverage),
surfaced as a 0-100 anomaly score that feeds the alert engine.

**3. Hotspot clustering** — `KMeans` over detection coordinates weighted by predicted risk, giving
patrol-ready hotspot centroids.

Risk bands: **Critical ≥ 75**, **High ≥ 50**, **Medium ≥ 25**, otherwise **Low**.

## Using your own data

Upload a CSV from the sidebar on any page.

* **Required columns:** `timestamp`, `zone`, `signal`
* **Optional:** `sensor_id`, `sensor_type`, `confidence`, `detected_count`, `latitude`,
  `longitude`, `distance_to_boundary_km`, `distance_to_road_km`, `distance_to_village_km`,
  `patrol_coverage`, `hours_since_patrol`, `weather`, `temperature_c`, `is_night`
* **Optional label:** `is_poaching_incident` (0/1) — present ⇒ the models retrain on your data,
  absent ⇒ the cached model scores it.

Missing optional columns are filled with safe defaults, bad timestamps are dropped, and missing
coordinates fall back to the zone centroid.

## Tests & linting

```bash
pytest -q          # 24 tests: data validation, model quality, alerts, analytics
ruff check .
```

The model test asserts the classifier beats random by a wide margin (`ROC AUC > 0.7`), so a broken
feature pipeline fails CI rather than silently degrading the dashboard.

## Resume bullets

* Built an end-to-end ML platform (scikit-learn + Streamlit) that scores 6k+ wildlife-sensor
  detections for poaching risk with a gradient-boosted classifier (ROC AUC ≈ 0.82) and explains
  predictions with permutation importance and a full model card.
* Fused supervised risk scores, Isolation Forest anomaly detection and 9 domain rules into a
  ranked alert queue with recommended ranger actions and live-tunable thresholds.
* Shipped a 4-page analytics dashboard (species activity, biodiversity indices, sensor health,
  KMeans poaching hotspots) with validated CSV ingestion, model caching and a 24-test pytest suite
  running in GitHub Actions.

All bundled data is synthetic and generated locally, so the project is safe to demo anywhere.
