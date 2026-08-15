"""Train and persist the WildGuard AI models from the command line.

Usage::

    python scripts/train_models.py             # sample dataset, cached models
    python scripts/train_models.py --force     # always retrain
    python scripts/train_models.py --events my_events.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildguard.data.loader import bootstrap_dataset, load_events  # noqa: E402
from wildguard.ml.hotspots import cluster_hotspots  # noqa: E402
from wildguard.ml.registry import load_or_train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the WildGuard AI models")
    parser.add_argument("--events", type=Path, help="CSV event log (defaults to the sample dataset)")
    parser.add_argument("--force", action="store_true", help="retrain even if a cached model matches")
    args = parser.parse_args()

    if args.events:
        events = load_events(str(args.events))
    else:
        events, _ = bootstrap_dataset()
    print(f"Loaded {len(events):,} detections spanning {events['timestamp'].min()} → {events['timestamp'].max()}")

    risk, anomaly, trained = load_or_train(events, force_retrain=args.force)
    print("Trained new models" if trained else "Loaded cached models from models/")

    if risk.metrics is not None:
        print("\nHoldout metrics")
        for name, value in risk.metrics.as_dict().items():
            print(f"  {name:<14} {value}")

    print("\nTop features")
    for row in risk.importances.head(6).itertuples():
        print(f"  {row.label:<28} {row.share:.1%}")

    scored = risk.score_events(events)
    print(f"\nMean predicted risk: {scored['risk_score'].mean():.1f}/100")
    print(f"High/Critical detections: {(scored['risk_score'] >= 50).sum():,}")

    anomaly_profiles = anomaly.score(events)
    print(f"Anomalous zone-days flagged: {int(anomaly_profiles['is_anomaly'].sum())}")

    hotspots = cluster_hotspots(scored)
    print("\nHotspots")
    for row in hotspots.itertuples():
        print(f"  {row.hotspot} {row.top_zone:<20} events={row.events:<5} mean risk={row.mean_risk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
