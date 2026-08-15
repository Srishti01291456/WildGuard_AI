"""Risk-weighted geospatial clustering of detections (poaching hotspots)."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from wildguard.config import RANDOM_SEED, risk_band


def cluster_hotspots(scored: pd.DataFrame, n_clusters: int = 5, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """KMeans over detection coordinates, weighted by predicted risk.

    Returns one row per hotspot with its centroid, size and dominant signal.
    """
    columns = ["hotspot", "latitude", "longitude", "events", "mean_risk",
               "dominant_signal", "top_zone", "risk_band"]
    if scored.empty or scored[["latitude", "longitude"]].dropna().empty:
        return pd.DataFrame(columns=columns)

    df = scored.dropna(subset=["latitude", "longitude"]).copy()
    k = int(max(1, min(n_clusters, df["latitude"].round(3).nunique(), len(df))))
    weights = df["risk_score"].clip(lower=1) if "risk_score" in df.columns else None

    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    df["hotspot"] = model.fit_predict(df[["latitude", "longitude"]], sample_weight=weights)

    grouped = df.groupby("hotspot")
    hotspots = pd.DataFrame(
        {
            "latitude": grouped["latitude"].mean().round(5),
            "longitude": grouped["longitude"].mean().round(5),
            "events": grouped.size(),
            "mean_risk": grouped["risk_score"].mean().round(1) if "risk_score" in df.columns else 0.0,
            "dominant_signal": grouped["signal"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "-"),
            "top_zone": grouped["zone"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "-"),
        }
    ).reset_index()
    hotspots["risk_band"] = hotspots["mean_risk"].map(lambda s: risk_band(s / 100))
    hotspots["hotspot"] = ["H" + str(i + 1) for i in range(len(hotspots))]
    return hotspots.sort_values("mean_risk", ascending=False).reset_index(drop=True)[columns]
