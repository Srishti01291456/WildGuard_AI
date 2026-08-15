"""Machine-learning layer: feature engineering, models and the model registry."""

from wildguard.ml.anomaly import ZoneAnomalyDetector
from wildguard.ml.hotspots import cluster_hotspots
from wildguard.ml.registry import load_or_train
from wildguard.ml.risk_model import ModelMetrics, PoachingRiskModel

__all__ = [
    "ZoneAnomalyDetector",
    "cluster_hotspots",
    "load_or_train",
    "ModelMetrics",
    "PoachingRiskModel",
]
