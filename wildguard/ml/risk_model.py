"""Feature 1 - supervised poaching-risk model.

A gradient-boosted classifier estimates ``P(poaching incident | detection)``.
The wrapper keeps the fitted pipeline, its holdout metrics and its feature
importances together so the dashboard can present a full model card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from wildguard.config import RANDOM_SEED, risk_band
from wildguard.ml.features import TARGET, base_feature_of, build_features, build_preprocessor, pretty


@dataclass
class ModelMetrics:
    """Holdout performance of the risk model."""

    roc_auc: float
    average_precision: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float
    support: int
    positives: int
    confusion: list[list[int]] = field(default_factory=list)
    roc_curve: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        return {
            "ROC AUC": f"{self.roc_auc:.3f}",
            "PR AUC": f"{self.average_precision:.3f}",
            "Accuracy": f"{self.accuracy:.1%}",
            "Precision": f"{self.precision:.1%}",
            "Recall": f"{self.recall:.1%}",
            "F1": f"{self.f1:.3f}",
            "Brier score": f"{self.brier:.3f}",
        }


class PoachingRiskModel:
    """Train / score wrapper around a scikit-learn pipeline."""

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self.pipeline: Pipeline | None = None
        self.metrics: ModelMetrics | None = None
        self.importances: pd.DataFrame = pd.DataFrame(columns=["feature", "importance"])
        self.threshold: float = 0.5
        self.trained_rows: int = 0

    # -- training -------------------------------------------------------------

    def fit(self, events: pd.DataFrame, test_size: float = 0.25) -> PoachingRiskModel:
        if TARGET not in events.columns:
            raise ValueError(
                "Training requires a labelled 'is_poaching_incident' column. "
                "Upload labelled data or use the bundled sample dataset."
            )

        x = build_features(events)
        y = events[TARGET].astype(int)
        stratify = y if y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=self.seed, stratify=stratify
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.08,
                        max_depth=6,
                        l2_regularization=0.5,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(x_train, y_train)
        self.trained_rows = len(x_train)
        self.metrics = self._evaluate(x_test, y_test)
        self.importances = self._importances(x_test, y_test)
        return self

    def _evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
        assert self.pipeline is not None
        proba = self.pipeline.predict_proba(x_test)[:, 1]
        pred = (proba >= self.threshold).astype(int)
        single_class = y_test.nunique() < 2
        fpr, tpr = ([], [])
        if not single_class:
            fpr_arr, tpr_arr, _ = roc_curve(y_test, proba)
            fpr, tpr = list(np.round(fpr_arr, 4)), list(np.round(tpr_arr, 4))
        return ModelMetrics(
            roc_auc=float("nan") if single_class else float(roc_auc_score(y_test, proba)),
            average_precision=float("nan") if single_class else float(average_precision_score(y_test, proba)),
            accuracy=float(accuracy_score(y_test, pred)),
            precision=float(precision_score(y_test, pred, zero_division=0)),
            recall=float(recall_score(y_test, pred, zero_division=0)),
            f1=float(f1_score(y_test, pred, zero_division=0)),
            brier=float(brier_score_loss(y_test, proba)),
            support=int(len(y_test)),
            positives=int(y_test.sum()),
            confusion=confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
            roc_curve={"fpr": fpr, "tpr": tpr},
        )

    def _importances(self, x_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
        """Permutation importance on the holdout set, grouped by source feature."""
        assert self.pipeline is not None
        sample = min(len(x_test), 1500)
        result = permutation_importance(
            self.pipeline,
            x_test.iloc[:sample],
            y_test.iloc[:sample],
            n_repeats=n_repeats,
            random_state=self.seed,
            scoring="roc_auc" if y_test.nunique() > 1 else "accuracy",
        )
        table = (
            pd.DataFrame({"feature": x_test.columns, "importance": result.importances_mean})
            .assign(feature=lambda d: d["feature"].map(base_feature_of))
            .groupby("feature", as_index=False)["importance"]
            .sum()
        )
        table["importance"] = table["importance"].clip(lower=0)
        total = table["importance"].sum()
        table["share"] = table["importance"] / total if total else 0.0
        table["label"] = table["feature"].map(pretty)
        return table.sort_values("importance", ascending=False).reset_index(drop=True)

    # -- inference ------------------------------------------------------------

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline.predict_proba(build_features(events))[:, 1]

    def score_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return the events with ``risk_score`` (0-100) and ``risk_band`` columns."""
        scored = events.copy()
        proba = self.predict_proba(events)
        scored["risk_probability"] = np.round(proba, 4)
        scored["risk_score"] = np.round(proba * 100, 1)
        scored["risk_band"] = [risk_band(p) for p in proba]
        return scored


def zone_threat_profile(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored events into a per-zone threat intelligence table."""
    if scored.empty:
        return pd.DataFrame(
            columns=["zone", "events", "mean_risk", "max_risk", "high_risk_events",
                     "threat_signals", "night_share", "patrol_coverage", "last_event", "risk_band"]
        )

    grouped = scored.groupby("zone")
    profile = pd.DataFrame(
        {
            "events": grouped.size(),
            "mean_risk": grouped["risk_score"].mean().round(1),
            "max_risk": grouped["risk_score"].max().round(1),
            "high_risk_events": grouped["risk_band"].apply(lambda s: int(s.isin(["High", "Critical"]).sum())),
            "threat_signals": grouped["signal_class"].apply(lambda s: int((s == "threat").sum())),
            "night_share": grouped["is_night"].mean().round(2),
            "patrol_coverage": grouped["patrol_coverage"].mean().round(2),
            "last_event": grouped["timestamp"].max(),
        }
    ).reset_index()
    profile["risk_band"] = profile["mean_risk"].map(lambda s: risk_band(s / 100))
    return profile.sort_values("mean_risk", ascending=False).reset_index(drop=True)
