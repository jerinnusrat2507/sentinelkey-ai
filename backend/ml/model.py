"""
model.py
--------
Loads the trained Isolation Forest bundle and converts its raw anomaly
score into a human-readable Trust Score between 0 and 100.

Trust Score interpretation:
    90-100 : Behavior strongly matches the baseline user
    70-89  : Normal, minor variation
    40-69  : Noticeable deviation -- worth flagging
    0-39   : Strong anomaly -- likely a different operator
"""

import os

import joblib
import numpy as np

_DEFAULT_MODEL_PATH = os.environ.get(
    "SENTINELKEY_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "trust_model.joblib"),
)


class TrustModel:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.bundle = None
        self._load()

    def _load(self):
        if os.path.isfile(self.model_path):
            self.bundle = joblib.load(self.model_path)
        else:
            self.bundle = None  # no model trained yet

    def is_ready(self) -> bool:
        return self.bundle is not None

    def score(self, feature_dict: dict) -> float:
        """Returns a trust score in [0, 100]. Returns 50.0 (neutral) if no model is loaded."""
        if not self.is_ready():
            return 50.0

        feature_names = self.bundle["feature_names"]
        x = np.array([[feature_dict[name] for name in feature_names]])
        x_scaled = self.bundle["scaler"].transform(x)

        # decision_function: higher = more normal, lower/negative = more anomalous.
        # Typical range roughly [-0.5, 0.5]; we squash it into 0-100 with a sigmoid-like map.
        raw = self.bundle["model"].decision_function(x_scaled)[0]
        trust = 100 / (1 + np.exp(-10 * raw))  # logistic squashing centered at raw=0
        return float(np.clip(trust, 0, 100))
