"""
train_model.py
---------------
Trains an Isolation Forest on a CSV of baseline behavioral feature vectors
(collected via `agent/collector.py --record baseline.csv` while the
legitimate user works normally for ~15-30 minutes).

The model learns what "normal" looks like; at inference time it scores
how anomalous a new feature vector is relative to that baseline, which
we convert into a 0-100 Trust Score.

Usage:
    python train_model.py --data ../../data/baseline.csv --out ../../models/trust_model.joblib
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append("../agent")
from features import FEATURE_NAMES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the SentinelKey AI trust model")
    parser.add_argument("--data", type=str, required=True, help="Path to baseline CSV")
    parser.add_argument("--out", type=str, required=True, help="Path to save trained model (.joblib)")
    parser.add_argument("--contamination", type=float, default=0.05,
                         help="Expected fraction of outliers in baseline data (default 0.05)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Baseline CSV is missing expected columns: {missing}")

    X = df[FEATURE_NAMES].values

    if len(X) < 30:
        print(f"[warn] Only {len(X)} samples found. For a reliable baseline, "
              f"aim for 200+ windows (~15-30 minutes of normal usage at a 5s window).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=args.contamination,
        random_state=42,
    )
    model.fit(X_scaled)

    # Bundle scaler + model together so inference stays consistent
    bundle = {"scaler": scaler, "model": model, "feature_names": FEATURE_NAMES}
    joblib.dump(bundle, args.out)

    scores = model.decision_function(X_scaled)
    print(f"[SentinelKey] Trained on {len(X)} samples.")
    print(f"[SentinelKey] Baseline anomaly score range: "
          f"min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}")
    print(f"[SentinelKey] Model saved to: {args.out}")


if __name__ == "__main__":
    main()
