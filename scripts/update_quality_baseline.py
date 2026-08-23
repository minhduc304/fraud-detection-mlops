"""Regenerate tests/fixtures/baseline_metrics.json from quality_sample.csv.

Run via: make update-quality-baseline
Run after intentional model improvements or features/schema.py changes.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from fraudstream.features.transforms import build_features

SAMPLE_CSV = Path("tests/fixtures/quality_sample.csv")
BASELINE_JSON = Path("tests/fixtures/baseline_metrics.json")


def main() -> None:
    df = pd.read_csv(SAMPLE_CSV)
    X_raw = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    X = build_features(X_raw)

    scale = int((y == 0).sum() / max((y == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    scores = model.predict_proba(X)[:, 1]
    pr_auc = float(average_precision_score(y, scores))

    baseline = {"pr_auc": pr_auc}
    BASELINE_JSON.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline updated: pr_auc={pr_auc:.4f} → {BASELINE_JSON}")


if __name__ == "__main__":
    main()
