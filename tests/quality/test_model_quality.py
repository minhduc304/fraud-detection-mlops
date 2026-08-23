"""Model quality gate: train on fixture, assert PR-AUC >= 95% of baseline."""
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_CSV = FIXTURES / "quality_sample.csv"
BASELINE_JSON = FIXTURES / "baseline_metrics.json"

REGRESSION_THRESHOLD = 0.95


@pytest.fixture(scope="module")
def fixture_data() -> tuple[pd.DataFrame, pd.Series]:
    assert SAMPLE_CSV.exists(), f"Fixture not found: {SAMPLE_CSV}"
    df = pd.read_csv(SAMPLE_CSV)
    return df.drop(columns=["isFraud"]), df["isFraud"]


@pytest.fixture(scope="module")
def baseline() -> dict[str, float]:
    assert BASELINE_JSON.exists(), f"Baseline not found: {BASELINE_JSON}"
    return json.loads(BASELINE_JSON.read_text())


def test_model_quality_gate(
    fixture_data: tuple[pd.DataFrame, pd.Series],
    baseline: dict[str, float],
) -> None:
    from fraudstream.features.transforms import build_features
    from xgboost import XGBClassifier

    X_raw, y = fixture_data
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

    min_pr_auc = baseline["pr_auc"] * REGRESSION_THRESHOLD
    assert pr_auc >= min_pr_auc, (
        f"Quality gate failed: pr_auc={pr_auc:.4f} < {min_pr_auc:.4f} "
        f"(baseline={baseline['pr_auc']:.4f}, threshold={REGRESSION_THRESHOLD})"
    )
