"""Unit tests for drift_check_dag's task callable (mocked S3, no Airflow runtime needed)."""
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from airflow.dags.drift_check_dag import drift_check_task

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "type_CASH_OUT",
    "type_TRANSFER",
]


def _reference_features(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS[:-2]}
    data["type_CASH_OUT"] = rng.integers(0, 2, n)
    data["type_TRANSFER"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def _predictions_df(n: int = 200, shift: float = 0.0) -> pd.DataFrame:
    df = _reference_features(n)
    for col in FEATURE_COLUMNS[:-2]:
        df[col] = df[col] + shift
    df["score"] = np.random.default_rng(1).uniform(0, 1, n)
    df["model_version"] = "1"
    return df


def _mock_s3_with_reference_and_predictions(predictions_df: pd.DataFrame | None) -> MagicMock:
    s3 = MagicMock()
    reference_scores = np.linspace(0, 1, 200)
    reference_features = _reference_features()

    def get_object(Bucket: str, Key: str) -> dict:
        buf = io.BytesIO()
        if Key == "reference/schema_v1/score_reference.npy":
            np.save(buf, reference_scores)
        elif Key == "reference/schema_v1/features_reference.parquet":
            reference_features.to_parquet(buf, index=False)
        elif Key == "predictions/dt=2026-08-24/part-0.parquet":
            predictions_df.to_parquet(buf, index=False)
        else:
            raise KeyError(Key)
        buf.seek(0)
        return {"Body": MagicMock(read=lambda: buf.getvalue())}

    s3.get_object.side_effect = get_object

    if predictions_df is None:
        s3.list_objects_v2.return_value = {}
    else:
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "predictions/dt=2026-08-24/part-0.parquet"}]
        }
    return s3


@patch("airflow.dags.drift_check_dag.push_drift_metrics")
def test_drift_check_skips_when_no_predictions(mock_push: MagicMock) -> None:
    s3 = _mock_s3_with_reference_and_predictions(None)
    drift_check_task(s3=s3, ds="2026-08-24")
    mock_push.assert_not_called()


@patch("airflow.dags.drift_check_dag.push_drift_metrics")
def test_drift_check_pushes_metrics_when_predictions_present(mock_push: MagicMock) -> None:
    predictions = _predictions_df(shift=0.0)
    s3 = _mock_s3_with_reference_and_predictions(predictions)

    drift_check_task(s3=s3, ds="2026-08-24")

    mock_push.assert_called_once()
    results = mock_push.call_args.args[0]
    assert "score" in results
    for col in FEATURE_COLUMNS:
        assert col in results


@patch("airflow.dags.drift_check_dag.push_drift_metrics")
def test_drift_check_score_psi_reflects_real_computation(mock_push: MagicMock) -> None:
    predictions = _predictions_df(shift=0.0)
    predictions["score"] = np.linspace(0, 1, 200)
    s3 = _mock_s3_with_reference_and_predictions(predictions)

    drift_check_task(s3=s3, ds="2026-08-24")

    results = mock_push.call_args.args[0]
    assert results["score"]["psi"] == pytest.approx(0.0, abs=1e-2)
