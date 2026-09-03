"""Unit tests for retrain DAG task callables (no Airflow runtime needed)."""
import io
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _parquet_body(n_rows: int) -> bytes:
    df = pd.DataFrame({"a": range(n_rows), "b": range(n_rows)})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_check_data_partition_exists() -> None:
    from airflow.dags.retrain_dag import check_data_partition

    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"Contents": [{"Key": "raw/transactions/dt=2024-01-01/hour=00/part-1.parquet"}]}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: _parquet_body(1001))}
    check_data_partition("fraudstream-lake", "2024-01-01", min_rows=1000, s3=s3)


def test_check_data_partition_missing_raises() -> None:
    from airflow.dags.retrain_dag import check_data_partition

    s3 = MagicMock()
    s3.list_objects_v2.return_value = {}
    with pytest.raises(FileNotFoundError):
        check_data_partition("fraudstream-lake", "2024-01-01", min_rows=1000, s3=s3)


def test_check_data_partition_insufficient_rows_raises() -> None:
    from airflow.dags.retrain_dag import check_data_partition

    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"Contents": [{"Key": "raw/transactions/dt=2024-01-01/hour=00/part-1.parquet"}]}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: _parquet_body(2))}
    with pytest.raises(ValueError, match="row count"):
        check_data_partition("fraudstream-lake", "2024-01-01", min_rows=1000, s3=s3)


def test_guaranteed_resources_request_equals_limit() -> None:
    from airflow.dags.retrain_dag import guaranteed_resources

    res = guaranteed_resources("2", "3Gi")
    assert res["requests"] == res["limits"] == {"cpu": "2", "memory": "3Gi"}


def test_training_pod_resources_are_set() -> None:
    from airflow.dags.retrain_dag import (
        EVALUATE_POD_CPU,
        EVALUATE_POD_MEMORY,
        FEATURE_BUILD_POD_CPU,
        FEATURE_BUILD_POD_MEMORY,
        TRAIN_POD_CPU,
        TRAIN_POD_MEMORY,
    )

    assert TRAIN_POD_CPU and TRAIN_POD_MEMORY and EVALUATE_POD_CPU and EVALUATE_POD_MEMORY
    assert FEATURE_BUILD_POD_CPU and FEATURE_BUILD_POD_MEMORY


def test_compute_psi_identical_distributions() -> None:
    from airflow.dags.drift_check_dag import compute_psi

    import numpy as np

    scores = np.linspace(0, 1, 100)
    psi = compute_psi(scores, scores)
    assert psi < 0.01


def test_compute_psi_different_distributions() -> None:
    from airflow.dags.drift_check_dag import compute_psi

    import numpy as np

    reference = np.array([0.1] * 100)
    current = np.array([0.9] * 100)
    psi = compute_psi(reference, current)
    assert psi > 0.2


def test_compute_psi_threshold_logged(caplog: pytest.LogCaptureFixture) -> None:
    from airflow.dags.drift_check_dag import compute_psi, log_psi_result

    import logging
    import numpy as np

    reference = np.array([0.1] * 100)
    current = np.array([0.9] * 100)
    psi = compute_psi(reference, current)
    with caplog.at_level(logging.WARNING):
        log_psi_result(psi, threshold=0.2)
    assert "WARNING" in caplog.text or any(r.levelname == "WARNING" for r in caplog.records)
