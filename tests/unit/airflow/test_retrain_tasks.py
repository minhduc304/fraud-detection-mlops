"""Unit tests for retrain DAG task callables (no Airflow runtime needed)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_check_data_partition_exists(tmp_path: Path) -> None:
    from airflow.dags.retrain_dag import check_data_partition

    # Should not raise when partition dir exists with enough rows
    partition_dir = tmp_path / "2024-01-01"
    partition_dir.mkdir()
    (partition_dir / "data.csv").write_text(
        "a,b\n" + "\n".join(f"{i},{i}" for i in range(1001))
    )
    check_data_partition(str(tmp_path), "2024-01-01", min_rows=1000)


def test_check_data_partition_missing_raises(tmp_path: Path) -> None:
    from airflow.dags.retrain_dag import check_data_partition

    with pytest.raises(FileNotFoundError):
        check_data_partition(str(tmp_path), "2024-01-01", min_rows=1000)


def test_check_data_partition_insufficient_rows_raises(tmp_path: Path) -> None:
    from airflow.dags.retrain_dag import check_data_partition

    partition_dir = tmp_path / "2024-01-01"
    partition_dir.mkdir()
    (partition_dir / "data.csv").write_text("a,b\n1,2\n3,4")
    with pytest.raises(ValueError, match="row count"):
        check_data_partition(str(tmp_path), "2024-01-01", min_rows=1000)


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
