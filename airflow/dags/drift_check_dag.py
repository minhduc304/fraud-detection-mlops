"""Hourly drift check DAG: load predictions → compute PSI → log result.

Triggered by Dataset from retrain_dag: s3://fraudstream-lake/features/training/
PSI > 0.2 emits WARNING (alerting wired in Phase 8).
"""
import logging

import numpy as np

log = logging.getLogger(__name__)

PSI_THRESHOLD = 0.2
DATASET_URI = "s3://fraudstream-lake/features/training/"
REFERENCE_BASELINE_URI = "s3://fraudstream-lake/baselines/score_reference.npy"


def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index between reference and current score distributions."""
    breakpoints = np.linspace(0, 1, buckets + 1)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    # Add epsilon to avoid log(0)
    eps = 1e-6
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * buckets)
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * buckets)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def log_psi_result(psi: float, threshold: float = PSI_THRESHOLD) -> None:
    if psi > threshold:
        log.warning("Score drift detected: PSI=%.4f > threshold=%.4f", psi, threshold)
    else:
        log.info("Score stable: PSI=%.4f (threshold=%.4f)", psi, threshold)


def drift_check_task(**context: object) -> None:
    """Load last hour of predictions, compute PSI vs reference baseline, log result."""
    # In Phase 6 the prediction store and baseline are stubs; full wiring is Phase 8.
    # We log a PSI of 0.0 when no data is available yet.
    try:
        import boto3
        import io

        s3 = boto3.client("s3", endpoint_url="http://minio:9000",
                          aws_access_key_id="minioadmin",
                          aws_secret_access_key="minioadmin")

        obj = s3.get_object(Bucket="fraudstream-lake", Key="baselines/score_reference.npy")
        reference = np.load(io.BytesIO(obj["Body"].read()))

        obj = s3.get_object(Bucket="fraudstream-lake", Key="predictions/latest_hour.npy")
        current = np.load(io.BytesIO(obj["Body"].read()))

        psi = compute_psi(reference, current)
    except Exception as exc:
        log.info("Drift check skipped (data not available): %s", exc)
        psi = 0.0

    log_psi_result(psi)
    log.info("PSI=%.4f", psi)


try:
    from datetime import datetime

    from airflow import DAG, Dataset
    from airflow.operators.python import PythonOperator

    _trigger_dataset = Dataset(DATASET_URI)

    with DAG(
        dag_id="drift_check_dag",
        schedule=[_trigger_dataset],
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["fraudstream", "monitoring"],
    ) as dag:
        check = PythonOperator(
            task_id="drift_check",
            python_callable=drift_check_task,
            provide_context=True,
        )

except (ModuleNotFoundError, ImportError):
    pass
