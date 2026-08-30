"""Hourly drift check DAG: load predictions → compute PSI/KS/chi2 → push to Pushgateway.

Triggered by Dataset from retrain_dag: s3://fraudstream-lake/features/training/
Sustained PSI breach fires an Alertmanager alert (Phase 8, prometheus/alert_rules.yml).
"""
import io
import logging
import os
from typing import Any

import boto3
import numpy as np
import pandas as pd

from fraudstream.features.schema import FEATURE_COLUMNS
from fraudstream.monitoring.drift import PSI_THRESHOLD, compute_feature_drift, compute_psi
from fraudstream.monitoring.exporters import push_drift_metrics

log = logging.getLogger(__name__)

BUCKET = "fraudstream-lake"
REFERENCE_PREFIX = "reference/schema_v1/"
DATASET_URI = "s3://fraudstream-lake/features/training/"
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://pushgateway:9091")


def log_psi_result(psi: float, threshold: float = PSI_THRESHOLD) -> None:
    if psi > threshold:
        log.warning("Score drift detected: PSI=%.4f > threshold=%.4f", psi, threshold)
    else:
        log.info("Score stable: PSI=%.4f (threshold=%.4f)", psi, threshold)


def _load_reference(s3: Any) -> tuple[np.ndarray, pd.DataFrame]:
    obj = s3.get_object(Bucket=BUCKET, Key=f"{REFERENCE_PREFIX}score_reference.npy")
    reference_scores = np.load(io.BytesIO(obj["Body"].read()))

    obj = s3.get_object(Bucket=BUCKET, Key=f"{REFERENCE_PREFIX}features_reference.parquet")
    reference_features = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    return reference_scores, reference_features


def _load_predictions(s3: Any, ds: str) -> pd.DataFrame | None:
    prefix = f"predictions/dt={ds}/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    if not keys:
        return None

    frames = []
    for key in keys:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        frames.append(pd.read_parquet(io.BytesIO(obj["Body"].read())))
    return pd.concat(frames, ignore_index=True)


def drift_check_task(s3: Any = None, **context: Any) -> None:
    """Load last hour of predictions, compute PSI/KS/chi2 vs reference, push to Pushgateway."""
    ds = context["ds"]
    s3 = s3 or boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

    current_df = _load_predictions(s3, ds)
    if current_df is None or current_df.empty:
        log.info("Drift check skipped: no predictions for dt=%s", ds)
        return

    reference_scores, reference_features = _load_reference(s3)

    score_psi = compute_psi(reference_scores, current_df["score"].to_numpy())
    log_psi_result(score_psi)

    results = compute_feature_drift(reference_features, current_df[FEATURE_COLUMNS])
    results["score"] = {"psi": score_psi, "breached": score_psi > PSI_THRESHOLD}

    push_drift_metrics(results, pushgateway_url=PUSHGATEWAY_URL)


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
