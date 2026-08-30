"""Daily retrain DAG: data quality check → feature build → train → evaluate/register → notify.

Publishes Airflow Dataset: s3://fraudstream-lake/features/training/
"""
import io
import logging
import os
from typing import Any

import boto3
import pandas as pd

log = logging.getLogger(__name__)

DATASET_URI = "s3://fraudstream-lake/features/training/"
BUCKET = "fraudstream-lake"


def check_data_partition(bucket: str, partition: str, min_rows: int = 1000, s3: Any = None) -> None:
    """Assert dt=<partition> has >= min_rows rows archived under raw/transactions/ in S3."""
    s3 = s3 or boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://minio:9000"),
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )
    prefix = f"raw/transactions/dt={partition}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])
    if not contents:
        raise FileNotFoundError(f"Partition not found: s3://{bucket}/{prefix}")

    row_count = 0
    for obj in contents:
        body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
        row_count += len(pd.read_parquet(io.BytesIO(body)))

    if row_count < min_rows:
        raise ValueError(f"Partition {partition} row count {row_count} < {min_rows}")

    log.info("Data check passed: partition=%s rows=%d", partition, row_count)


def notify_task(**context: object) -> None:
    ti = context.get("ti")
    log.info("Retrain complete. Task instance: %s", ti)


try:
    from datetime import datetime

    from kubernetes.client import models as k8s

    from airflow import DAG, Dataset
    from airflow.operators.python import PythonOperator
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

    _dataset = Dataset(DATASET_URI)

    _training_volumes = [
        k8s.V1Volume(
            name="data",
            host_path=k8s.V1HostPathVolumeSource(path="/mnt/data", type="Directory"),
        ),
        k8s.V1Volume(
            name="models",
            host_path=k8s.V1HostPathVolumeSource(path="/mnt/models", type="Directory"),
        ),
    ]
    _training_volume_mounts = [
        k8s.V1VolumeMount(name="data", mount_path="/app/data"),
        k8s.V1VolumeMount(name="models", mount_path="/app/models"),
    ]

    with DAG(
        dag_id="retrain_dag",
        schedule="@daily",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["fraudstream", "training"],
    ) as dag:
        yesterday = "{{ ds }}"

        data_check = PythonOperator(
            task_id="data_quality_check",
            python_callable=check_data_partition,
            op_kwargs={
                "bucket": BUCKET,
                "partition": yesterday,
                "min_rows": 1000,
            },
            outlets=[_dataset],
        )

        feature_build = KubernetesPodOperator(
            task_id="feature_build",
            namespace="fraudstream",
            name="retrain-feature-build",
            image="fraudstream-training:local",
            image_pull_policy="Never",
            cmds=["python", "-m", "fraudstream.training.build_features_retrain"],
            service_account_name="airflow",
            volumes=_training_volumes,
            volume_mounts=_training_volume_mounts,
            is_delete_operator_pod=True,
            get_logs=True,
        )

        train = KubernetesPodOperator(
            task_id="train",
            namespace="fraudstream",
            name="retrain-train",
            image="fraudstream-training:local",
            image_pull_policy="Never",
            cmds=["python", "-m", "fraudstream.training.train"],
            service_account_name="airflow",
            volumes=_training_volumes,
            volume_mounts=_training_volume_mounts,
            env_vars={
                "MLFLOW_TRACKING_URI": "http://mlflow.fraudstream.svc.cluster.local:5000",
                "MINIO_ENDPOINT": "http://host.docker.internal:9000",
            },
            is_delete_operator_pod=True,
            get_logs=True,
        )

        evaluate = KubernetesPodOperator(
            task_id="evaluate_and_register",
            namespace="fraudstream",
            name="retrain-evaluate",
            image="fraudstream-training:local",
            image_pull_policy="Never",
            cmds=["python", "-m", "fraudstream.training.evaluate"],
            service_account_name="airflow",
            volumes=_training_volumes,
            volume_mounts=_training_volume_mounts,
            env_vars={
                "MLFLOW_TRACKING_URI": "http://mlflow.fraudstream.svc.cluster.local:5000",
                "MINIO_ENDPOINT": "http://host.docker.internal:9000",
            },
            is_delete_operator_pod=True,
            get_logs=True,
        )

        notify = PythonOperator(
            task_id="notify",
            python_callable=notify_task,
            provide_context=True,
        )

        data_check >> feature_build >> train >> evaluate >> notify

except (ModuleNotFoundError, ImportError):
    # Airflow not installed — DAG object not created; task callables still importable for tests.
    pass
