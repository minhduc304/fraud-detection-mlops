"""Daily retrain DAG: data quality check → feature build → train → evaluate/register → notify.

Publishes Airflow Dataset: s3://fraudstream-lake/features/training/
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DATASET_URI = "s3://fraudstream-lake/features/training/"


def check_data_partition(base_path: str, partition: str, min_rows: int = 1000) -> None:
    """Assert yesterday's partition dir exists in base_path and has >= min_rows rows."""
    partition_dir = Path(base_path) / partition
    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition not found: {partition_dir}")

    row_count = 0
    for csv_file in partition_dir.glob("*.csv"):
        with open(csv_file) as f:
            row_count += sum(1 for _ in f) - 1  # subtract header

    if row_count < min_rows:
        raise ValueError(f"Partition {partition} row count {row_count} < {min_rows}")

    log.info("Data check passed: partition=%s rows=%d", partition, row_count)


def build_features_task() -> None:
    from fraudstream.features.transforms import build_features
    import pandas as pd
    from pathlib import Path

    raw_path = Path("data/raw/paysim.csv")
    out_path = Path("data/processed/features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    features = build_features(df)
    features.to_parquet(out_path, index=False)
    log.info("Features built: %d rows → %s", len(features), out_path)


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
                "base_path": "s3://fraudstream-lake/raw/",
                "partition": yesterday,
                "min_rows": 1000,
            },
            outlets=[_dataset],
        )

        feature_build = PythonOperator(
            task_id="feature_build",
            python_callable=build_features_task,
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
            env_vars={"MLFLOW_TRACKING_URI": "http://mlflow.fraudstream.svc.cluster.local:5000"},
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
            env_vars={"MLFLOW_TRACKING_URI": "http://mlflow.fraudstream.svc.cluster.local:5000"},
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
