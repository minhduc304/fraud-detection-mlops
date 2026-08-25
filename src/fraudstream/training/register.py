import io
import pickle
from pathlib import Path
from typing import Any

import boto3
import mlflow
import mlflow.exceptions
import numpy as np
import pandas as pd

BUCKET = "fraudstream-lake"


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def promote_if_better(
    run_id: str,
    model_name: str,
    model_artifact_path: str,
    pr_auc: float,
    min_pr_auc: float,
) -> bool:
    """Register model version and promote to @staging alias if it beats current @champion.

    First run (no @champion): promote if pr_auc >= min_pr_auc.
    Subsequent runs: promote if pr_auc > champion's pr_auc tag.
    Returns True if promoted.
    """
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    mv = mlflow.register_model(model_uri, model_name)
    client.set_model_version_tag(model_name, mv.version, "pr_auc", str(pr_auc))

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_pr_auc = float(champion.tags.get("pr_auc", 0.0))
        should_promote = pr_auc > champion_pr_auc
    except mlflow.exceptions.MlflowException:
        should_promote = pr_auc >= min_pr_auc

    if should_promote:
        client.set_registered_model_alias(model_name, "staging", mv.version)
        client.set_registered_model_alias(model_name, "champion", mv.version)
        client.set_registered_model_alias(model_name, "production", mv.version)
        print(f"Promoted v{mv.version} to @staging/@champion/@production (pr_auc={pr_auc:.4f})")
    else:
        print(f"Not promoted. pr_auc={pr_auc:.4f}")

    return should_promote


def write_reference_baseline(
    scores: pd.Series,
    features_df: pd.DataFrame,
    s3: Any = None,
    minio_endpoint: str = "http://localhost:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
) -> None:
    """Write the champion's evaluation-set score/feature distributions as the drift reference."""
    s3 = s3 or boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    score_buf = io.BytesIO()
    np.save(score_buf, scores.to_numpy())
    s3.put_object(
        Bucket=BUCKET, Key="reference/schema_v1/score_reference.npy", Body=score_buf.getvalue()
    )

    features_buf = io.BytesIO()
    features_df.to_parquet(features_buf, index=False)
    s3.put_object(
        Bucket=BUCKET,
        Key="reference/schema_v1/features_reference.parquet",
        Body=features_buf.getvalue(),
    )
