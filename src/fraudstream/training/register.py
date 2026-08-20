import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.exceptions


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
        print(f"Promoted version {mv.version} to @staging  (pr_auc={pr_auc:.4f})")
    else:
        print(f"Not promoted. pr_auc={pr_auc:.4f}")

    return should_promote
