"""Repoint fraudstream-classifier's staging/champion/production aliases to a given version.

Recovery tool for when the aliased version's artifacts are lost (e.g. an mlruns wipe) and
`serving` can't load a model. Also resets the champion `pr_auc` tag to the target version's,
so `promote_if_better` isn't blocked by a ghost score.

Run via:
    MLFLOW_TRACKING_URI=http://localhost:30500 \
        uv run python scripts/repoint_production_alias.py <version>
"""
import sys

import mlflow

MODEL_NAME = "fraudstream-classifier"
ALIASES = ("staging", "champion", "production")


def main() -> None:
    version = sys.argv[1]
    client = mlflow.MlflowClient()

    mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    print(f"v{version} loads OK")

    mv = client.get_model_version(MODEL_NAME, version)
    pr_auc = mv.tags.get("pr_auc")

    for alias in ALIASES:
        client.set_registered_model_alias(MODEL_NAME, alias, version)
    if pr_auc is not None:
        client.set_model_version_tag(MODEL_NAME, version, "pr_auc", pr_auc)

    for alias in ALIASES:
        resolved = client.get_model_version_by_alias(MODEL_NAME, alias)
        print(f"@{alias} -> v{resolved.version}")
    print(f"champion pr_auc tag = {pr_auc}")


if __name__ == "__main__":
    main()
