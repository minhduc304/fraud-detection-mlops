"""Stage 3: train LR + XGBoost from processed features, save models + start MLflow run."""
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraudstream.features.schema import SCHEMA_VERSION
from fraudstream.training.register import save_model

PARAMS_PATH = Path("params.yaml")
DVC_FILE = Path("data/raw/paysim.csv.dvc")
IN_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def _data_hash() -> str:
    info = yaml.safe_load(DVC_FILE.read_text())
    return str(info["outs"][0].get("md5", "unknown"))


def _git_sha() -> str:
    env_sha = os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())

    X_train = pd.read_parquet(IN_DIR / "X_train.parquet")
    y_train = pd.read_parquet(IN_DIR / "y_train.parquet")["isFraud"]

    scale = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Train: {len(X_train):,} rows  scale_pos_weight: {scale}")

    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(
            {
                "split.ratio": params["split"]["ratio"],
                "xgb.n_estimators": params["xgb"]["n_estimators"],
                "xgb.max_depth": params["xgb"]["max_depth"],
                "xgb.learning_rate": params["xgb"]["learning_rate"],
                "scale_pos_weight": scale,
                "data_hash": _data_hash(),
            }
        )
        mlflow.set_tags({"git_sha": _git_sha(), "schema_version": SCHEMA_VERSION})

        print("Training logistic regression...")
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_train, y_train)

        print("Training XGBoost...")
        xgb_params = params["xgb"]
        xgb = XGBClassifier(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            scale_pos_weight=scale,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_model(lr, MODELS_DIR / "lr.pkl")
        save_model(xgb, MODELS_DIR / "xgb.pkl")
        mlflow.sklearn.log_model(lr, "lr_model")
        mlflow.xgboost.log_model(xgb, "xgb_model")

        (MODELS_DIR / "run_id.txt").write_text(run.info.run_id)
        print(f"Models saved. MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
