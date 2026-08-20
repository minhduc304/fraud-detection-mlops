"""Stage 3: train LR + XGBoost from processed features, save models."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraudstream.training.register import save_model

PARAMS_PATH = Path("params.yaml")
IN_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())

    X_train = pd.read_parquet(IN_DIR / "X_train.parquet")
    y_train = pd.read_parquet(IN_DIR / "y_train.parquet")["isFraud"]

    scale = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Train: {len(X_train):,} rows  scale_pos_weight: {scale}")

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
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
