"""Stage 4: evaluate models, log metrics + artifacts to existing MLflow run."""
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraudstream.features.schema import FEATURE_COLUMNS
from fraudstream.training.register import load_model, promote_if_better, write_reference_baseline

IN_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SCHEMA_PATH = Path("src/fraudstream/features/schema.py")
PARAMS_PATH = Path("params.yaml")
MODEL_NAME = "fraudstream-classifier"
LR_ARTIFACT = "lr_model"
XGB_ARTIFACT = "xgb_model"


def pr_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    return float(average_precision_score(y_true, y_score))


def precision_at_recall(
    y_true: pd.Series, y_score: pd.Series, min_recall: float = 0.8
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall >= min_recall
    return float(np.max(precision[mask])) if mask.any() else 0.0


def report(name: str, y_true: pd.Series, y_score: pd.Series) -> dict[str, str | float]:
    return {
        "model": name,
        "pr_auc": pr_auc(y_true, y_score),
        "precision_at_recall_80": precision_at_recall(y_true, y_score, min_recall=0.8),
    }


def _log_pr_curve(
    y_test: pd.Series,
    lr_scores: pd.Series,
    xgb_scores: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for scores, label in [(lr_scores, "LR"), (xgb_scores, "XGBoost")]:
        p, r, _ = precision_recall_curve(y_test, scores)
        ax.plot(r, p, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(f.name, artifact_path="plots")
    plt.close(fig)


def main() -> None:
    run_id = (MODELS_DIR / "run_id.txt").read_text().strip()

    X_test = pd.read_parquet(IN_DIR / "X_test.parquet")
    y_test = pd.read_parquet(IN_DIR / "y_test.parquet")["isFraud"]

    lr = load_model(MODELS_DIR / "lr.pkl")
    xgb = load_model(MODELS_DIR / "xgb.pkl")

    lr_scores = pd.Series(lr.predict_proba(X_test)[:, 1])
    xgb_scores = pd.Series(xgb.predict_proba(X_test)[:, 1])

    lr_report = report("logistic_regression", y_test, lr_scores)
    xgb_report = report("xgboost", y_test, xgb_scores)

    print("\n--- Results ---")
    header = f"{'Model':<22} {'PR-AUC':>8} {'P@R80':>8}"
    print(header)
    print("-" * len(header))
    for r in [lr_report, xgb_report]:
        print(f"{r['model']:<22} {r['pr_auc']:>8.4f} {r['precision_at_recall_80']:>8.4f}")

    winner = max([lr_report, xgb_report], key=lambda r: r["pr_auc"])
    print(f"\nWinner: {winner['model']}  PR-AUC: {winner['pr_auc']:.4f}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "pr_auc_lr": float(lr_report["pr_auc"]),
                "pr_auc_xgb": float(xgb_report["pr_auc"]),
                "precision_at_recall_80_lr": float(lr_report["precision_at_recall_80"]),
                "precision_at_recall_80_xgb": float(xgb_report["precision_at_recall_80"]),
            }
        )
        mlflow.log_param("winner", winner["model"])
        mlflow.log_artifact(str(SCHEMA_PATH), artifact_path="schema")
        _log_pr_curve(y_test, lr_scores, xgb_scores)

    params = yaml.safe_load(PARAMS_PATH.read_text())
    run_id = (MODELS_DIR / "run_id.txt").read_text().strip()
    winner_artifact = XGB_ARTIFACT if winner["model"] == "xgboost" else LR_ARTIFACT
    winner_scores = xgb_scores if winner["model"] == "xgboost" else lr_scores
    promoted = promote_if_better(
        run_id=run_id,
        model_name=MODEL_NAME,
        model_artifact_path=winner_artifact,
        pr_auc=float(winner["pr_auc"]),
        min_pr_auc=params["min_pr_auc"],
    )
    if promoted:
        write_reference_baseline(winner_scores, X_test[FEATURE_COLUMNS])


if __name__ == "__main__":
    main()
