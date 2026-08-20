from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraudstream.training.register import load_model

IN_DIR = Path("data/processed")
MODELS_DIR = Path("models")


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


def main() -> None:
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


if __name__ == "__main__":
    main()
