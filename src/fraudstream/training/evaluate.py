import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


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
