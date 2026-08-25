"""Pure distributional-drift math: PSI, KS, chi-squared. No I/O.

Thresholds (documented, no external tuning source — standard industry bands):
- PSI > 0.2: significant shift (0.1-0.2 moderate, <0.1 stable).
- KS / chi-squared: p-value < 0.05 (standard 5% significance level).
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

CATEGORICAL_COLUMNS = ["type_CASH_OUT", "type_TRANSFER"]

PSI_THRESHOLD = 0.2
KS_P_THRESHOLD = 0.05
CHI2_P_THRESHOLD = 0.05


def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between reference and current continuous distributions."""
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max())
    breakpoints = np.linspace(lo, hi, buckets + 1)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    eps = 1e-6
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * buckets)
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * buckets)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov two-sample test. Returns (statistic, p_value)."""
    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def compute_chi2(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Chi-squared test on a reference-vs-current contingency table of category counts.

    Returns (statistic, p_value).
    """
    categories = np.union1d(np.unique(reference), np.unique(current))
    ref_counts = [int((reference == c).sum()) for c in categories]
    cur_counts = [int((current == c).sum()) for c in categories]
    statistic, p_value, _, _ = chi2_contingency([ref_counts, cur_counts])
    return float(statistic), float(p_value)


def compute_feature_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> dict[str, dict[str, float | bool]]:
    """Per-feature drift: PSI+KS for continuous columns, chi-squared for categorical ones."""
    results: dict[str, dict[str, float | bool]] = {}

    for column in reference_df.columns:
        reference = reference_df[column].to_numpy()
        current = current_df[column].to_numpy()

        if column in CATEGORICAL_COLUMNS:
            statistic, p_value = compute_chi2(reference, current)
            results[column] = {
                "chi2_statistic": statistic,
                "chi2_p_value": p_value,
                "breached": p_value < CHI2_P_THRESHOLD,
            }
        else:
            psi = compute_psi(reference, current)
            ks_statistic, ks_p_value = compute_ks(reference, current)
            results[column] = {
                "psi": psi,
                "ks_statistic": ks_statistic,
                "ks_p_value": ks_p_value,
                "breached": psi > PSI_THRESHOLD or ks_p_value < KS_P_THRESHOLD,
            }

    return results
