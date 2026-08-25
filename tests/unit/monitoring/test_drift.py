import numpy as np
import pandas as pd

from fraudstream.monitoring.drift import (
    compute_chi2,
    compute_feature_drift,
    compute_ks,
    compute_psi,
)

RNG = np.random.default_rng(42)


def test_psi_identical_distributions_no_breach() -> None:
    reference = RNG.normal(0, 1, 1000)
    current = RNG.normal(0, 1, 1000)
    psi = compute_psi(reference, current)
    assert psi < 0.2


def test_psi_shifted_distribution_breaches() -> None:
    reference = RNG.normal(0, 1, 1000)
    current = reference + 5
    psi = compute_psi(reference, current)
    assert psi > 0.2


def test_ks_identical_distributions_no_breach() -> None:
    reference = RNG.normal(0, 1, 1000)
    current = RNG.normal(0, 1, 1000)
    statistic, p_value = compute_ks(reference, current)
    assert p_value > 0.05


def test_ks_shifted_distribution_breaches() -> None:
    reference = RNG.normal(0, 1, 1000)
    current = reference + 5
    statistic, p_value = compute_ks(reference, current)
    assert p_value < 0.05


def test_chi2_identical_distributions_no_breach() -> None:
    reference = RNG.integers(0, 2, 1000)
    current = RNG.integers(0, 2, 1000)
    statistic, p_value = compute_chi2(reference, current)
    assert p_value > 0.05


def test_chi2_shifted_distribution_breaches() -> None:
    reference = RNG.choice([0, 1], size=1000, p=[0.9, 0.1])
    current = RNG.choice([0, 1], size=1000, p=[0.1, 0.9])
    statistic, p_value = compute_chi2(reference, current)
    assert p_value < 0.05


def _feature_frames(shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            "amount": RNG.normal(0, 1, 1000),
            "oldbalanceOrg": RNG.normal(0, 1, 1000),
            "type_CASH_OUT": RNG.integers(0, 2, 1000),
        }
    )
    current = pd.DataFrame(
        {
            "amount": reference["amount"] + shift,
            "oldbalanceOrg": reference["oldbalanceOrg"] + shift,
            "type_CASH_OUT": RNG.integers(0, 2, 1000),
        }
    )
    return reference, current


def test_feature_drift_no_shift_no_breach() -> None:
    reference, current = _feature_frames(shift=0.0)
    result = compute_feature_drift(reference, current)
    assert set(result) == {"amount", "oldbalanceOrg", "type_CASH_OUT"}
    for feature, metrics in result.items():
        assert metrics["breached"] is False


def test_feature_drift_shift_breaches_continuous_features() -> None:
    reference, current = _feature_frames(shift=5.0)
    result = compute_feature_drift(reference, current)
    assert result["amount"]["breached"] is True
    assert result["oldbalanceOrg"]["breached"] is True


def test_feature_drift_continuous_has_psi_and_ks() -> None:
    reference, current = _feature_frames(shift=0.0)
    result = compute_feature_drift(reference, current)
    assert "psi" in result["amount"]
    assert "ks_statistic" in result["amount"]
    assert "ks_p_value" in result["amount"]


def test_feature_drift_categorical_has_chi2() -> None:
    reference, current = _feature_frames(shift=0.0)
    result = compute_feature_drift(reference, current)
    assert "chi2_statistic" in result["type_CASH_OUT"]
    assert "chi2_p_value" in result["type_CASH_OUT"]
    assert "psi" not in result["type_CASH_OUT"]
