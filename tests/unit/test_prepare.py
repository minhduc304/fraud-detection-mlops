import pandas as pd
import pytest


def _split_by_step(df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror of prepare.py split logic for offline testing."""
    split_step = df["step"].quantile(ratio)
    train = df[df["step"] <= split_step].reset_index(drop=True)
    test = df[df["step"] > split_step].reset_index(drop=True)
    return train, test


@pytest.fixture()
def toy_df() -> pd.DataFrame:
    return pd.DataFrame({"step": list(range(1, 11)), "val": list(range(10))})


def test_split_ratio_approx(toy_df: pd.DataFrame) -> None:
    train, test = _split_by_step(toy_df, 0.8)
    assert len(train) + len(test) == len(toy_df)
    assert len(train) >= len(test)


def test_no_overlap(toy_df: pd.DataFrame) -> None:
    train, test = _split_by_step(toy_df, 0.8)
    assert set(train["step"]).isdisjoint(set(test["step"]))


def test_train_has_lower_steps(toy_df: pd.DataFrame) -> None:
    train, test = _split_by_step(toy_df, 0.8)
    assert train["step"].max() < test["step"].min()


def test_empty_test_at_ratio_1(toy_df: pd.DataFrame) -> None:
    train, test = _split_by_step(toy_df, 1.0)
    assert len(test) == 0
    assert len(train) == len(toy_df)
