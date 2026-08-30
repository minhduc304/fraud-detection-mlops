"""Unit tests for the chunked split+featurize module (offline, synthetic CSV)."""
from pathlib import Path

import pandas as pd
import pytest

from fraudstream.features.transforms import build_features
from fraudstream.training.build_features_retrain import compute_split_step, split_and_featurize

RATIO = 0.8


def _synthetic_df(n_rows: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": range(1, n_rows + 1),
            "type": ["CASH_OUT" if i % 3 == 0 else "TRANSFER" if i % 3 == 1 else "PAYMENT" for i in range(n_rows)],
            "amount": [float(i * 10) for i in range(n_rows)],
            "oldbalanceOrg": [float(i * 100) for i in range(n_rows)],
            "newbalanceOrig": [float(i * 90) for i in range(n_rows)],
            "oldbalanceDest": [float(i * 50) for i in range(n_rows)],
            "newbalanceDest": [float(i * 60) for i in range(n_rows)],
            "isFraud": [i % 5 == 0 for i in range(n_rows)],
        }
    )


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "paysim.csv"
    _synthetic_df().to_csv(csv_path, index=False)
    return csv_path


def test_compute_split_step_matches_direct_quantile(synthetic_csv: Path) -> None:
    df = pd.read_csv(synthetic_csv)
    expected = df["step"].quantile(RATIO)

    actual = compute_split_step(synthetic_csv, RATIO)

    assert actual == expected


def test_chunked_output_matches_whole_dataframe(synthetic_csv: Path, tmp_path: Path) -> None:
    df = pd.read_csv(synthetic_csv)
    split_step = df["step"].quantile(RATIO)

    train_df = df[df["step"] <= split_step].reset_index(drop=True)
    test_df = df[df["step"] > split_step].reset_index(drop=True)
    expected_X_train = build_features(train_df)
    expected_y_train = train_df[["isFraud"]]
    expected_X_test = build_features(test_df)
    expected_y_test = test_df[["isFraud"]]

    out_dir = tmp_path / "out"
    # chunksize smaller than test-set size to force multi-chunk writes on both sides
    split_and_featurize(synthetic_csv, split_step, out_dir, chunksize=17)

    pd.testing.assert_frame_equal(
        pd.read_parquet(out_dir / "X_train.parquet").reset_index(drop=True), expected_X_train
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(out_dir / "y_train.parquet").reset_index(drop=True), expected_y_train
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(out_dir / "X_test.parquet").reset_index(drop=True), expected_X_test
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(out_dir / "y_test.parquet").reset_index(drop=True), expected_y_test
    )
