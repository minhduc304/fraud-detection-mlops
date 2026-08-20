import pandas as pd
import pytest

from fraudstream.features.schema import FEATURE_COLUMNS
from fraudstream.features.transforms import _add_balance_errors, _add_type_dummies, build_features


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["CASH_OUT", "PAYMENT"],
            "amount": [100.0, 200.0],
            "nameOrig": ["C1", "C2"],
            "oldbalanceOrg": [500.0, 300.0],
            "newbalanceOrig": [400.0, 100.0],
            "nameDest": ["M1", "M2"],
            "oldbalanceDest": [0.0, 50.0],
            "newbalanceDest": [100.0, 250.0],
            "isFraud": [0, 0],
            "isFlaggedFraud": [0, 0],
        }
    )


class TestAddBalanceErrors:
    def test_errorBalanceOrig_formula(self, raw_df: pd.DataFrame) -> None:
        out = _add_balance_errors(raw_df)
        # newbalanceOrig + amount - oldbalanceOrg
        expected = raw_df["newbalanceOrig"] + raw_df["amount"] - raw_df["oldbalanceOrg"]
        pd.testing.assert_series_equal(out["errorBalanceOrig"], expected, check_names=False)

    def test_errorBalanceDest_formula(self, raw_df: pd.DataFrame) -> None:
        out = _add_balance_errors(raw_df)
        # oldbalanceDest + amount - newbalanceDest
        expected = raw_df["oldbalanceDest"] + raw_df["amount"] - raw_df["newbalanceDest"]
        pd.testing.assert_series_equal(out["errorBalanceDest"], expected, check_names=False)

    def test_no_mutation(self, raw_df: pd.DataFrame) -> None:
        original_cols = list(raw_df.columns)
        _add_balance_errors(raw_df)
        assert list(raw_df.columns) == original_cols


class TestAddTypeDummies:
    def test_cash_out_flagged(self, raw_df: pd.DataFrame) -> None:
        out = _add_type_dummies(raw_df)
        assert out["type_CASH_OUT"].iloc[0] == 1
        assert out["type_CASH_OUT"].iloc[1] == 0

    def test_transfer_flagged(self) -> None:
        df = pd.DataFrame({"type": ["TRANSFER", "DEBIT"]})
        out = _add_type_dummies(df)
        assert out["type_TRANSFER"].iloc[0] == 1
        assert out["type_TRANSFER"].iloc[1] == 0

    def test_no_mutation(self, raw_df: pd.DataFrame) -> None:
        original_cols = list(raw_df.columns)
        _add_type_dummies(raw_df)
        assert list(raw_df.columns) == original_cols


class TestBuildFeatures:
    def test_output_columns_exact(self, raw_df: pd.DataFrame) -> None:
        out = build_features(raw_df)
        assert list(out.columns) == FEATURE_COLUMNS

    def test_output_row_count_preserved(self, raw_df: pd.DataFrame) -> None:
        out = build_features(raw_df)
        assert len(out) == len(raw_df)

    def test_no_mutation(self, raw_df: pd.DataFrame) -> None:
        original_cols = list(raw_df.columns)
        build_features(raw_df)
        assert list(raw_df.columns) == original_cols

    def test_numeric_dtype(self, raw_df: pd.DataFrame) -> None:
        out = build_features(raw_df)
        for col in out.columns:
            assert pd.api.types.is_numeric_dtype(out[col]), f"{col} not numeric"
