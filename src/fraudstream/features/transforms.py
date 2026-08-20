import pandas as pd

from fraudstream.features.schema import FEATURE_COLUMNS


def _add_balance_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["errorBalanceOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    return out


def _add_type_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type_CASH_OUT"] = (out["type"] == "CASH_OUT").astype(int)
    out["type_TRANSFER"] = (out["type"] == "TRANSFER").astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_balance_errors(df)
    out = _add_type_dummies(out)
    return out[FEATURE_COLUMNS]
