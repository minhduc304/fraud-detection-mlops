import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from fraudstream.training.register import load_model, save_model, write_reference_baseline


def test_round_trip(tmp_path: Path) -> None:
    model = {"weights": [1.0, 2.0], "bias": 0.5}
    path = tmp_path / "model.pkl"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded == model


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "model.pkl"
    save_model({"x": 1}, path)
    assert path.exists()


def test_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pkl")


def _s3() -> MagicMock:
    return MagicMock()


def test_write_reference_baseline_writes_both_objects() -> None:
    s3 = _s3()
    scores = pd.Series([0.1, 0.2, 0.3])
    features_df = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "type_CASH_OUT": [0, 1, 0]})

    write_reference_baseline(scores, features_df, s3=s3)

    assert s3.put_object.call_count == 2
    keys = {c.kwargs["Key"] for c in s3.put_object.call_args_list}
    assert keys == {
        "reference/schema_v1/score_reference.npy",
        "reference/schema_v1/features_reference.parquet",
    }
    for c in s3.put_object.call_args_list:
        assert c.kwargs["Bucket"] == "fraudstream-lake"


def test_write_reference_baseline_score_body_roundtrips() -> None:
    s3 = _s3()
    scores = pd.Series([0.1, 0.2, 0.3])
    features_df = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "type_CASH_OUT": [0, 1, 0]})

    write_reference_baseline(scores, features_df, s3=s3)

    score_call = next(
        c
        for c in s3.put_object.call_args_list
        if c.kwargs["Key"] == "reference/schema_v1/score_reference.npy"
    )
    loaded = np.load(io.BytesIO(score_call.kwargs["Body"]))
    np.testing.assert_allclose(loaded, scores.to_numpy())


def test_write_reference_baseline_features_body_roundtrips() -> None:
    s3 = _s3()
    scores = pd.Series([0.1, 0.2, 0.3])
    features_df = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "type_CASH_OUT": [0, 1, 0]})

    write_reference_baseline(scores, features_df, s3=s3)

    features_call = next(
        c
        for c in s3.put_object.call_args_list
        if c.kwargs["Key"] == "reference/schema_v1/features_reference.parquet"
    )
    loaded = pd.read_parquet(io.BytesIO(features_call.kwargs["Body"]))
    pd.testing.assert_frame_equal(loaded, features_df)
