import io
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from fraudstream.serving.prediction_logger import PredictionLogger


def _s3() -> MagicMock:
    return MagicMock()


def _record(score: float = 0.5) -> dict:
    return {"score": score, "model_version": "7", "amount": 1000.0}


def test_add_accumulates_without_flushing_below_threshold() -> None:
    logger = PredictionLogger(s3=_s3(), flush_size=3)
    logger.add(_record())
    logger.add(_record())
    assert len(logger._buffer) == 2
    logger._s3.put_object.assert_not_called()


def test_add_flushes_at_threshold() -> None:
    s3 = _s3()
    logger = PredictionLogger(s3=s3, flush_size=2)
    logger.add(_record(0.1))
    logger.add(_record(0.9))
    s3.put_object.assert_called_once()
    assert len(logger._buffer) == 0


def test_flush_writes_parquet_to_correct_prefix() -> None:
    s3 = _s3()
    logger = PredictionLogger(s3=s3, flush_size=100)
    logger.add(_record())
    logger.flush(now=datetime(2026, 8, 22, 14, 0, 0))

    s3.put_object.assert_called_once()
    kwargs = s3.put_object.call_args[1]
    assert kwargs["Bucket"] == "fraudstream-lake"
    assert kwargs["Key"].startswith("predictions/dt=2026-08-22/")
    assert kwargs["Key"].endswith(".parquet")
    body = kwargs["Body"]
    df = pd.read_parquet(io.BytesIO(body))
    assert len(df) == 1
    assert df["score"].iloc[0] == pytest.approx(0.5)


def test_flush_noop_on_empty_buffer() -> None:
    s3 = _s3()
    logger = PredictionLogger(s3=s3, flush_size=10)
    logger.flush()
    s3.put_object.assert_not_called()


def test_flush_clears_buffer() -> None:
    logger = PredictionLogger(s3=_s3(), flush_size=100)
    logger.add(_record())
    logger.flush()
    assert len(logger._buffer) == 0
