from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from fraudstream.serving.app import create_app
from fraudstream.serving.model_loader import ModelLoader


class _FakeModelVersion:
    def __init__(self, version: str) -> None:
        self.version = version


class _RawModel:
    """Stand-in for the raw estimator behind the pyfunc wrapper."""

    def predict_proba(self, df: object) -> np.ndarray:
        return np.array([[0.13, 0.87]])


class _FixtureModel:
    """Stand-in for a pyfunc-loaded model. `predict` returns a class label (what the mlflow
    xgboost/sklearn pyfunc wrapper actually does); `get_raw_model().predict_proba` is the
    fraud probability serving must report."""

    def predict(self, df: object) -> np.ndarray:
        return np.array([1.0])

    def get_raw_model(self) -> _RawModel:
        return _RawModel()


def _valid_txn() -> dict:
    return {
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "nameOrig": "C1",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "nameDest": "M1",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


def _client_with_loaded_model() -> TestClient:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _FakeModelVersion("7")
    loader = ModelLoader(client=client, load_model_fn=MagicMock(return_value=_FixtureModel()))
    loader.load()
    app = create_app(loader=loader)
    return TestClient(app)


def test_predict_returns_score_and_model_version() -> None:
    client = _client_with_loaded_model()
    resp = client.post("/predict", json=_valid_txn())
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] == 0.87
    assert body["model_version"] == "7"


def test_predict_score_is_probability_not_class_label() -> None:
    client = _client_with_loaded_model()
    body = client.post("/predict", json=_valid_txn()).json()
    assert body["score"] == 0.87  # predict_proba[:, 1], not predict() which returns 1.0


def test_predict_rejects_malformed_body() -> None:
    client = _client_with_loaded_model()
    bad = _valid_txn()
    del bad["amount"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_returns_503_when_model_not_loaded() -> None:
    loader = ModelLoader(client=MagicMock())
    app = create_app(loader=loader)
    resp = TestClient(app).post("/predict", json=_valid_txn())
    assert resp.status_code == 503
