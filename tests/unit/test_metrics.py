from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from fraudstream.serving.app import create_app
from fraudstream.serving.model_loader import ModelLoader


class _FakeModelVersion:
    def __init__(self, version: str) -> None:
        self.version = version


class _RawModel:
    def predict_proba(self, df: object) -> np.ndarray:
        return np.array([[0.58, 0.42]])


class _FixtureModel:
    def predict(self, df: object) -> np.ndarray:
        return np.array([0.0])

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


def test_metrics_endpoint_exposes_all_four_metric_names_after_predict() -> None:
    client = _client_with_loaded_model()
    client.post("/predict", json=_valid_txn())

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "fraud_prediction_latency_seconds" in body
    assert "fraud_prediction_score" in body
    assert "fraud_feature_value" in body
    assert 'feature="amount"' in body
    assert "fraud_model_version_info" in body
    assert 'version="7"' in body
