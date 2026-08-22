from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from fraudstream.serving.app import create_app
from fraudstream.serving.model_loader import ModelLoader


class _FakeModelVersion:
    def __init__(self, version: str) -> None:
        self.version = version


def _loader(model_loaded: bool) -> ModelLoader:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _FakeModelVersion("1")
    loader = ModelLoader(client=client, load_model_fn=MagicMock(return_value="fake-model"))
    if model_loaded:
        loader.load()
    return loader


def test_health_always_ok() -> None:
    app = create_app(loader=_loader(model_loaded=False))
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_false_when_model_not_loaded() -> None:
    app = create_app(loader=_loader(model_loaded=False))
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 503


def test_ready_true_when_model_loaded() -> None:
    app = create_app(loader=_loader(model_loaded=True))
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}
