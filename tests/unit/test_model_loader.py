from unittest.mock import MagicMock

import pytest

from fraudstream.serving.model_loader import ALIAS, MODEL_NAME, ModelLoader


class _FakeModelVersion:
    def __init__(self, version: str) -> None:
        self.version = version


def _fake_client(version: str = "3") -> MagicMock:
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _FakeModelVersion(version)
    return client


def test_load_fetches_by_production_alias() -> None:
    client = _fake_client("3")
    load_model_fn = MagicMock(return_value="fake-model")

    loader = ModelLoader(client=client, load_model_fn=load_model_fn)
    model = loader.load()

    client.get_model_version_by_alias.assert_called_once_with(MODEL_NAME, ALIAS)
    load_model_fn.assert_called_once_with(f"models:/{MODEL_NAME}/3")
    assert model == "fake-model"
    assert loader.version == "3"


def test_load_caches_and_does_not_refetch() -> None:
    client = _fake_client("1")
    loader = ModelLoader(client=client, load_model_fn=MagicMock(return_value="model-v1"))

    loader.load()
    loader.load()

    assert client.get_model_version_by_alias.call_count == 1


def test_reload_refetches_even_when_cached() -> None:
    client = _fake_client("1")
    loader = ModelLoader(client=client, load_model_fn=MagicMock(return_value="model-v1"))
    loader.load()

    client.get_model_version_by_alias.return_value = _FakeModelVersion("2")
    loader._load_model_fn = MagicMock(return_value="model-v2")
    model = loader.reload()

    assert client.get_model_version_by_alias.call_count == 2
    assert model == "model-v2"
    assert loader.version == "2"


def test_version_is_none_before_load() -> None:
    loader = ModelLoader(client=_fake_client())
    assert loader.version is None


def test_model_property_raises_if_no_model_loaded() -> None:
    loader = ModelLoader(client=_fake_client())
    with pytest.raises(RuntimeError):
        loader.model
