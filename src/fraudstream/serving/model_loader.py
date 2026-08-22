from collections.abc import Callable
from typing import Any

import mlflow

MODEL_NAME = "fraudstream-classifier"
ALIAS = "production"


class ModelLoader:
    def __init__(
        self,
        client: mlflow.MlflowClient | None = None,
        load_model_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self._client = client or mlflow.MlflowClient()
        self._load_model_fn = load_model_fn or mlflow.pyfunc.load_model
        self._model: Any = None
        self._version: str | None = None

    def load(self) -> Any:
        if self._model is None:
            self.reload()
        return self._model

    def reload(self) -> Any:
        mv = self._client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        self._model = self._load_model_fn(f"models:/{MODEL_NAME}/{mv.version}")
        self._version = mv.version
        return self._model

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded yet — call load() first.")
        return self._model
