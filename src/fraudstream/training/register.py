import pickle
from pathlib import Path
from typing import Any


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
