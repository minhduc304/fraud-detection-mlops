from pathlib import Path

import pytest

from fraudstream.training.register import load_model, save_model


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
