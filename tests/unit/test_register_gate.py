"""Offline tests for promote_if_better — all cases mock mlflow client."""
from unittest.mock import MagicMock, patch

import mlflow.exceptions
import pytest

from fraudstream.training.register import promote_if_better

MODEL_NAME = "fraudstream-classifier"
RUN_ID = "abc123"
ARTIFACT = "xgb_model"
MIN_PR_AUC = 0.70


def _mock_client(champion_pr_auc: float | None = None) -> MagicMock:
    client = MagicMock()
    if champion_pr_auc is None:
        client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
            "No champion"
        )
    else:
        mock_version = MagicMock()
        mock_version.tags = {"pr_auc": str(champion_pr_auc)}
        client.get_model_version_by_alias.return_value = mock_version
    return client


@patch("fraudstream.training.register.mlflow.register_model")
@patch("fraudstream.training.register.mlflow.MlflowClient")
def test_floor_fails_no_promotion(MockClient: MagicMock, mock_register: MagicMock) -> None:
    MockClient.return_value = _mock_client(champion_pr_auc=None)
    mock_register.return_value = MagicMock(version="1")

    result = promote_if_better(RUN_ID, MODEL_NAME, ARTIFACT, pr_auc=0.65, min_pr_auc=MIN_PR_AUC)

    assert result is False
    MockClient.return_value.set_registered_model_alias.assert_not_called()


@patch("fraudstream.training.register.mlflow.register_model")
@patch("fraudstream.training.register.mlflow.MlflowClient")
def test_floor_passes_first_run(MockClient: MagicMock, mock_register: MagicMock) -> None:
    MockClient.return_value = _mock_client(champion_pr_auc=None)
    mock_register.return_value = MagicMock(version="1")

    result = promote_if_better(RUN_ID, MODEL_NAME, ARTIFACT, pr_auc=0.85, min_pr_auc=MIN_PR_AUC)

    assert result is True
    MockClient.return_value.set_registered_model_alias.assert_called_once_with(
        MODEL_NAME, "staging", "1"
    )


@patch("fraudstream.training.register.mlflow.register_model")
@patch("fraudstream.training.register.mlflow.MlflowClient")
def test_beats_champion_promotes(MockClient: MagicMock, mock_register: MagicMock) -> None:
    MockClient.return_value = _mock_client(champion_pr_auc=0.80)
    mock_register.return_value = MagicMock(version="2")

    result = promote_if_better(RUN_ID, MODEL_NAME, ARTIFACT, pr_auc=0.90, min_pr_auc=MIN_PR_AUC)

    assert result is True
    MockClient.return_value.set_registered_model_alias.assert_called_once_with(
        MODEL_NAME, "staging", "2"
    )


@patch("fraudstream.training.register.mlflow.register_model")
@patch("fraudstream.training.register.mlflow.MlflowClient")
def test_loses_to_champion_no_promotion(MockClient: MagicMock, mock_register: MagicMock) -> None:
    MockClient.return_value = _mock_client(champion_pr_auc=0.80)
    mock_register.return_value = MagicMock(version="3")

    result = promote_if_better(RUN_ID, MODEL_NAME, ARTIFACT, pr_auc=0.75, min_pr_auc=MIN_PR_AUC)

    assert result is False
    MockClient.return_value.set_registered_model_alias.assert_not_called()
