from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from fraudstream.monitoring.alert_webhook import create_app

FIRING_PAYLOAD = {
    "alerts": [
        {
            "status": "firing",
            "labels": {"alertname": "ScoreDriftHigh", "feature": "score"},
            "annotations": {"summary": "Fraud score distribution has drifted"},
        }
    ]
}

RESOLVED_PAYLOAD = {
    "alerts": [
        {
            "status": "resolved",
            "labels": {"alertname": "ScoreDriftHigh", "feature": "score"},
            "annotations": {},
        }
    ]
}

MIXED_PAYLOAD = {
    "alerts": [
        {"status": "resolved", "labels": {"alertname": "A"}, "annotations": {}},
        {"status": "firing", "labels": {"alertname": "B"}, "annotations": {}},
    ]
}


def _http_client() -> MagicMock:
    client = MagicMock()
    client.post.return_value = MagicMock(status_code=200)
    return client


def test_firing_alert_triggers_retrain_dag() -> None:
    http_client = _http_client()
    app = create_app(http_client=http_client)

    resp = TestClient(app).post("/alert", json=FIRING_PAYLOAD)

    assert resp.status_code == 200
    http_client.post.assert_called_once()
    call = http_client.post.call_args
    assert call.args[0] == "http://airflow-webserver:8080/api/v1/dags/retrain_dag/dagRuns"
    assert "dag_run_id" in call.kwargs["json"]
    assert call.kwargs["auth"] == ("airflow", "airflow")


def test_resolved_only_alert_does_not_trigger() -> None:
    http_client = _http_client()
    app = create_app(http_client=http_client)

    resp = TestClient(app).post("/alert", json=RESOLVED_PAYLOAD)

    assert resp.status_code == 200
    http_client.post.assert_not_called()


def test_mixed_payload_triggers_once() -> None:
    http_client = _http_client()
    app = create_app(http_client=http_client)

    resp = TestClient(app).post("/alert", json=MIXED_PAYLOAD)

    assert resp.status_code == 200
    http_client.post.assert_called_once()


def test_empty_alerts_list_does_not_trigger() -> None:
    http_client = _http_client()
    app = create_app(http_client=http_client)

    resp = TestClient(app).post("/alert", json={"alerts": []})

    assert resp.status_code == 200
    http_client.post.assert_not_called()
