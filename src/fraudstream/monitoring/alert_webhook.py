"""Translator: Alertmanager's webhook payload -> Airflow's DAG-run REST API.

Alertmanager's alert JSON doesn't match Airflow's dagRuns request schema, so this
service sits between them. On any firing alert in the batch, triggers one
retrain_dag run (not one per alert — Alertmanager already groups related alerts).
"""
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI

AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080")
AIRFLOW_AUTH = ("airflow", "airflow")


def create_app(http_client: Any = None) -> FastAPI:
    app = FastAPI()
    app.state.http_client = http_client or httpx.Client()

    @app.post("/alert")
    def alert(payload: dict[str, Any]) -> dict[str, bool]:
        alerts = payload.get("alerts", [])
        if not any(a.get("status") == "firing" for a in alerts):
            return {"triggered": False}

        dag_run_id = f"drift_triggered__{uuid.uuid4().hex}"
        app.state.http_client.post(
            f"{AIRFLOW_URL}/api/v1/dags/retrain_dag/dagRuns",
            json={
                "dag_run_id": dag_run_id,
                "logical_date": datetime.now(UTC).isoformat(),
                "conf": {"triggered_by": "drift_alert"},
            },
            auth=AIRFLOW_AUTH,
        )
        return {"triggered": True}

    return app


app = create_app()
