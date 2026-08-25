.PHONY: lint test train serve up mlflow-server repro ingest-up produce consume update-quality-baseline platform-up platform-down

lint:
	uv run ruff check src/ && uv run mypy src/

test:
	uv run pytest tests/unit/ -q

train:
	uv run python -m fraudstream.training.train

serve:
	uv run uvicorn fraudstream.serving.app:app --host 0.0.0.0 --port 8000

up:
	docker compose up

mlflow-server:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

repro:
	uv run dvc repro

ingest-up:
	docker compose up -d zookeeper kafka schema-registry minio minio-init

produce:
	uv run python -m fraudstream.ingest.producer

consume:
	uv run python -m fraudstream.ingest.consumer

# Persistent local kind cluster hosting in-cluster Airflow + mlflow (Phase 7).
# Replaces the old compose-based `airflow-up` target — Airflow now runs in-cluster.
# Separate from CD's throwaway kind cluster used only for serving smoke tests.
platform-up:
	kind get clusters | grep -q '^fraudstream-platform$$' || kind create cluster --config infra/kind/platform-cluster.yaml
	kubectl --context kind-fraudstream-platform get namespace fraudstream || kubectl --context kind-fraudstream-platform create namespace fraudstream

platform-down:
	kind delete cluster --name fraudstream-platform

# Run after intentional model improvements or features/schema.py changes.
update-quality-baseline:
	uv run python scripts/update_quality_baseline.py
