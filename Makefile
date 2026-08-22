.PHONY: lint test train serve up mlflow-server repro ingest-up produce consume

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
