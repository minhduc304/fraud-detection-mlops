FROM python:3.11-slim-bookworm AS builder
WORKDIR /build
RUN pip install uv
# Explicit minimal set (not `uv sync` against the full pyproject.toml) — serving only
# needs these to run app.py/model_loader.py/prediction_logger.py + deserialize the
# champion model. Skips dvc/confluent-kafka/lightgbm/httpx, which serving never imports.
RUN uv venv .venv && uv pip install --python .venv/bin/python \
    fastapi uvicorn pandas mlflow boto3 prometheus-client pydantic scikit-learn "xgboost>=3.2.0"

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /build/.venv /app/.venv
COPY src/ /app/src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

USER appuser
EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "fraudstream.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
