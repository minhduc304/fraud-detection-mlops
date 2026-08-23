FROM python:3.11-slim AS builder
WORKDIR /build
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /build/.venv /app/.venv
COPY src/ /app/src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

USER appuser
EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "fraudstream.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
