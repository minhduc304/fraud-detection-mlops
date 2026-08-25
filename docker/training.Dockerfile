FROM python:3.11-slim-bookworm AS builder
WORKDIR /build
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/.venv /app/.venv
COPY src/ /app/src/
COPY params.yaml /app/params.yaml

ARG GIT_SHA=unknown
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    GIT_SHA=${GIT_SHA}

CMD ["python", "-m", "fraudstream.training.train"]
