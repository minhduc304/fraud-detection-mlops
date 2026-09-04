FROM python:3.11-slim-bookworm AS builder
WORKDIR /build
RUN pip install uv
RUN uv venv .venv && uv pip install --python .venv/bin/python \
    "confluent-kafka[avro]" pandas pydantic httpx

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /build/.venv /app/.venv
COPY src/ /app/src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

USER appuser
CMD ["/app/.venv/bin/python", "-m", "fraudstream.ingest.scoring_consumer"]
