FROM python:3.11-slim-bookworm
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser

RUN pip install --no-cache-dir fastapi httpx "uvicorn[standard]"

COPY src/fraudstream/__init__.py /app/src/fraudstream/__init__.py
COPY src/fraudstream/monitoring/__init__.py /app/src/fraudstream/monitoring/__init__.py
COPY src/fraudstream/monitoring/alert_webhook.py /app/src/fraudstream/monitoring/alert_webhook.py

ENV PYTHONPATH="/app/src"

USER appuser
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "fraudstream.monitoring.alert_webhook:app", "--host", "0.0.0.0", "--port", "8001"]
