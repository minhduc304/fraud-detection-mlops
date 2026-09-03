FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

USER airflow
COPY --chown=airflow:root src/ /opt/fraudstream/src/
RUN pip install --no-cache-dir pydantic numpy pandas scipy prometheus-client boto3
ENV PYTHONPATH="/opt/fraudstream/src:${PYTHONPATH}"
