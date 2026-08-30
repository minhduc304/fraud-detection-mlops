import io
import os
import threading
import uuid
from datetime import datetime
from typing import Any

import boto3
import pandas as pd

BUCKET = "fraudstream-lake"


class PredictionLogger:
    def __init__(
        self,
        s3: Any = None,
        flush_size: int = 100,
        flush_interval: float = 60.0,
        minio_endpoint: str | None = None,
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
    ) -> None:
        self._s3 = s3 or boto3.client(
            "s3",
            endpoint_url=minio_endpoint or os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
        )
        self._flush_size = flush_size
        self._flush_interval = flush_interval
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._flush_size:
                self._flush_locked()

    def flush(self, now: datetime | None = None) -> None:
        with self._lock:
            self._flush_locked(now=now)

    def _flush_locked(self, now: datetime | None = None) -> None:
        if not self._buffer:
            return
        dt = now or datetime.now()
        df = pd.DataFrame(self._buffer)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        key = f"predictions/dt={dt:%Y-%m-%d}/part-{uuid.uuid4().hex}.parquet"
        self._s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        self._buffer = []

    def start(self) -> None:
        self._schedule_timer()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.flush()

    def _schedule_timer(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._timer_fire)
        self._timer.daemon = True
        self._timer.start()

    def _timer_fire(self) -> None:
        self.flush()
        self._schedule_timer()
