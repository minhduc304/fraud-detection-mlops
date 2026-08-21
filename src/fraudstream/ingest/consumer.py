import argparse
import io
import json
import uuid
from datetime import datetime
from typing import Any, TypeVar, cast

import boto3
import pandas as pd
from confluent_kafka import DeserializingConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import StringDeserializer
from pydantic import ValidationError

from fraudstream.features.schema import RawTransaction
from fraudstream.ingest.producer import TOPIC

T = TypeVar("T")

BUCKET = "fraudstream-lake"
DEAD_LETTER_PREFIX = "raw/_dead_letter"


def build_partition_path(dt: datetime) -> str:
    return f"raw/transactions/dt={dt:%Y-%m-%d}/hour={dt:%H}"


def group_into_batches(events: list[T], batch_size: int) -> list[list[T]]:
    return [events[i : i + batch_size] for i in range(0, len(events), batch_size)]


def validate_or_route(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    dead_letter: list[dict[str, Any]] = []
    for event in events:
        try:
            RawTransaction.model_validate(event)
            valid.append(event)
        except ValidationError as exc:
            dead_letter.append({"event": event, "reason": str(exc)})
    return valid, dead_letter


def _build_consumer(
    bootstrap_servers: str, schema_registry_url: str, group_id: str
) -> DeserializingConsumer:
    registry_client = SchemaRegistryClient({"url": schema_registry_url})
    value_deserializer = AvroDeserializer(registry_client)
    consumer = DeserializingConsumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "key.deserializer": StringDeserializer("utf_8"),
            "value.deserializer": value_deserializer,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([TOPIC])
    return consumer


def _write_parquet(s3: Any, events: list[dict[str, Any]], key_prefix: str) -> None:
    df = pd.DataFrame(events)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    key = f"{key_prefix}/part-{uuid.uuid4().hex}.parquet"
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())


def _write_dead_letter(s3: Any, entries: list[dict[str, Any]], dt: datetime) -> None:
    key_prefix = f"{DEAD_LETTER_PREFIX}/dt={dt:%Y-%m-%d}"
    body = "\n".join(json.dumps(entry) for entry in entries).encode("utf-8")
    key = f"{key_prefix}/part-{uuid.uuid4().hex}.jsonl"
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume transactions.raw and write to MinIO")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--schema-registry-url", default="http://localhost:8081")
    parser.add_argument("--group-id", default="fraudstream-consumer")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--minio-endpoint", default="http://localhost:9000")
    parser.add_argument("--minio-access-key", default="minioadmin")
    parser.add_argument("--minio-secret-key", default="minioadmin")
    parser.add_argument("--poll-timeout", type=float, default=1.0)
    parser.add_argument("--max-messages", type=int, default=-1)
    args = parser.parse_args()

    s3 = boto3.client(
        "s3",
        endpoint_url=args.minio_endpoint,
        aws_access_key_id=args.minio_access_key,
        aws_secret_access_key=args.minio_secret_key,
    )
    consumer = _build_consumer(args.bootstrap_servers, args.schema_registry_url, args.group_id)

    def _flush(batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        valid, dead_letter = validate_or_route(batch)
        now = datetime.now()
        if valid:
            _write_parquet(s3, valid, build_partition_path(now))
        if dead_letter:
            _write_dead_letter(s3, dead_letter, now)

    buffer: list[dict[str, Any]] = []
    consumed = 0
    try:
        while args.max_messages < 0 or consumed < args.max_messages:
            msg = consumer.poll(args.poll_timeout)
            if msg is None:
                continue
            if msg.error():
                continue
            buffer.append(cast(dict[str, Any], msg.value()))
            consumed += 1
            if len(buffer) >= args.batch_size:
                _flush(buffer)
                buffer = []
        _flush(buffer)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
