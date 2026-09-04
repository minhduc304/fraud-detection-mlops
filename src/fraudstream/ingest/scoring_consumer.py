import argparse
import logging
import os
from typing import Any, cast

import httpx
from confluent_kafka import DeserializingConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import StringDeserializer

from fraudstream.features.schema import RawTransaction
from fraudstream.ingest.producer import TOPIC

logger = logging.getLogger(__name__)

PREDICT_FIELDS = tuple(RawTransaction.model_fields)


def to_predict_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {field: event[field] for field in PREDICT_FIELDS}


def score_event(
    event: dict[str, Any], client: httpx.Client, serving_url: str
) -> tuple[str, float | None]:
    try:
        response = client.post(f"{serving_url}/predict", json=to_predict_payload(event))
    except httpx.HTTPError as exc:
        logger.warning("predict request failed, skipping event: %s", exc)
        return "skip", None
    if response.status_code >= 500:
        logger.warning("predict returned %s, skipping event", response.status_code)
        return "skip", None
    score = float(response.json()["score"])
    # Hook: a `transactions.scored` publish ({txn_id, score, model_version}) would go here.
    # Deferred — nothing consumes such a topic yet. See scoring-consumer-gap.md.
    return "ok", score


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


def run(
    consumer: Any,
    client: httpx.Client,
    serving_url: str,
    poll_timeout: float,
    max_messages: int,
) -> None:
    consumed = 0
    try:
        while max_messages < 0 or consumed < max_messages:
            msg = consumer.poll(poll_timeout)
            if msg is None:
                continue
            if msg.error():
                continue
            score_event(cast(dict[str, Any], msg.value()), client, serving_url)
            consumed += 1
    finally:
        consumer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume transactions.raw and POST each event to serving /predict"
    )
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--schema-registry-url", default="http://localhost:8081")
    parser.add_argument("--group-id", default="fraudstream-scoring")
    parser.add_argument(
        "--serving-url", default=os.environ.get("SERVING_URL", "http://localhost:8000")
    )
    parser.add_argument("--poll-timeout", type=float, default=1.0)
    parser.add_argument("--max-messages", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    consumer = _build_consumer(args.bootstrap_servers, args.schema_registry_url, args.group_id)
    with httpx.Client(timeout=10.0) as client:
        run(consumer, client, args.serving_url, args.poll_timeout, args.max_messages)


if __name__ == "__main__":
    main()
