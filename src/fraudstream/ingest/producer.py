import argparse
import json
import time
from typing import Any

import pandas as pd
from confluent_kafka import SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import StringSerializer

from fraudstream.ingest.schemas import RAW_TRANSACTION_AVRO_SCHEMA

SECONDS_PER_STEP = 3600.0
TOPIC = "transactions.raw"


def compute_emit_delays(steps: list[int], speed: float) -> list[float]:
    """Wall-clock delay (seconds) from replay start for each event's `step`."""
    return [step * SECONDS_PER_STEP / speed for step in steps]


def apply_drift(
    event: dict[str, Any], index: int, drift_after: int, factor: float
) -> dict[str, Any]:
    """Scale `amount` by `factor` once `index` reaches `drift_after`, else pass through."""
    if index < drift_after:
        return event
    drifted = dict(event)
    drifted["amount"] = event["amount"] * factor
    return drifted


def _build_producer(bootstrap_servers: str, schema_registry_url: str) -> SerializingProducer:
    registry_client = SchemaRegistryClient({"url": schema_registry_url})
    value_serializer = AvroSerializer(registry_client, json.dumps(RAW_TRANSACTION_AVRO_SCHEMA))
    return SerializingProducer(
        {
            "bootstrap.servers": bootstrap_servers,
            "key.serializer": StringSerializer("utf_8"),
            "value.serializer": value_serializer,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay paysim.csv as a Kafka stream")
    parser.add_argument("--csv", default="data/raw/paysim.csv")
    parser.add_argument("--speed", type=float, default=1000.0)
    parser.add_argument("--drift-after", type=int, default=-1)
    parser.add_argument("--drift-factor", type=float, default=3.0)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--schema-registry-url", default="http://localhost:8081")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    steps = df["step"].tolist()
    delays = compute_emit_delays(steps, args.speed)

    producer = _build_producer(args.bootstrap_servers, args.schema_registry_url)
    start = time.monotonic()
    for index, (row, delay) in enumerate(zip(df.to_dict("records"), delays, strict=True)):
        event = dict(row)
        if args.drift_after >= 0:
            event = apply_drift(event, index, args.drift_after, args.drift_factor)
        wait = start + delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        producer.produce(topic=TOPIC, key=str(event["nameOrig"]), value=event)
        producer.poll(0)
    producer.flush()


if __name__ == "__main__":
    main()
