from fraudstream.features.schema import RawTransaction
from fraudstream.ingest.schemas import RAW_TRANSACTION_AVRO_SCHEMA


def test_avro_field_names_match_pydantic_model() -> None:
    avro_fields = {f["name"] for f in RAW_TRANSACTION_AVRO_SCHEMA["fields"]}
    pydantic_fields = set(RawTransaction.model_fields.keys())
    assert avro_fields == pydantic_fields


def test_avro_schema_has_no_extra_or_missing_fields() -> None:
    avro_field_names = [f["name"] for f in RAW_TRANSACTION_AVRO_SCHEMA["fields"]]
    pydantic_field_names = list(RawTransaction.model_fields.keys())
    assert len(avro_field_names) == len(pydantic_field_names)


def test_avro_schema_name_and_type() -> None:
    assert RAW_TRANSACTION_AVRO_SCHEMA["type"] == "record"
    assert RAW_TRANSACTION_AVRO_SCHEMA["name"] == "RawTransaction"
