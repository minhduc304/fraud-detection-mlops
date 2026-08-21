RAW_TRANSACTION_AVRO_SCHEMA = {
    "type": "record",
    "name": "RawTransaction",
    "namespace": "fraudstream.ingest",
    "fields": [
        {"name": "step", "type": "int"},
        {"name": "type", "type": "string"},
        {"name": "amount", "type": "double"},
        {"name": "nameOrig", "type": "string"},
        {"name": "oldbalanceOrg", "type": "double"},
        {"name": "newbalanceOrig", "type": "double"},
        {"name": "nameDest", "type": "string"},
        {"name": "oldbalanceDest", "type": "double"},
        {"name": "newbalanceDest", "type": "double"},
        {"name": "isFraud", "type": "int"},
        {"name": "isFlaggedFraud", "type": "int"},
    ],
}
