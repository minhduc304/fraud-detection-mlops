from datetime import datetime

from fraudstream.ingest.consumer import build_partition_path, group_into_batches, validate_or_route


def test_partition_path_basic() -> None:
    dt = datetime(2026, 1, 15, 9, 30, 0)
    assert build_partition_path(dt) == "raw/transactions/dt=2026-01-15/hour=09"


def test_partition_path_hour_boundary() -> None:
    dt = datetime(2026, 1, 15, 23, 59, 59)
    assert build_partition_path(dt) == "raw/transactions/dt=2026-01-15/hour=23"


def test_partition_path_day_boundary() -> None:
    dt = datetime(2026, 1, 16, 0, 0, 0)
    assert build_partition_path(dt) == "raw/transactions/dt=2026-01-16/hour=00"


def test_partition_path_single_digit_hour_padded() -> None:
    dt = datetime(2026, 1, 15, 3, 0, 0)
    assert build_partition_path(dt) == "raw/transactions/dt=2026-01-15/hour=03"


def test_group_into_batches_exact_multiple() -> None:
    events = list(range(10))
    batches = group_into_batches(events, batch_size=5)
    assert batches == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_group_into_batches_remainder() -> None:
    events = list(range(7))
    batches = group_into_batches(events, batch_size=3)
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_group_into_batches_empty() -> None:
    assert group_into_batches([], batch_size=5) == []


def test_group_into_batches_smaller_than_batch() -> None:
    events = [1, 2]
    assert group_into_batches(events, batch_size=5) == [[1, 2]]


def _valid_event() -> dict:
    return {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "nameDest": "M1",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


def test_validate_or_route_valid_passes_through_unchanged() -> None:
    event = _valid_event()
    valid, dead_letter = validate_or_route([event])
    assert valid == [event]
    assert dead_letter == []


def test_validate_or_route_malformed_goes_to_dead_letter() -> None:
    bad = _valid_event()
    del bad["amount"]
    valid, dead_letter = validate_or_route([bad])
    assert valid == []
    assert len(dead_letter) == 1
    assert dead_letter[0]["event"] == bad
    assert "reason" in dead_letter[0]


def test_validate_or_route_mixed_batch() -> None:
    good = _valid_event()
    bad = _valid_event()
    bad["amount"] = "not-a-number"
    valid, dead_letter = validate_or_route([good, bad])
    assert valid == [good]
    assert len(dead_letter) == 1
