from fraudstream.ingest.producer import apply_drift, compute_emit_delays


def test_delays_are_monotonic_non_decreasing() -> None:
    steps = [0, 1, 1, 2, 5]
    delays = compute_emit_delays(steps, speed=1.0)
    assert delays == sorted(delays)


def test_delay_zero_at_step_zero() -> None:
    delays = compute_emit_delays([0, 1, 2], speed=1.0)
    assert delays[0] == 0.0


def test_speed_multiplier_scales_delay_inversely() -> None:
    steps = [0, 1, 2]
    delays_1x = compute_emit_delays(steps, speed=1.0)
    delays_10x = compute_emit_delays(steps, speed=10.0)
    assert delays_10x[-1] == delays_1x[-1] / 10.0


def test_same_step_yields_same_delay() -> None:
    delays = compute_emit_delays([3, 3, 3], speed=1.0)
    assert delays[0] == delays[1] == delays[2]


def test_one_hour_step_is_3600_seconds_at_speed_1() -> None:
    delays = compute_emit_delays([0, 1], speed=1.0)
    assert delays[1] == 3600.0


def _event(amount: float = 100.0) -> dict:
    return {"amount": amount, "oldbalanceOrg": 500.0, "newbalanceOrig": 400.0}


def test_no_drift_before_threshold() -> None:
    event = _event()
    result = apply_drift(event, index=5, drift_after=10, factor=2.0)
    assert result == event


def test_drift_active_at_threshold() -> None:
    event = _event(amount=100.0)
    result = apply_drift(event, index=10, drift_after=10, factor=2.0)
    assert result["amount"] == 200.0


def test_drift_active_after_threshold() -> None:
    event = _event(amount=100.0)
    result = apply_drift(event, index=50, drift_after=10, factor=2.0)
    assert result["amount"] == 200.0


def test_drift_does_not_mutate_input() -> None:
    event = _event(amount=100.0)
    apply_drift(event, index=10, drift_after=10, factor=2.0)
    assert event["amount"] == 100.0


def test_drift_only_scales_amount_field() -> None:
    event = _event(amount=100.0)
    result = apply_drift(event, index=10, drift_after=10, factor=2.0)
    assert result["oldbalanceOrg"] == event["oldbalanceOrg"]
    assert result["newbalanceOrig"] == event["newbalanceOrig"]
