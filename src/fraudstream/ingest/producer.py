from typing import Any

SECONDS_PER_STEP = 3600.0


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
