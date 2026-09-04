import httpx
import pytest

from fraudstream.ingest.scoring_consumer import run, score_event, to_predict_payload


def _event() -> dict:
    return {
        "step": 1,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": "C1",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "nameDest": "C2",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_to_predict_payload_maps_all_rawtransaction_fields() -> None:
    event = _event()
    event["extra_avro_field"] = "ignored"
    assert to_predict_payload(event) == _event()


def test_score_event_ok_parses_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(200, json={"score": 0.42, "model_version": "3"})

    with _client(handler) as client:
        status, score = score_event(_event(), client, "http://serving:8000")
    assert status == "ok"
    assert score == pytest.approx(0.42)


def test_score_event_5xx_skips_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "Model not loaded"})

    with _client(handler) as client:
        status, score = score_event(_event(), client, "http://serving:8000")
    assert status == "skip"
    assert score is None


def test_score_event_connection_error_skips_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client:
        status, score = score_event(_event(), client, "http://serving:8000")
    assert status == "skip"
    assert score is None


class _FakeMessage:
    def __init__(self, value: dict) -> None:
        self._value = value

    def error(self) -> None:
        return None

    def value(self) -> dict:
        return self._value


class _FakeConsumer:
    def __init__(self, messages: list, raise_first: Exception | None = None) -> None:
        self._messages = list(messages)
        self._raise_first = raise_first
        self.closed = False

    def poll(self, timeout: float):
        if self._raise_first is not None:
            exc, self._raise_first = self._raise_first, None
            raise exc
        return self._messages.pop(0) if self._messages else None

    def close(self) -> None:
        self.closed = True


def test_run_recovers_from_consume_error() -> None:
    from confluent_kafka import KafkaError
    from confluent_kafka.error import ConsumeError

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"score": 0.1, "model_version": "1"})

    err = ConsumeError(KafkaError(KafkaError.UNKNOWN_TOPIC_OR_PART))
    consumer = _FakeConsumer([_FakeMessage(_event())], raise_first=err)
    with _client(handler) as client:
        run(consumer, client, "http://serving:8000", poll_timeout=0.0, max_messages=1)
    assert len(calls) == 1
    assert consumer.closed


def test_run_honors_max_messages() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"score": 0.1, "model_version": "1"})

    consumer = _FakeConsumer([_FakeMessage(_event()) for _ in range(5)])
    with _client(handler) as client:
        run(consumer, client, "http://serving:8000", poll_timeout=0.0, max_messages=2)
    assert len(calls) == 2
    assert consumer.closed
