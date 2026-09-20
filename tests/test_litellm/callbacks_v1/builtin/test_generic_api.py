import json
from datetime import datetime, timezone

import pytest

from litellm.callbacks_v1.builtin import generic_api
from litellm.callbacks_v1.builtin.port import CallRecord
from litellm.callbacks_v1.builtin.runtime import CallJoin, Sink
from tests.test_litellm.callbacks_v1.builtin.support import golden_call

PORT = generic_api.GenericApi(generic_api.Config(endpoint="http://generic.test"))
FAILURES_ONLY = generic_api.GenericApi(
    generic_api.Config(endpoint="http://generic.test", event_types=frozenset({"llm_api_failure"}))
)
# A sink is handed the clock whether or not it stamps a time; this one does not.
UNUSED_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

COMMON = {
    "litellm_call_id": "call-123",
    "call_type": "ocr",
    "startTime": 10.0,
    "model": "model",
    "custom_llm_provider": "provider",
    "model_parameters": {"temperature": 0.2},
}


def joined(terminal: str) -> CallRecord:
    join = CallJoin()
    records = [join.accept(envelope) for envelope in golden_call(terminal)]  # pyright: ignore[reportArgumentType]  # terminal is an EventName literal at every call site
    (record,) = [record for record in records if record is not None]
    return record


def test_a_succeeded_call_uses_the_standard_logging_payload_names() -> None:
    assert PORT.payload(joined("call.succeeded"), UNUSED_NOW) == {
        **COMMON,
        "id": "response",
        "stream": False,
        "status": "success",
        "endTime": 12.5,
        "response_time": 2.5,
        "response": {"id": "response"},
    }


def test_a_failed_call_carries_error_information_instead_of_a_response() -> None:
    assert PORT.payload(joined("call.failed"), UNUSED_NOW) == {
        **COMMON,
        "id": "call-123",
        "stream": True,
        "status": "failure",
        "endTime": 11.0,
        "response_time": 1.0,
        "error_str": "bad request",
        "error_information": {"error_class": "ValueError", "error_message": "bad request", "error_code": "400"},
    }


@pytest.mark.parametrize(
    ("log_format", "bodies"),
    [
        ("json_array", ['[{"id": "a"}, {"id": "b"}]']),
        ("ndjson", ['{"id": "a"}\n{"id": "b"}']),
        ("single", ['{"id": "a"}', '{"id": "b"}']),
    ],
)
def test_a_batch_is_framed_by_log_format(log_format: generic_api.LogFormat, bodies: list[str]) -> None:
    port = generic_api.GenericApi(generic_api.Config(endpoint="http://generic.test", log_format=log_format))
    batch = ({"id": "a"}, {"id": "b"})

    sent = port.deliveries(batch)  # pyright: ignore[reportArgumentType]  # framing does not depend on the record's fields

    assert [delivery.body.decode() for delivery in sent] == bodies
    assert {delivery.url for delivery in sent} == {"http://generic.test"}


def test_an_outcome_left_out_of_event_types_has_no_record() -> None:
    assert FAILURES_ONLY.payload(joined("call.succeeded"), UNUSED_NOW) is None
    assert FAILURES_ONLY.payload(joined("call.failed"), UNUSED_NOW) is not None


def test_event_types_filter_what_is_queued() -> None:
    from tests.test_litellm.callbacks_v1.builtin.support import FakeTransport

    transport = FakeTransport()
    sink = Sink("generic_api", FAILURES_ONLY, transport)
    for terminal in ("call.succeeded", "call.failed"):
        for envelope in golden_call(terminal):
            sink.on_event(envelope)

    assert sink.outbox.flush()
    assert [record["status"] for body in transport.bodies() for record in body] == ["failure"]  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]  # json bodies are lists of records here


def test_config_parses_the_legacy_header_variable() -> None:
    config = generic_api.Config.from_env(
        {"GENERIC_LOGGER_ENDPOINT": "http://e", "GENERIC_LOGGER_HEADERS": "a=1, b = 2"}
    )
    assert config.headers == (("Content-Type", "application/json"), ("a", "1"), ("b", "2"))
    assert json.loads(json.dumps(config.headers)) == [["Content-Type", "application/json"], ["a", "1"], ["b", "2"]]
