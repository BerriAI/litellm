from collections.abc import Generator
from typing import Final

import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge import callbacks_v1_python
from litellm.rust_bridge.callbacks_v1_python import EVENTS, EnvelopeV1, RequestFactsV1, WirePatchV1
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    OCR_RESPONSE,
    call_native_aocr,
    call_native_ocr,
    request_body,
)

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


class Observer:
    name: Final = "observer"
    schema: Final = 1
    events: Final = EVENTS

    def __init__(self) -> None:
        self.envelopes: Final[list[EnvelopeV1]] = []

    def on_event(self, event: EnvelopeV1) -> None:
        self.envelopes.append(event)


class Interceptor:
    name: Final = "interceptor"
    schema: Final = 1
    events: Final[frozenset[str]] = frozenset()

    def __init__(self) -> None:
        self.bodies: Final[list[object]] = []

    def before_send(self, request: RequestFactsV1) -> WirePatchV1 | None:
        self.bodies.append(request["body"])
        return {"headers": {"set": [("x-v1-interceptor", "seen")]}}


@pytest.fixture
def observer() -> Generator[Observer]:
    subscriber: Final = Observer()
    callbacks_v1_python.register(subscriber)
    try:
        yield subscriber
    finally:
        callbacks_v1_python.unregister(subscriber)


@pytest.fixture
def interceptor() -> Generator[Interceptor]:
    subscriber: Final = Interceptor()
    callbacks_v1_python.register(subscriber)
    try:
        yield subscriber
    finally:
        callbacks_v1_python.unregister(subscriber)


class LegacyCallId(CustomLogger):
    def __init__(self) -> None:
        self.call_ids: Final[list[str]] = []

    def log_pre_api_call(self, model, messages, kwargs):
        self.call_ids.append(kwargs["litellm_call_id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_ocr_runs_legacy_and_v1_callbacks_on_the_same_call(
    ocr_server: RecordingServer, observer: Observer, asynchronous: bool
) -> None:
    legacy: Final = LegacyCallId()

    if asynchronous:
        await call_native_aocr(ocr_server, callbacks=[legacy])
    else:
        call_native_ocr(ocr_server, callbacks=[legacy])
    await drain_logging()

    assert [envelope["event"]["type"] for envelope in observer.envelopes] == [
        "call.started",
        "request.sending",
        "response.received",
        "call.succeeded",
    ]
    assert [envelope["seq"] for envelope in observer.envelopes] == sorted(
        envelope["seq"] for envelope in observer.envelopes
    )
    assert len(legacy.call_ids) == 1
    assert {envelope["call_id"] for envelope in observer.envelopes} == set(legacy.call_ids)


def test_native_ocr_v1_interceptor_does_not_expose_the_wire_body(
    ocr_server: RecordingServer, interceptor: Interceptor
) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["include_image_base64"] = True

    call_native_ocr(ocr_server, callbacks=[Edit()])

    assert len(interceptor.bodies) == 1
    body: Final = interceptor.bodies[0]
    assert body == "[REDACTED]"
    request: Final = ocr_server.requests[-1]
    assert request.headers["x-v1-interceptor"] == "seen"


def test_native_ocr_failure_reaches_both_contracts_once(
    recording_server: RecordingServer, observer: Observer
) -> None:
    recording_server.default_response = ResponseSpec(status=400, body={"message": "bad document"})
    failures: Final[list[object]] = []

    class Failure(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            failures.append(kwargs["exception"])

    with pytest.raises(Exception) as raised:  # noqa: PT011  # the mapped provider error class is covered elsewhere
        call_native_ocr(recording_server, callbacks=[Failure()])

    assert failures == [raised.value]
    terminal: Final = [
        envelope for envelope in observer.envelopes if envelope["event"]["type"] in {"call.succeeded", "call.failed"}
    ]
    assert [envelope["event"]["type"] for envelope in terminal] == ["call.failed"]
    assert observer.envelopes[-1] is terminal[0]
