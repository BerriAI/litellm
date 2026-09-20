from collections.abc import AsyncIterator, Generator
from typing import Final

import pytest

import litellm
from litellm import callbacks_v1
from litellm.callbacks_v1 import EVENTS, EnvelopeV1, EventName, RequestFactsV1, WirePatchV1
from litellm.integrations.custom_logger import CustomLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
)

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


class BothObserver:
    """Declares both kinds; an async call must select the async one and never the sync one."""

    name: Final = "both-observer"
    schema: Final = 1
    events: Final = EVENTS

    def __init__(self) -> None:
        self.envelopes: Final[list[EnvelopeV1]] = []
        self.sync_calls: Final[list[EnvelopeV1]] = []

    def on_event(self, event: EnvelopeV1) -> None:
        self.sync_calls.append(event)

    async def async_on_event(self, event: EnvelopeV1) -> None:
        self.envelopes.append(event)


class AsyncInterceptor:
    name: Final = "async-interceptor"
    schema: Final = 1
    events: Final[frozenset[EventName]] = frozenset()

    def __init__(self) -> None:
        self.requests: Final[list[RequestFactsV1]] = []

    async def async_before_send(self, request: RequestFactsV1) -> WirePatchV1 | None:
        self.requests.append(request)
        return {"headers": {"set": [("x-v1-async-interceptor", "seen")]}}


@pytest.fixture
def observer() -> Generator[BothObserver]:
    subscriber: Final = BothObserver()
    callbacks_v1.register(subscriber)
    try:
        yield subscriber
    finally:
        callbacks_v1.unregister(subscriber)


@pytest.fixture
def interceptor() -> Generator[AsyncInterceptor]:
    subscriber: Final = AsyncInterceptor()
    callbacks_v1.register(subscriber)
    try:
        yield subscriber
    finally:
        callbacks_v1.unregister(subscriber)


class LegacyCallId(CustomLogger):
    def __init__(self) -> None:
        self.call_ids: Final[list[str]] = []

    def log_pre_api_call(self, model, messages, kwargs):
        self.call_ids.append(kwargs["litellm_call_id"])


@pytest.mark.asyncio
async def test_native_messages_async_call_selects_the_async_observer_and_shares_the_legacy_call_id(
    messages_server: RecordingServer, observer: BothObserver
) -> None:
    legacy: Final = LegacyCallId()

    await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[legacy]))

    assert observer.sync_calls == []
    assert [envelope["event"]["type"] for envelope in observer.envelopes] == [
        "call.started",
        "request.sending",
        "response.received",
        "call.succeeded",
    ]
    assert [envelope["seq"] for envelope in observer.envelopes] == [0, 1, 2, 3]
    assert {envelope["call_type"] for envelope in observer.envelopes} == {"anthropic_messages"}
    assert len(legacy.call_ids) == 1
    assert {envelope["call_id"] for envelope in observer.envelopes} == set(legacy.call_ids)


@pytest.mark.asyncio
async def test_native_messages_async_interceptor_patch_reaches_the_provider(
    messages_server: RecordingServer, interceptor: AsyncInterceptor
) -> None:
    await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert len(interceptor.requests) == 1
    request: Final = interceptor.requests[0]
    assert request["custom_llm_provider"] == "anthropic"
    assert request["headers"] != []
    assert messages_server.requests[-1].headers["x-v1-async-interceptor"] == "seen"


@pytest.mark.asyncio
async def test_native_messages_stream_marks_the_terminal_envelope_streamed(
    messages_server: RecordingServer, observer: BothObserver
) -> None:
    messages_server.enqueue(STREAM)

    stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    assert isinstance(stream, AsyncIterator)
    delivered: Final = [chunk async for chunk in stream]

    assert delivered
    terminal: Final = observer.envelopes[-1]
    assert terminal["event"]["type"] == "call.succeeded"
    assert terminal["event"]["streamed"] is True
