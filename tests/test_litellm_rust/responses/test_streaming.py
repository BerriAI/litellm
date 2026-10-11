from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

import litellm
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.llms.openai import (
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.test_inference import RESPONSES_MODEL, RESPONSES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension

ITEM: Final = RESPONSES_RESPONSE["output"][0]
PART: Final = ITEM["content"][0]
OPENED: Final = {**RESPONSES_RESPONSE, "status": "in_progress", "output": [], "usage": None}
EVENTS: Final = (
    ("response.created", {"type": "response.created", "response": OPENED}),
    ("response.in_progress", {"type": "response.in_progress", "response": OPENED}),
    (
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**ITEM, "status": "in_progress", "content": []},
        },
    ),
    (
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": "msg_native",
            "output_index": 0,
            "content_index": 0,
            "part": {**PART, "text": ""},
        },
    ),
    (
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "item_id": "msg_native",
            "output_index": 0,
            "content_index": 0,
            "delta": "native ",
        },
    ),
    (
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "item_id": "msg_native",
            "output_index": 0,
            "content_index": 0,
            "delta": "response",
        },
    ),
    (
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": "msg_native",
            "output_index": 0,
            "content_index": 0,
            "text": PART["text"],
        },
    ),
    (
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": "msg_native",
            "output_index": 0,
            "content_index": 0,
            "part": PART,
        },
    ),
    ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": ITEM}),
    ("response.completed", {"type": "response.completed", "response": RESPONSES_RESPONSE}),
)
EVENT_TYPES: Final = (
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    OutputItemAddedEvent,
    ContentPartAddedEvent,
    OutputTextDeltaEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ContentPartDoneEvent,
    OutputItemDoneEvent,
    ResponseCompletedEvent,
)
STREAM: Final = ResponseSpec(body=None, events=EVENTS)


@pytest.fixture(autouse=True)
def opt_responses_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.RESPONSES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


@pytest.fixture
def responses_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = STREAM
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": RESPONSES_MODEL,
        "input": "hello",
        "max_output_tokens": 32,
        "api_key": "test-key",
        "api_base": server.base_url,
        "stream": True,
        **kwargs,
    }


def assert_served_natively(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert server.requests[0].path == "/responses"
    assert server.requests[0].body["stream"] is True
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.asyncio
async def test_native_responses_stream_yields_the_typed_events_and_logs_success_once_with_the_completed_response(
    responses_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.aresponses(**arguments(responses_server, callbacks=[recorder]))
    assert isinstance(stream, AsyncIterator)
    assert get_hidden_params_dict(stream)["additional_headers"]["x-litellm-rust"] == "true"
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [event async for event in stream]

    events: Final = [first, *rest]
    assert [type(event) for event in events] == list(EVENT_TYPES)
    assert "".join(event.delta for event in events if isinstance(event, OutputTextDeltaEvent)) == PART["text"]
    assert events[-1].response.id == RESPONSES_RESPONSE["id"]
    assert_served_natively(responses_server)
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].kwargs["stream"] is True
    assert success[0].kwargs["completion_start_time"] is not None
    assert success[0].response.id == RESPONSES_RESPONSE["id"]
    assert success[0].response.usage.total_tokens == RESPONSES_RESPONSE["usage"]["total_tokens"]
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_responses_stream_records_the_completed_response_before_handing_over_the_terminal_event(
    responses_server: RecordingServer,
) -> None:
    stream: Final = await litellm.aresponses(**arguments(responses_server))
    assert isinstance(stream, AsyncIterator)

    seen: Final = [(type(event), stream.completed_response is not None) async for event in stream]

    assert seen[:-1] == [(event_type, False) for event_type in EVENT_TYPES[:-1]]
    assert seen[-1] == (ResponseCompletedEvent, True)
    assert stream.completed_response.response.id == RESPONSES_RESPONSE["id"]


def test_native_sync_responses_stream_yields_the_typed_events_and_logs_success_once(
    responses_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    stream: Final = litellm.responses(**arguments(responses_server, callbacks=[recorder]))
    assert isinstance(stream, Iterator)

    events: Final = list(stream)
    assert [type(event) for event in events] == list(EVENT_TYPES)
    assert_served_natively(responses_server)
    success: Final = recorder.wait_for("async_log_success_event")
    assert len(success) == 1
    assert success[0].response.id == RESPONSES_RESPONSE["id"]


@pytest.mark.asyncio
async def test_native_responses_stream_tolerates_a_done_marker_and_comments(responses_server: RecordingServer) -> None:
    wire: Final = b"".join((*STREAM.payloads()[:2], b": keepalive\n\n", *STREAM.payloads()[2:], b"data: [DONE]\n\n"))
    responses_server.enqueue(ResponseSpec(body=wire, headers={"Content-Type": "text/event-stream"}))

    stream: Final = await litellm.aresponses(**arguments(responses_server))
    assert isinstance(stream, AsyncIterator)

    assert [type(event) async for event in stream] == list(EVENT_TYPES)


@pytest.mark.asyncio
async def test_native_responses_stream_closed_early_logs_no_success_without_a_terminal_event(
    responses_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.aresponses(**arguments(responses_server, callbacks=[recorder]))
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()
    await drain_logging()

    assert "async_log_success_event" not in recorder.names
    assert "async_log_failure_event" not in recorder.names
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
