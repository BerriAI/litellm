from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

from tests.test_litellm_rust.responses.data import RESPONSES_EVENTS, RESPONSES_RESPONSE, STREAMING, call
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, relaxed
from tests.test_litellm_rust.support.results import consume, response_id
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=RESPONSES_EVENTS)


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.mark.asyncio
@STREAMING
async def test_stream_relays_provider_events_and_logs_success_once_after_the_last_chunk(
    server: RecordingServer,
) -> None:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await call(server, stream=True, callbacks=[recorder])
    assert isinstance(stream, AsyncIterator)
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [chunk async for chunk in stream]

    assert_served_natively(server)
    assert [getattr(chunk, "type", None) for chunk in (first, *rest)] == [event for event, _ in RESPONSES_EVENTS]
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["completion_start_time"] is not None
    assert response_id(success[0].response) == RESPONSES_RESPONSE["id"]
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
@STREAMING
async def test_sync_stream_relays_provider_events_and_logs_success_once(server: RecordingServer) -> None:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await call(server, asynchronous=False, stream=True, callbacks=[recorder])
    chunks: Final = await consume(stream)

    assert_served_natively(server)
    assert [getattr(chunk, "type", None) for chunk in chunks] == [event for event, _ in RESPONSES_EVENTS]
    assert len(await recorder.wait_for_async("log_success_event")) == 1
