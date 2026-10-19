from collections.abc import AsyncIterator
from typing import Final

import pytest

from tests.test_litellm_rust.messages.data import MESSAGES_RESPONSE, MESSAGES_SSE, call
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.results import consume
from tests.test_litellm_rust.support.routing import assert_served_natively
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

pytestmark = pytest.mark.requires_rust_extension

@pytest.mark.asyncio
async def test_stream_relays_provider_events_and_logs_success_once_after_the_last_chunk(
    server: SharedProvider,
) -> None:
    stream_reply: Final = Reply(chunks=MESSAGES_SSE, content_type="text/event-stream")
    server.expect(stream_reply)
    recorder: Final = RecordingLogger()

    stream: Final = await call(server, stream=True, callbacks=[recorder])
    assert isinstance(stream, AsyncIterator)
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [chunk async for chunk in stream]

    assert_served_natively(server)
    assert first + b"".join(rest) == b"".join(MESSAGES_SSE)
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["completion_start_time"] is not None
    usage: Final = success[0].response.usage
    assert usage.prompt_tokens == MESSAGES_RESPONSE["usage"]["input_tokens"]
    assert usage.completion_tokens == MESSAGES_RESPONSE["usage"]["output_tokens"]
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_sync_stream_relays_provider_events_and_logs_success_once(server: SharedProvider) -> None:
    server.expect(Reply(chunks=MESSAGES_SSE, content_type="text/event-stream"))
    recorder: Final = RecordingLogger()

    stream: Final = await call(server, asynchronous=False, stream=True, callbacks=[recorder])
    chunks: Final = await consume(stream)

    assert_served_natively(server)
    assert b"".join(chunks) == b"".join(MESSAGES_SSE)
    assert len(await recorder.wait_for_async("async_log_success_event")) == 1


@pytest.mark.asyncio
async def test_stream_closed_early_logs_success_once(server: SharedProvider) -> None:
    server.expect(Reply(chunks=MESSAGES_SSE, content_type="text/event-stream"))
    recorder: Final = RecordingLogger()

    stream: Final = await call(server, stream=True, callbacks=[recorder])
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()

    assert len(await recorder.wait_for_async("async_log_success_event")) == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert_served_natively(server)
