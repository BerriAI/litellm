import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.messages.data import MESSAGES_EVENTS, MESSAGES_RESPONSE, call
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, relaxed
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.mark.asyncio
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
    assert first + b"".join(rest) == b"".join(STREAM.payloads())
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert isinstance(success[0].kwargs, dict)
    assert success[0].kwargs["completion_start_time"] is not None
    usage: Final = success[0].response.usage
    assert usage.prompt_tokens == MESSAGES_RESPONSE["usage"]["input_tokens"]
    assert usage.completion_tokens == MESSAGES_RESPONSE["usage"]["output_tokens"]
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_sync_stream_raises_like_python_before_sending(server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    with pytest.raises(ValueError, match="anthropic_messages_handler is not implemented for sync calls"):
        await call(server, asynchronous=False, stream=True, callbacks=[recorder])

    assert not server.requests
    assert not recorder.names


@pytest.mark.asyncio
async def test_stream_closed_early_logs_success_once(server: RecordingServer) -> None:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await call(server, stream=True, callbacks=[recorder])
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()

    assert len(await recorder.wait_for_async("async_log_success_event")) == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_stream_cancelled_while_waiting_on_the_provider_logs_success_for_what_it_read(
    server: RecordingServer,
) -> None:
    server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS, chunk_delay=1))
    recorder: Final = RecordingLogger()

    stream: Final = await call(server, stream=True, callbacks=[recorder])
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(anext(stream), timeout=0.2)

    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].response.usage.prompt_tokens == MESSAGES_RESPONSE["usage"]["input_tokens"]
    assert "log_failure_event" not in recorder.names


async def _streamed_pre_call_input(server: RecordingServer) -> object:
    server.enqueue(STREAM)
    recorder: Final = RecordingLogger()
    with rebound(litellm, "callbacks", [recorder]):
        stream: Final = await call(
            server, stream=True, system="Be brief", metadata={"user_id": "user-1"}, stop_sequences=["END"]
        )
        assert isinstance(stream, AsyncIterator)
        _ = [chunk async for chunk in stream]
    pre_call: Final = recorder.wait_for("log_pre_api_call")[0].kwargs
    assert isinstance(pre_call, dict)
    return pre_call["input"]


@pytest.mark.asyncio
async def test_stream_logs_the_request_body_as_input_in_the_python_handler_key_order(server: RecordingServer) -> None:
    logged: Final = await _streamed_pre_call_input(server)

    assert_served_natively(server)
    assert logged == [
        {
            "role": "user",
            "content": '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"model":"claude-sonnet-5",'
            '"metadata":{"user_id":"user-1"},"stop_sequences":["END"],"stream":true,"system":"Be brief"}',
        }
    ]
