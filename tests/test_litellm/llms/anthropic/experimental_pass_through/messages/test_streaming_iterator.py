import asyncio
import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.experimental_pass_through.messages import streaming_iterator as streaming_iterator_module
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    INCOMPLETE_STREAM_ERROR_MESSAGE,
    BaseAnthropicMessagesStreamingIterator,
    _incomplete_stream_error_sse_event,
    _is_message_stop_chunk,
)


class _RecordingLoggingIterator(BaseAnthropicMessagesStreamingIterator):
    def __init__(self, litellm_logging_obj: LiteLLMLoggingObj, request_body: dict):
        super().__init__(litellm_logging_obj=litellm_logging_obj, request_body=request_body)
        self.logged_chunks: list = []

    async def _handle_streaming_logging(self, collected_chunks):
        self.logged_chunks = list(collected_chunks)


def _make_logging_obj(test_name: str) -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="chat",
        start_time=datetime.now(),
        litellm_call_id=test_name,
        function_id=test_name,
    )


def _make_iterator(test_name: str) -> BaseAnthropicMessagesStreamingIterator:
    return BaseAnthropicMessagesStreamingIterator(
        litellm_logging_obj=_make_logging_obj(test_name),
        request_body={},
    )


async def _collect(iterator, stream):
    return [chunk async for chunk in iterator.async_sse_wrapper(stream)]


TRUNCATED_TOOL_USE_EVENTS = (
    {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 1}}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "tooluse_1", "name": "write", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"path": "/builder/docs/QUAL'},
    },
)


@pytest.mark.asyncio
async def test_async_sse_wrapper_emits_error_event_when_stream_ends_without_message_stop():
    """
    Regression test for LIT-3724: a Bedrock stream that goes silent
    mid tool_use must not be passed through as a successful, complete
    SSE stream. An `error` SSE event must be appended so strict clients
    (Anthropic SDK, Claude Code) surface the truncation instead of
    crashing on unterminated tool-call JSON.
    """

    async def _truncated_stream():
        for event in TRUNCATED_TOOL_USE_EVENTS:
            yield event

    iterator = _make_iterator("test_truncated_stream_emits_error")
    chunks = await _collect(iterator, _truncated_stream())

    assert len(chunks) == len(TRUNCATED_TOOL_USE_EVENTS) + 1
    error_chunk = chunks[-1].decode()
    assert error_chunk.startswith("event: error\n")
    assert error_chunk.endswith("\n\n")

    error_payload = json.loads(error_chunk.split("data: ", 1)[1])
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"
    assert error_payload["error"]["message"] == INCOMPLETE_STREAM_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_async_sse_wrapper_no_error_event_on_complete_stream():
    async def _complete_stream():
        for event in TRUNCATED_TOOL_USE_EVENTS:
            yield event
        yield {"type": "content_block_stop", "index": 0}
        yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}
        yield {"type": "message_stop"}

    iterator = _make_iterator("test_complete_stream_no_error")
    chunks = await _collect(iterator, _complete_stream())

    assert len(chunks) == len(TRUNCATED_TOOL_USE_EVENTS) + 3
    decoded = [chunk.decode() for chunk in chunks]
    assert decoded[-1].startswith("event: message_stop\n")
    assert not any(chunk.startswith("event: error\n") for chunk in decoded)


@pytest.mark.asyncio
async def test_async_sse_wrapper_emits_error_event_on_empty_stream():
    async def _empty_stream():
        return
        yield

    iterator = _make_iterator("test_empty_stream_emits_error")
    chunks = await _collect(iterator, _empty_stream())

    assert len(chunks) == 1
    assert chunks[0].decode().startswith("event: error\n")


@pytest.mark.asyncio
async def test_async_sse_wrapper_treats_message_stop_bytes_as_complete():
    async def _byte_stream():
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield b'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    iterator = _make_iterator("test_byte_stream_message_stop")
    chunks = await _collect(iterator, _byte_stream())

    assert len(chunks) == 2
    assert not any(chunk.startswith(b"event: error\n") for chunk in chunks)


def test_is_message_stop_chunk():
    assert _is_message_stop_chunk({"type": "message_stop"}) is True
    assert _is_message_stop_chunk({"type": "message_delta"}) is False
    assert _is_message_stop_chunk(b'event: message_stop\ndata: {}\n\n') is True
    assert _is_message_stop_chunk(b"raw-bytes") is False
    assert _is_message_stop_chunk("message_stop") is False


def test_is_message_stop_chunk_ignores_substring_in_payload():
    """
    Regression: a `content_block_delta` frame whose payload happens to contain
    the literal string `message_stop` (e.g. inside a tool's partial_json) must
    not be treated as a terminal stop event.
    """
    delta_frame_with_substring = (
        b'event: content_block_delta\n'
        b'data: {"type": "content_block_delta", "delta": '
        b'{"type": "input_json_delta", "partial_json": "\\"message_stop\\""}}\n\n'
    )
    assert _is_message_stop_chunk(delta_frame_with_substring) is False


@pytest.mark.asyncio
async def test_async_sse_wrapper_emits_error_when_bytes_stream_only_mentions_message_stop_in_payload():
    """
    Regression for the bytes-branch substring false positive: a stream whose
    payload text contains `message_stop` (but never emits the actual
    `event: message_stop` frame) must still be flagged as incomplete.
    """
    async def _byte_stream():
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield (
            b'event: content_block_delta\n'
            b'data: {"type": "content_block_delta", "delta": '
            b'{"type": "input_json_delta", "partial_json": "\\"message_stop\\""}}\n\n'
        )

    iterator = _make_iterator("test_bytes_substring_does_not_mark_complete")
    chunks = await _collect(iterator, _byte_stream())

    assert len(chunks) == 3
    assert chunks[-1].decode().startswith("event: error\n")


@pytest.mark.asyncio
async def test_async_sse_wrapper_does_not_double_error_on_provider_error_dict():
    """
    Regression: when the provider itself terminates the stream with an
    `error` event (without a `message_stop`), the wrapper must forward that
    error and not append a second synthetic incomplete-stream error.
    """
    provider_error = {"type": "error", "error": {"type": "overloaded_error", "message": "boom"}}

    async def _error_terminated_stream():
        yield {"type": "message_start", "message": {"id": "msg_1"}}
        yield provider_error

    iterator = _make_iterator("test_provider_error_terminal_dict")
    chunks = await _collect(iterator, _error_terminated_stream())

    assert len(chunks) == 2
    error_frames = [c for c in chunks if c.startswith(b"event: error\n")]
    assert len(error_frames) == 1
    payload = json.loads(error_frames[0].decode().split("data: ", 1)[1])
    assert payload == provider_error


@pytest.mark.asyncio
async def test_async_sse_wrapper_does_not_double_error_on_provider_error_bytes():
    async def _byte_stream():
        yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
        yield b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error"}}\n\n'

    iterator = _make_iterator("test_provider_error_terminal_bytes")
    chunks = await _collect(iterator, _byte_stream())

    assert len(chunks) == 2
    error_frames = [c for c in chunks if c.startswith(b"event: error\n")]
    assert len(error_frames) == 1


@pytest.mark.asyncio
async def test_async_sse_wrapper_excludes_synthetic_error_event_from_logged_chunks():
    async def _truncated_stream():
        for event in TRUNCATED_TOOL_USE_EVENTS:
            yield event

    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_synthetic_error_not_logged"),
        request_body={},
    )
    chunks = await _collect(iterator, _truncated_stream())

    assert chunks[-1].startswith(b"event: error\n")
    assert iterator.logged_chunks == chunks[:-1]
    assert not any(chunk.startswith(b"event: error\n") for chunk in iterator.logged_chunks)


def test_incomplete_stream_error_sse_event_is_valid_anthropic_error():
    event = _incomplete_stream_error_sse_event().decode()
    lines = event.split("\n")
    assert lines[0] == "event: error"
    payload = json.loads(lines[1].removeprefix("data: "))
    assert payload == {
        "type": "error",
        "error": {"type": "api_error", "message": INCOMPLETE_STREAM_ERROR_MESSAGE},
    }
    assert event.endswith("\n\n")


_STREAM_PREFIX = (
    {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 52, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "The Roman"}},
)
_STREAM_TAIL = (
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " Empire ..."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 64}},
    {"type": "message_stop"},
)


def _output_tokens_from_logged_chunks(chunks: list[bytes]) -> int | None:
    """Read the last output_tokens the billing path would see from the SSE bytes."""
    latest: int | None = None
    for raw in chunks:
        for line in raw.decode().splitlines():
            if not line.startswith("data:"):
                continue
            data = json.loads(line[len("data:"):].strip())
            usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(usage, dict) and usage.get("output_tokens") is not None:
                latest = usage["output_tokens"]
    return latest


@pytest.mark.asyncio
async def test_async_sse_wrapper_bills_full_stream_after_client_disconnect():
    """
    Regression: on a client disconnect mid-stream the upstream provider keeps
    generating (and billing) the full response. The wrapper must keep draining
    that upstream to its terminal ``message_delta`` and bill the real
    output_tokens (64), not the partial count the client drained before leaving
    (the message_start placeholder, 1).

    A ``tail_gated`` event holds back the stream tail until the client has
    disconnected, so the tail can only be captured by a drain that survives the
    client teardown - exactly the path the previous implementation dropped.
    """
    tail_gated = asyncio.Event()
    upstream_fully_drained = asyncio.Event()

    async def _gated_stream():
        for event in _STREAM_PREFIX:
            yield event
        await tail_gated.wait()
        for event in _STREAM_TAIL:
            yield event
        upstream_fully_drained.set()

    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_bills_full_stream_after_disconnect"),
        request_body={},
    )

    gen = iterator.async_sse_wrapper(_gated_stream())

    client_chunks = []
    async for chunk in gen:
        client_chunks.append(chunk)
        if len(client_chunks) == len(_STREAM_PREFIX):
            break
    await gen.aclose()

    tail_gated.set()
    await asyncio.wait_for(upstream_fully_drained.wait(), timeout=5)
    for _ in range(100):
        if iterator.logged_chunks:
            break
        await asyncio.sleep(0.01)

    assert len(client_chunks) == len(_STREAM_PREFIX)

    assert iterator.logged_chunks, "pump never billed after client disconnect"
    assert _output_tokens_from_logged_chunks(iterator.logged_chunks) == 64
    assert any(c.startswith(b"event: message_stop\n") for c in iterator.logged_chunks)
    assert not any(c.startswith(b"event: error\n") for c in iterator.logged_chunks)


@pytest.mark.asyncio
async def test_async_sse_wrapper_bills_full_stream_when_client_reads_all():
    """Happy path: when the client drains the whole stream, billing still sees
    the terminal output_tokens (64) and the client gets every chunk."""
    tail_gated = asyncio.Event()
    tail_gated.set()  # no gating; full stream flows immediately

    async def _full_stream():
        for event in (*_STREAM_PREFIX, *_STREAM_TAIL):
            yield event

    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_bills_full_stream_happy_path"),
        request_body={},
    )
    client_chunks = [chunk async for chunk in iterator.async_sse_wrapper(_full_stream())]

    for _ in range(100):
        if iterator.logged_chunks:
            break
        await asyncio.sleep(0.01)

    assert len(client_chunks) == len(_STREAM_PREFIX) + len(_STREAM_TAIL)
    assert _output_tokens_from_logged_chunks(iterator.logged_chunks) == 64
    assert not any(c.startswith(b"event: error\n") for c in iterator.logged_chunks)


class _ProviderStreamError(Exception):
    """Stand-in for a provider-specific streaming failure carrying a status code."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.asyncio
async def test_async_sse_wrapper_reraises_upstream_error_to_connected_client():
    """
    Regression: an upstream failure (Bedrock read / decode / chunk-conversion)
    before message_stop must propagate the ORIGINAL provider exception to a
    still-connected client, so the proxy's failure handling keeps the
    provider-specific status. The pump must not swallow it into a generic
    api_error event + normal termination.
    """

    async def _failing_stream():
        yield {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 52, "output_tokens": 1}}}
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}}
        raise _ProviderStreamError("bedrock stream blew up", status_code=529)

    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_reraises_upstream_error"),
        request_body={},
    )

    received = []
    with pytest.raises(_ProviderStreamError) as excinfo:
        async for chunk in iterator.async_sse_wrapper(_failing_stream()):
            received.append(chunk)

    assert excinfo.value.status_code == 529
    assert received
    assert not any(c.startswith(b"event: error\n") for c in received)
    assert iterator.logged_chunks == []


@pytest.mark.asyncio
async def test_async_sse_wrapper_salvages_partial_spend_on_upstream_error_after_disconnect():
    """
    When the upstream errors AFTER the client has already disconnected there is
    no live client to re-raise to and no failure hook will run, so the pump
    salvages partial spend from what it collected instead of dropping the row.
    """
    tail_gated = asyncio.Event()

    async def _gated_failing_stream():
        yield {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 52, "output_tokens": 1}}}
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}}
        await tail_gated.wait()
        raise _ProviderStreamError("late failure", status_code=500)

    iterator = _RecordingLoggingIterator(
        litellm_logging_obj=_make_logging_obj("test_salvage_partial_on_late_error"),
        request_body={},
    )

    gen = iterator.async_sse_wrapper(_gated_failing_stream())
    received = [await gen.__anext__(), await gen.__anext__()]
    await gen.aclose()  # client disconnects before the upstream error

    tail_gated.set()  # let the upstream raise now, after disconnect
    for _ in range(100):
        if iterator.logged_chunks:
            break
        await asyncio.sleep(0.01)

    assert len(received) == 2
    assert iterator.logged_chunks == received


@pytest.mark.asyncio
async def test_async_sse_wrapper_applies_backpressure_to_slow_client(monkeypatch):
    """
    Regression: the relay queue is bounded, so a slow client throttles the
    upstream read instead of letting the pump buffer the whole response in
    memory. With a tiny queue and a client that reads a single chunk, the pump
    must stall after producing only a queue's worth of chunks ahead, not race
    to the end of a large stream.
    """
    monkeypatch.setattr(streaming_iterator_module, "ANTHROPIC_MESSAGES_STREAM_RELAY_QUEUE_MAXSIZE", 2)

    total = 200
    produced = 0

    async def _fast_stream():
        nonlocal produced
        for i in range(total):
            produced += 1
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"t{i}"}}

    iterator = _make_iterator("test_backpressure_slow_client")
    gen = iterator.async_sse_wrapper(_fast_stream())
    try:
        await gen.__anext__()
        for _ in range(500):
            await asyncio.sleep(0)
        assert produced <= 2 + 3, f"pump ran ahead unthrottled: produced {produced} of {total}"
        assert produced < total
    finally:
        await gen.aclose()


@pytest.mark.asyncio
async def test_async_sse_wrapper_bills_partial_when_detached_drain_cap_reached(monkeypatch):
    """
    Regression: when the concurrent detached-drain cap is already reached, a
    pump whose client has disconnected must bill what it collected instead of
    continuing to drain (and accumulating) the rest of a large upstream stream,
    so slow/abandoned clients can't pin unbounded worker state.

    The cap slot set is pre-occupied so the single slot is unavailable when this
    pump reaches its first post-disconnect chunk; that isolates the cap decision
    from multi-pump scheduling races.
    """
    monkeypatch.setattr(streaming_iterator_module, "ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS", 1)
    monkeypatch.setattr(streaming_iterator_module, "ANTHROPIC_MESSAGES_STREAM_RELAY_QUEUE_MAXSIZE", 4)

    async def _hold_slot():
        await asyncio.sleep(3600)

    holder = asyncio.ensure_future(_hold_slot())
    streaming_iterator_module._DETACHED_STREAM_DRAINS.add(holder)
    tail_reached = False

    async def _long_stream():
        nonlocal tail_reached
        yield {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 5, "output_tokens": 1}}}
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}}
        for i in range(100):
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"more{i}"}}
        tail_reached = True
        yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}}
        yield {"type": "message_stop"}

    iterator = _RecordingLoggingIterator(litellm_logging_obj=_make_logging_obj("drain_cap_full"), request_body={})
    try:
        gen = iterator.async_sse_wrapper(_long_stream())
        await gen.__anext__()  # message_start
        await gen.__anext__()  # first delta
        await gen.aclose()  # client disconnects; 100+ chunks remain upstream

        for _ in range(200):
            if iterator.logged_chunks:
                break
            await asyncio.sleep(0)

        assert iterator.logged_chunks, "capped pump never billed"
        assert len(iterator.logged_chunks) <= 2 + streaming_iterator_module.ANTHROPIC_MESSAGES_STREAM_RELAY_QUEUE_MAXSIZE
        assert len(iterator.logged_chunks) < 100
        assert not any(c.startswith(b"event: message_stop\n") for c in iterator.logged_chunks)
        assert tail_reached is False, "pump kept draining past the cap instead of stopping"
    finally:
        holder.cancel()
        streaming_iterator_module._DETACHED_STREAM_DRAINS.discard(holder)


@pytest.mark.asyncio
async def test_async_sse_wrapper_aborts_upstream_when_detached_drain_cap_reached(monkeypatch):
    """
    Regression: when the cap is full and a disconnected pump bails, it must call
    aclose on the upstream stream so the provider stops generating and billing,
    not continue running the stream while we record only the partial prefix.
    """
    monkeypatch.setattr(streaming_iterator_module, "ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS", 1)
    monkeypatch.setattr(streaming_iterator_module, "ANTHROPIC_MESSAGES_STREAM_RELAY_QUEUE_MAXSIZE", 4)

    async def _hold_slot():
        await asyncio.sleep(3600)

    holder = asyncio.ensure_future(_hold_slot())
    streaming_iterator_module._DETACHED_STREAM_DRAINS.add(holder)

    class _AbortableStream:
        def __init__(self):
            self.aclose_called = False
            self._remaining = iter(
                (
                    {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 5, "output_tokens": 1}}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
                )
                + tuple(
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"t{i}"}}
                    for i in range(50)
                )
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._remaining)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            self.aclose_called = True

    stream = _AbortableStream()
    iterator = _RecordingLoggingIterator(litellm_logging_obj=_make_logging_obj("abort_upstream_at_cap"), request_body={})
    try:
        gen = iterator.async_sse_wrapper(stream)
        await gen.__anext__()
        await gen.__anext__()
        await gen.aclose()

        for _ in range(200):
            if iterator.logged_chunks:
                break
            await asyncio.sleep(0)

        assert iterator.logged_chunks, "capped pump never billed"
        assert stream.aclose_called, "upstream aclose was not called when the detached-drain cap was reached"
    finally:
        holder.cancel()
        streaming_iterator_module._DETACHED_STREAM_DRAINS.discard(holder)


@pytest.mark.asyncio
async def test_abort_upstream_logs_warning_when_aclose_raises(caplog):
    """_abort_upstream must swallow and log any exception from aclose()."""
    import logging

    class _ExplodingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("aclose exploded")

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await BaseAnthropicMessagesStreamingIterator._abort_upstream(_ExplodingStream())

    assert any("abort" in r.message and "RuntimeError" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_enqueue_for_client_returns_false_when_already_detached():
    """_enqueue_for_client must return False immediately (without touching the queue)
    when client_detached is already set before the call."""
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
        BaseAnthropicMessagesStreamingIterator,
    )

    queue: asyncio.Queue[bytes | None | BaseException] = asyncio.Queue(maxsize=1)
    client_detached = asyncio.Event()
    client_detached.set()

    result = await BaseAnthropicMessagesStreamingIterator._enqueue_for_client(queue, client_detached, b"chunk")
    assert result is False
    assert queue.empty()


@pytest.mark.asyncio
async def test_enqueue_for_client_returns_false_when_client_detaches_while_queue_full():
    """_enqueue_for_client must return False (and cancel the put) when the queue
    is full and client_detached fires before space becomes available."""
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
        BaseAnthropicMessagesStreamingIterator,
    )

    queue: asyncio.Queue[bytes | None | BaseException] = asyncio.Queue(maxsize=1)
    queue.put_nowait(b"already-full")

    client_detached = asyncio.Event()

    async def _set_detached_soon():
        await asyncio.sleep(0.01)
        client_detached.set()

    asyncio.create_task(_set_detached_soon())
    result = await BaseAnthropicMessagesStreamingIterator._enqueue_for_client(queue, client_detached, b"new-chunk")
    assert result is False
    assert queue.qsize() == 1
    assert queue.get_nowait() == b"already-full"


@pytest.mark.asyncio
async def test_async_sse_wrapper_drains_detached_when_cap_available(monkeypatch):
    """Complement to the cap test: with a slot free, a disconnected pump drains
    the full upstream and bills the terminal usage, and releases its slot after."""
    monkeypatch.setattr(streaming_iterator_module, "ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS", 1)
    monkeypatch.setattr(streaming_iterator_module, "ANTHROPIC_MESSAGES_STREAM_RELAY_QUEUE_MAXSIZE", 4)

    async def _stream():
        yield {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 5, "output_tokens": 1}}}
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}}
        for i in range(20):
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"m{i}"}}
        yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}}
        yield {"type": "message_stop"}

    iterator = _RecordingLoggingIterator(litellm_logging_obj=_make_logging_obj("drain_cap_free"), request_body={})
    gen = iterator.async_sse_wrapper(_stream())
    await gen.__anext__()
    await gen.__anext__()
    await gen.aclose()

    for _ in range(300):
        if iterator.logged_chunks:
            break
        await asyncio.sleep(0.01)

    assert any(c.startswith(b"event: message_stop\n") for c in iterator.logged_chunks)
    assert len(streaming_iterator_module._DETACHED_STREAM_DRAINS) == 0
