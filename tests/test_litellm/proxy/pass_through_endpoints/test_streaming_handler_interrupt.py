"""Regression tests for PassThroughStreamingHandler.chunk_processor."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _make_streaming_response(chunks, error: Exception | None = None):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200

    async def _aiter_bytes():
        for c in chunks:
            yield c
        if error is not None:
            raise error

    mock.aiter_bytes = _aiter_bytes
    return mock


class _RecordingLoggingWorker:
    def __init__(self):
        self.coroutines = []

    def ensure_initialized_and_enqueue(self, async_coroutine):
        self.coroutines.append(async_coroutine)
        async_coroutine.close()


@pytest.mark.asyncio
async def test_chunk_processor_logs_on_normal_completion():
    chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]
    response = _make_streaming_response(chunks)
    route_logging = AsyncMock()
    logging_worker = _RecordingLoggingWorker()

    received = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=MagicMock(),
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=route_logging,
        logging_worker=logging_worker,
    ):
        received.append(chunk)

    assert received == chunks
    assert len(logging_worker.coroutines) == 1
    call_kwargs = route_logging.call_args.kwargs
    assert call_kwargs["raw_bytes"] == chunks


@pytest.mark.asyncio
async def test_chunk_processor_logs_on_client_disconnect():
    chunks = [b"event-1", b"event-2", b"event-3"]
    response = _make_streaming_response(chunks)
    route_logging = AsyncMock()
    logging_worker = _RecordingLoggingWorker()

    gen = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=MagicMock(),
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=route_logging,
        logging_worker=logging_worker,
    )

    first = await gen.__anext__()
    await gen.aclose()

    assert first == chunks[0]
    assert len(logging_worker.coroutines) == 1
    call_kwargs = route_logging.call_args.kwargs
    assert call_kwargs["raw_bytes"] == [chunks[0]]


@pytest.mark.asyncio
async def test_chunk_processor_dispatches_stream_read_error_without_success_logging():
    error = RuntimeError("upstream stream failed")
    response = _make_streaming_response([b"partial"], error=error)
    dispatcher = AsyncMock()

    route_logging = AsyncMock()
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    generator = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=logging_obj,
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=route_logging,
        dynamic_failure_dispatcher=dispatcher,
        logging_worker=_RecordingLoggingWorker(),
    )

    assert await generator.__anext__() == b"partial"
    with pytest.raises(RuntimeError, match="upstream stream failed"):
        await generator.__anext__()

    dispatcher.assert_awaited_once_with(logging_obj, error)
    route_logging.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_processor_does_not_schedule_success_logging_for_upstream_error_without_dispatcher():
    response = _make_streaming_response([b"partial"], error=RuntimeError("upstream stream failed"))

    route_logging = AsyncMock()
    generator = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=MagicMock(),
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=route_logging,
        logging_worker=_RecordingLoggingWorker(),
    )

    assert await generator.__anext__() == b"partial"
    with pytest.raises(RuntimeError, match="upstream stream failed"):
        await generator.__anext__()

    route_logging.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_processor_does_not_schedule_success_logging_for_upstream_error():
    """A 4xx/5xx upstream response is already logged as a failure by the caller
    before this generator starts; scheduling success logging here too would
    double-log the same request in SpendLogs."""
    chunks = [b'{"error": "denied"}']
    response = _make_streaming_response(chunks)
    response.status_code = 403

    route_logging = AsyncMock()
    logging_worker = _RecordingLoggingWorker()
    received = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=MagicMock(),
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=route_logging,
        logging_worker=logging_worker,
    ):
        received.append(chunk)

    assert received == chunks
    route_logging.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_processor_does_not_schedule_logging_when_no_chunks():
    response = _make_streaming_response([])

    route_logging = AsyncMock()
    logging_worker = _RecordingLoggingWorker()
    received = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=MagicMock(),
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=route_logging,
        logging_worker=logging_worker,
    ):
        received.append(chunk)

    assert received == []
    route_logging.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_processor_routes_logging_through_logging_worker():
    """The spend-log coroutine must be handed to the durable logging worker, which
    keeps a strong reference and drains on shutdown, instead of a bare
    asyncio.create_task that the event loop only weak-references and can drop
    under GC/load, silently losing the SpendLogs row for a successful call."""
    chunks = [b"chunk-1", b"chunk-2"]
    response = _make_streaming_response(chunks)

    logging_worker = _RecordingLoggingWorker()
    received = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=MagicMock(),
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=AsyncMock(),
        logging_worker=logging_worker,
    ):
        received.append(chunk)

    assert received == chunks
    assert len(logging_worker.coroutines) == 1
    assert asyncio.iscoroutine(logging_worker.coroutines[0])


@pytest.mark.asyncio
async def test_chunk_processor_routes_logging_through_logging_worker_on_disconnect():
    """Even when the client disconnects mid-stream, the partial-usage log must go
    through the durable logging worker rather than a droppable bare task."""
    chunks = [b"event-1", b"event-2", b"event-3"]
    response = _make_streaming_response(chunks)

    logging_worker = _RecordingLoggingWorker()
    gen = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-3-haiku"},
        litellm_logging_obj=MagicMock(),
        endpoint_type=EndpointType.GENERIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/bedrock/model/claude/invoke-with-response-stream",
        route_streaming_logging=AsyncMock(),
        logging_worker=logging_worker,
    )
    await gen.__anext__()
    await gen.aclose()

    assert len(logging_worker.coroutines) == 1
    assert asyncio.iscoroutine(logging_worker.coroutines[0])


def _logging_obj_with_write_once_cst():
    """Build a MagicMock that mirrors the real Logging behavior: _update_completion_start_time
    latches self.completion_start_time so the write-once guard actually latches."""
    obj = MagicMock()
    obj.completion_start_time = None

    def _update(*, completion_start_time):
        obj.completion_start_time = completion_start_time

    obj._update_completion_start_time.side_effect = _update
    return obj


@pytest.mark.asyncio
async def test_chunk_processor_stamps_completion_start_time_on_first_chunk():
    """Regression: LIT-4185 — streaming pass-through must stamp completion_start_time on
    the first upstream chunk. Otherwise _success_handler_helper_fn falls back to
    completion_start_time = end_time, and Prometheus/OTEL/SpendLogs TTFT reads as
    total request duration (0 speedup)."""
    chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]
    response = _make_streaming_response(chunks)

    mock_logging_obj = _logging_obj_with_write_once_cst()
    received = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-haiku-4-5"},
        litellm_logging_obj=mock_logging_obj,
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/v1/messages",
        route_streaming_logging=AsyncMock(),
        logging_worker=_RecordingLoggingWorker(),
    ):
        received.append(chunk)

    assert received == chunks
    mock_logging_obj._update_completion_start_time.assert_called_once()
    stamped = mock_logging_obj._update_completion_start_time.call_args.kwargs["completion_start_time"]
    assert isinstance(stamped, datetime)


@pytest.mark.asyncio
async def test_chunk_processor_does_not_reset_completion_start_time_on_later_chunks():
    """The stamp must be write-once: reading it on chunk 2/3 must not overwrite a real TTFT
    from chunk 1 (which would collapse TTFT to time-to-last-chunk)."""
    chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]
    response = _make_streaming_response(chunks)

    real_first = datetime(2020, 1, 1, 0, 0, 0)
    mock_logging_obj = MagicMock()
    # Simulate first-chunk stamp having already landed (e.g. under contention or a
    # prior wrapper that already set it): later chunks must be no-ops.
    mock_logging_obj.completion_start_time = real_first

    async for _ in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "claude-haiku-4-5"},
        litellm_logging_obj=mock_logging_obj,
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/v1/messages",
        route_streaming_logging=AsyncMock(),
        logging_worker=_RecordingLoggingWorker(),
    ):
        pass

    mock_logging_obj._update_completion_start_time.assert_not_called()
    assert mock_logging_obj.completion_start_time == real_first


@pytest.mark.asyncio
async def test_chunk_processor_stamps_completion_start_time_on_cost_injection_path():
    """The cost-injection branch runs alongside a hot path; both must stamp TTFT."""
    import litellm as litellm_mod

    chunks = [b"event: message_start\ndata: {}\n\n"]
    response = _make_streaming_response(chunks)

    mock_logging_obj = _logging_obj_with_write_once_cst()
    mock_logging_obj.model_call_details = {"model": "claude-haiku-4-5"}
    route_logging = AsyncMock()
    logging_worker = _RecordingLoggingWorker()

    original = getattr(litellm_mod, "include_cost_in_streaming_usage", False)
    litellm_mod.include_cost_in_streaming_usage = True
    try:
        async for _ in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-haiku-4-5"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(),
            url_route="/v1/messages",
            route_streaming_logging=route_logging,
            logging_worker=logging_worker,
        ):
            pass
    finally:
        litellm_mod.include_cost_in_streaming_usage = original

    mock_logging_obj._update_completion_start_time.assert_called_once()
    route_logging.assert_called_once()
    assert len(logging_worker.coroutines) == 1


def _openai_passthrough_stream_chunks():
    return [
        (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{"content":"Hi"}}],"usage":null}\n\n'
        ),
        b": keepalive\n\n",
        (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],'
            b'"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15,'
            b'"prompt_tokens_details":{"cached_tokens":0,"audio_tokens":0},'
            b'"completion_tokens_details":{"reasoning_tokens":0,"audio_tokens":0,'
            b'"accepted_prediction_tokens":0,"rejected_prediction_tokens":0}}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]


async def _collect_openai_passthrough_chunks(chunks, endpoint_type):
    response = _make_streaming_response(chunks)
    received = []
    async for chunk in PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={"model": "gpt-4o-mini", "stream": True},
        litellm_logging_obj=MagicMock(),
        endpoint_type=endpoint_type,
        start_time=datetime.now(),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/openai/v1/chat/completions",
        route_streaming_logging=AsyncMock(),
        logging_worker=_RecordingLoggingWorker(),
    ):
        received.append(chunk)
    await asyncio.sleep(0)
    return received


@pytest.mark.asyncio
async def test_chunk_processor_injects_cost_into_openai_passthrough_usage_frame(monkeypatch):
    """Regression: issue #36492 — with include_cost_in_streaming_usage on, the final
    OpenAI passthrough chat.completion.chunk usage frame must carry usage.cost, like
    every other streaming surface already does."""
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    chunks = _openai_passthrough_stream_chunks()

    received = await _collect_openai_passthrough_chunks(chunks, EndpointType.OPENAI)

    assert received[0] == chunks[0]
    assert received[1] == chunks[1]
    assert received[3] == chunks[3]
    final_payload = json.loads(received[2].decode("utf-8").split("data:", 1)[1].strip())
    pricing = litellm.model_cost["gpt-4o-mini"]
    expected_cost = 11 * pricing["input_cost_per_token"] + 4 * pricing["output_cost_per_token"]
    assert final_payload["usage"]["cost"] == pytest.approx(expected_cost)
    assert final_payload["usage"]["cost"] > 0
    assert final_payload["usage"]["prompt_tokens"] == 11
    assert final_payload["usage"]["completion_tokens"] == 4
    assert final_payload["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_chunk_processor_injects_cost_into_usage_frame_fragmented_across_chunks(monkeypatch):
    """Regression: an SSE usage frame split across transport chunks must still get
    cost injected once the frame completes, instead of passing through untouched."""
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    whole = _openai_passthrough_stream_chunks()
    usage_frame = whole[2]
    split_at = len(usage_frame) // 2
    chunks = [whole[0], whole[1], usage_frame[:split_at], usage_frame[split_at:], whole[3]]

    received = await _collect_openai_passthrough_chunks(chunks, EndpointType.OPENAI)

    reassembled = b"".join(received).decode("utf-8")
    usage_lines = [ln for ln in reassembled.split("\n") if '"total_tokens"' in ln]
    assert len(usage_lines) == 1
    final_payload = json.loads(usage_lines[0].split("data:", 1)[1].strip())
    assert final_payload["usage"]["cost"] > 0
    assert final_payload["usage"]["prompt_tokens"] == 11
    assert reassembled.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_chunk_processor_streams_crlf_delimited_frames_live_and_injects_cost(monkeypatch):
    """Regression: CRLF-delimited SSE frames must flow as they complete instead of
    buffering until EOF, and the usage frame must still get cost injected."""
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    chunks = [chunk.replace(b"\n\n", b"\r\n\r\n") for chunk in _openai_passthrough_stream_chunks()]

    received = await _collect_openai_passthrough_chunks(chunks, EndpointType.OPENAI)

    assert len(received) == len(chunks)
    assert received[0] == chunks[0]
    injected_usage_frame = received[2]
    assert injected_usage_frame.endswith(b"\r\n\r\n")
    assert b"\n" not in injected_usage_frame.replace(b"\r\n", b"")
    reassembled = b"".join(received).decode("utf-8")
    usage_lines = [ln for ln in reassembled.replace("\r\n", "\n").split("\n") if '"total_tokens"' in ln]
    assert len(usage_lines) == 1
    final_payload = json.loads(usage_lines[0].split("data:", 1)[1].strip())
    assert final_payload["usage"]["cost"] > 0


@pytest.mark.asyncio
async def test_chunk_processor_flag_off_leaves_openai_passthrough_stream_byte_identical(monkeypatch):
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)
    chunks = _openai_passthrough_stream_chunks()

    received = await _collect_openai_passthrough_chunks(chunks, EndpointType.OPENAI)

    assert received == chunks


@pytest.mark.asyncio
async def test_chunk_processor_flag_on_leaves_openai_frames_without_usage_untouched(monkeypatch):
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    chunks = [
        (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{"content":"Hi"}}],"usage":null}\n\n'
        ),
        b": keepalive\n\n",
        b"not json at all\n\n",
        b"data: [DONE]\n\n",
    ]

    received = await _collect_openai_passthrough_chunks(chunks, EndpointType.OPENAI)

    assert received == chunks


@pytest.mark.asyncio
async def test_chunk_processor_flag_on_leaves_generic_passthrough_untouched(monkeypatch):
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    chunks = _openai_passthrough_stream_chunks()

    received = await _collect_openai_passthrough_chunks(chunks, EndpointType.GENERIC)

    assert received == chunks


def test_convert_raw_bytes_survives_truncated_multibyte_sequence():
    """A stream cut mid-multibyte-sequence (client disconnect) must still decode
    via errors="replace" so the usage events already received are logged, instead
    of raising UnicodeDecodeError and dropping the whole request from SpendLogs."""
    # the 3-byte "☃" (E2 98 83) is cut after 2 bytes, leaving an invalid sequence
    # that strict utf-8 decode would raise on, discarding the message_delta line too
    truncated_codepoint = "☃".encode("utf-8")[:2]
    raw_bytes = [
        b'data: {"text": "' + truncated_codepoint,
        b'\ndata: {"type": "message_delta"}\n',
    ]

    lines = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)

    assert any('"type": "message_delta"' in line for line in lines)
