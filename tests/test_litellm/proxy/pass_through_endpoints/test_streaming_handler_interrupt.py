"""Regression tests for LIT-2642 — interrupted pass-through streams must still log usage."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _make_streaming_response(chunks):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200

    async def _aiter_bytes():
        for c in chunks:
            yield c

    mock.aiter_bytes = _aiter_bytes
    return mock


@pytest.mark.asyncio
async def test_chunk_processor_logs_on_normal_completion():
    chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]
    response = _make_streaming_response(chunks)

    mock_logging_obj = MagicMock()
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        ):
            received.append(chunk)

        await asyncio.sleep(0)

    assert received == chunks
    mock_route.assert_called_once()
    call_kwargs = mock_route.call_args.kwargs
    assert call_kwargs["raw_bytes"] == chunks


@pytest.mark.asyncio
async def test_chunk_processor_logs_on_client_disconnect():
    chunks = [b"event-1", b"event-2", b"event-3"]
    response = _make_streaming_response(chunks)

    mock_logging_obj = MagicMock()
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        gen = PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        )

        first = await gen.__anext__()
        await gen.aclose()

        await asyncio.sleep(0)

    assert first == chunks[0]
    mock_route.assert_called_once()
    call_kwargs = mock_route.call_args.kwargs
    assert call_kwargs["raw_bytes"] == [chunks[0]]


@pytest.mark.asyncio
async def test_chunk_processor_does_not_schedule_success_logging_for_upstream_error():
    """A 4xx/5xx upstream response is already logged as a failure by the caller
    before this generator starts; scheduling success logging here too would
    double-log the same request in SpendLogs."""
    chunks = [b'{"error": "denied"}']
    response = _make_streaming_response(chunks)
    response.status_code = 403

    mock_logging_obj = MagicMock()
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        ):
            received.append(chunk)

        await asyncio.sleep(0)

    assert received == chunks
    mock_route.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_processor_does_not_schedule_logging_when_no_chunks():
    response = _make_streaming_response([])

    mock_logging_obj = MagicMock()
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        ):
            received.append(chunk)

    assert received == []
    mock_route.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_processor_routes_logging_through_logging_worker():
    """The spend-log coroutine must be handed to the durable logging worker, which
    keeps a strong reference and drains on shutdown, instead of a bare
    asyncio.create_task that the event loop only weak-references and can drop
    under GC/load, silently losing the SpendLogs row for a successful call."""
    chunks = [b"chunk-1", b"chunk-2"]
    response = _make_streaming_response(chunks)

    enqueued = []

    def _capture(async_coroutine):
        enqueued.append(async_coroutine)
        async_coroutine.close()

    with (
        patch.object(
            PassThroughStreamingHandler,
            "_route_streaming_logging_to_handler",
            new=AsyncMock(),
        ),
        patch.object(
            GLOBAL_LOGGING_WORKER,
            "ensure_initialized_and_enqueue",
            side_effect=_capture,
        ) as mock_enqueue,
    ):
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=MagicMock(),
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(),
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        ):
            received.append(chunk)

    assert received == chunks
    mock_enqueue.assert_called_once()
    assert asyncio.iscoroutine(enqueued[0])


@pytest.mark.asyncio
async def test_chunk_processor_routes_logging_through_logging_worker_on_disconnect():
    """Even when the client disconnects mid-stream, the partial-usage log must go
    through the durable logging worker rather than a droppable bare task."""
    chunks = [b"event-1", b"event-2", b"event-3"]
    response = _make_streaming_response(chunks)

    enqueued = []

    def _capture(async_coroutine):
        enqueued.append(async_coroutine)
        async_coroutine.close()

    with (
        patch.object(
            PassThroughStreamingHandler,
            "_route_streaming_logging_to_handler",
            new=AsyncMock(),
        ),
        patch.object(
            GLOBAL_LOGGING_WORKER,
            "ensure_initialized_and_enqueue",
            side_effect=_capture,
        ) as mock_enqueue,
    ):
        gen = PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-3-haiku"},
            litellm_logging_obj=MagicMock(),
            endpoint_type=EndpointType.GENERIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(),
            url_route="/bedrock/model/claude/invoke-with-response-stream",
        )
        await gen.__anext__()
        await gen.aclose()

    mock_enqueue.assert_called_once()
    assert asyncio.iscoroutine(enqueued[0])


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
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ):
        received = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-haiku-4-5"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/v1/messages",
        ):
            received.append(chunk)

        await asyncio.sleep(0)

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
    mock_passthrough_handler = MagicMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ):
        async for _ in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "claude-haiku-4-5"},
            litellm_logging_obj=mock_logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=datetime.now(),
            passthrough_success_handler_obj=mock_passthrough_handler,
            url_route="/v1/messages",
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
    mock_passthrough_handler = MagicMock()

    original = getattr(litellm_mod, "include_cost_in_streaming_usage", False)
    litellm_mod.include_cost_in_streaming_usage = True
    try:
        with patch.object(
            PassThroughStreamingHandler,
            "_route_streaming_logging_to_handler",
            new=AsyncMock(),
        ):
            async for _ in PassThroughStreamingHandler.chunk_processor(
                response=response,
                request_body={"model": "claude-haiku-4-5"},
                litellm_logging_obj=mock_logging_obj,
                endpoint_type=EndpointType.ANTHROPIC,
                start_time=datetime.now(),
                passthrough_success_handler_obj=mock_passthrough_handler,
                url_route="/v1/messages",
            ):
                pass
    finally:
        litellm_mod.include_cost_in_streaming_usage = original

    mock_logging_obj._update_completion_start_time.assert_called_once()


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
