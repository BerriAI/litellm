"""
Tests for PassThroughStreamingHandler.chunk_processor timing corrections:

  1. `start_time` is overridden by `litellm_logging_obj.start_time` when the
     latter is earlier (true request-entry timestamp).
  2. `completion_start_time` is recorded on the first emitted chunk so TTFT
     reflects real time-to-first-token instead of `endTime`.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)


def _build_logging_obj(true_start: datetime, completion_start_time=None):
    """A minimal LiteLLMLoggingObj-shaped stub the handler can read from."""
    logging_obj = MagicMock()
    logging_obj.start_time = true_start
    logging_obj.completion_start_time = completion_start_time
    logging_obj._update_completion_start_time = MagicMock(
        side_effect=lambda completion_start_time: setattr(
            logging_obj, "completion_start_time", completion_start_time
        )
    )
    logging_obj.model_call_details = {}
    logging_obj.standard_logging_object = None
    return logging_obj


async def _build_response_with_chunks(chunks):
    """Wrap a list of bytes chunks into an httpx.Response-shaped mock that
    yields them via aiter_bytes()."""

    async def _aiter_bytes():
        for c in chunks:
            yield c

    response = MagicMock(spec=httpx.Response)
    response.aiter_bytes = _aiter_bytes
    response.headers = {}
    return response


def _drain(gen):
    """Synchronously drain an async generator to a list."""

    async def _collect():
        return [x async for x in gen]

    return asyncio.run(_collect())


def test_chunk_processor_uses_logging_obj_start_time_when_earlier(monkeypatch):
    """Caller's start_time is captured at the streaming iterator constructor,
    which runs after the upstream HTTP response is already in. The logging
    object's start_time reflects when the client request entered the proxy
    — if it's earlier, it should win.
    """
    earlier = datetime(2026, 1, 1, 12, 0, 0)
    later = earlier + timedelta(seconds=1)

    logging_obj = _build_logging_obj(true_start=earlier)

    response = asyncio.run(_build_response_with_chunks([b"chunk-1", b"chunk-2"]))
    captured_start_time = {}

    async def fake_log_streaming_request(*args, **kwargs):
        captured_start_time["start_time"] = kwargs.get("start_time")

    monkeypatch.setattr(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        AsyncMock(side_effect=fake_log_streaming_request),
    )

    gen = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={},
        litellm_logging_obj=logging_obj,
        endpoint_type=MagicMock(),
        start_time=later,
        passthrough_success_handler_obj=MagicMock(),
        url_route="/v1/messages",
    )
    _drain(gen)

    assert captured_start_time["start_time"] == earlier, (
        "Handler must override the later caller-supplied start_time with the "
        "earlier logging-obj start_time."
    )


def test_chunk_processor_keeps_caller_start_time_when_earlier(monkeypatch):
    """When the caller-supplied start_time is already earlier than the
    logging-obj's start_time (atypical but possible if logging was
    re-stamped), don't pull it forward."""
    earlier = datetime(2026, 1, 1, 12, 0, 0)
    later = earlier + timedelta(seconds=1)

    logging_obj = _build_logging_obj(true_start=later)

    response = asyncio.run(_build_response_with_chunks([b"chunk-1"]))
    captured_start_time = {}

    async def fake_log_streaming_request(*args, **kwargs):
        captured_start_time["start_time"] = kwargs.get("start_time")

    monkeypatch.setattr(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        AsyncMock(side_effect=fake_log_streaming_request),
    )

    gen = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={},
        litellm_logging_obj=logging_obj,
        endpoint_type=MagicMock(),
        start_time=earlier,
        passthrough_success_handler_obj=MagicMock(),
        url_route="/v1/messages",
    )
    _drain(gen)

    assert (
        captured_start_time["start_time"] == earlier
    ), "Caller-supplied start_time should win when it's already the earlier value."


def test_chunk_processor_records_completion_start_on_first_chunk(monkeypatch):
    """First chunk arrival populates litellm_logging_obj.completion_start_time
    so SpendLogs.completionStartTime reflects real TTFT instead of falling
    back to end_time."""
    logging_obj = _build_logging_obj(
        true_start=datetime(2026, 1, 1, 12, 0, 0),
        completion_start_time=None,
    )

    response = asyncio.run(_build_response_with_chunks([b"chunk-1", b"chunk-2"]))

    monkeypatch.setattr(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        AsyncMock(),
    )

    gen = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={},
        litellm_logging_obj=logging_obj,
        endpoint_type=MagicMock(),
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/v1/messages",
    )
    _drain(gen)

    logging_obj._update_completion_start_time.assert_called_once()
    assert isinstance(logging_obj.completion_start_time, datetime)


def test_chunk_processor_does_not_overwrite_existing_completion_start(monkeypatch):
    """If a downstream layer has already set completion_start_time
    (e.g. wrapper iterator), the handler must not overwrite it."""
    pre_set = datetime(2026, 1, 1, 12, 0, 5)
    logging_obj = _build_logging_obj(
        true_start=datetime(2026, 1, 1, 12, 0, 0),
        completion_start_time=pre_set,
    )

    response = asyncio.run(_build_response_with_chunks([b"chunk-1"]))

    monkeypatch.setattr(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        AsyncMock(),
    )

    gen = PassThroughStreamingHandler.chunk_processor(
        response=response,
        request_body={},
        litellm_logging_obj=logging_obj,
        endpoint_type=MagicMock(),
        start_time=datetime(2026, 1, 1, 12, 0, 0),
        passthrough_success_handler_obj=MagicMock(),
        url_route="/v1/messages",
    )
    _drain(gen)

    logging_obj._update_completion_start_time.assert_not_called()
    assert logging_obj.completion_start_time == pre_set
