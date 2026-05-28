"""
LIT-3407 regression tests for DataDog 413 (Payload Too Large) handling.

Bug summary
-----------
async_send_compressed_data sits on top of the LiteLLM async httpx handler,
which converts any 4xx into a MaskedHTTPStatusError before returning. So the
production codepath for a 413 from Datadog is an *exception*, not a Response
with status_code == 413 - the old `if response.status_code == 413` branch
inside async_send_batch was unreachable, the bare `except` re-queued the
whole oversized batch, and every flush interval replayed the same 413
forever.

These tests pin the new behaviour:
    * Masked 413 exception -> recursive halve-and-retry, single-event drop.
    * Direct 413 Response  -> same recursive halve-and-retry, single-event drop.
    * Partial split: first half 413, second half 202 -> first half is dropped
      after a 1-event recurse, second half is delivered in one call.
    * Non-413 transport errors -> events are preserved for retry.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import Request, Response

from litellm.integrations.datadog.datadog import DataDogLogger, _is_413_error
from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError
from litellm.types.integrations.datadog import DatadogPayload


@pytest.fixture
def datadog_env(monkeypatch):
    monkeypatch.setenv("DD_API_KEY", "test_api_key")
    monkeypatch.setenv("DD_SITE", "test.datadoghq.com")


def _payloads(n):
    return [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message=f'{{"event": {i}}}',
            service="svc",
            status="info",
        )
        for i in range(n)
    ]


def _masked_413():
    """Construct the MaskedHTTPStatusError(413) the production httpx path raises."""
    req = Request("POST", "https://intake-test.datadoghq.com/api/v2/logs")
    resp = Response(413, request=req, text="Payload Too Large")
    orig = httpx.HTTPStatusError("413 Payload Too Large", request=req, response=resp)
    return MaskedHTTPStatusError(orig, message="Payload Too Large", text="Payload Too Large")


# --------------------------------------------------------------------------- #
# _is_413_error                                                               #
# --------------------------------------------------------------------------- #
def test_is_413_error_detects_masked_httpx_exception():
    assert _is_413_error(_masked_413()) is True


def test_is_413_error_returns_false_for_non_413_status():
    req = Request("POST", "https://example.com")
    resp = Response(500, request=req, text="boom")
    exc = httpx.HTTPStatusError("500", request=req, response=resp)
    assert _is_413_error(exc) is False


def test_is_413_error_returns_false_for_unrelated_exception():
    assert _is_413_error(RuntimeError("not http")) is False


# --------------------------------------------------------------------------- #
# Masked-exception 413 path (the actual production path)                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_masked_413_exception_triggers_chunking_and_drop(datadog_env):
    """
    Mirror production: async_send_compressed_data raises MaskedHTTPStatusError(413)
    instead of returning a Response. Before LIT-3407 this raise fell through to
    the bare `except` and the whole batch was re-queued -> infinite retry loop.
    The fix splits the batch recursively until single events are reached, then
    drops those single events with a verbose error log.
    """
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(4)
    logger.async_send_compressed_data = AsyncMock(side_effect=_masked_413())

    await logger.async_send_batch()

    # Recursive halving of 4 events: 1 full + 2 halves + 4 single events = 7 calls.
    assert logger.async_send_compressed_data.await_count == 7
    # Queue is fully drained - no infinite retry.
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_masked_413_does_not_loop_on_subsequent_flushes(datadog_env):
    """
    Sanity: after a 413 flush drains the queue, subsequent flushes do not
    re-flush the same oversize batch (which was the LIT-3407 symptom).
    """
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(2)
    logger.async_send_compressed_data = AsyncMock(side_effect=_masked_413())

    await logger.async_send_batch()
    first_calls = logger.async_send_compressed_data.await_count
    assert logger.log_queue == []

    # Nothing new in queue -> next flush should be a no-op send.
    await logger.async_send_batch()
    assert logger.async_send_compressed_data.await_count == first_calls


@pytest.mark.asyncio
async def test_413_split_partial_success_only_failing_slice_chunks(datadog_env):
    """
    First-half 413, second-half 202.
    First half (events 0, 1) is split into [0] and [1] and dropped.
    Second half (events 2, 3) is delivered in a single call.
    """
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(4)

    delivered_batches = []
    call_count = {"n": 0}

    async def fake_send(batch):
        call_count["n"] += 1
        n = call_count["n"]
        if n == 1:
            # Full batch of 4 -> 413 (overall too big).
            raise _masked_413()
        if n == 2:
            # First half [event0, event1] -> 413.
            raise _masked_413()
        if n == 3:
            # event 0 alone -> 413 (dropped).
            raise _masked_413()
        if n == 4:
            # event 1 alone -> 413 (dropped).
            raise _masked_413()
        if n == 5:
            # Second half [event2, event3] -> 202 (delivered).
            delivered_batches.append([p["message"] for p in batch])
            return Response(
                202, request=Request("POST", "https://example.com"), text="Accepted"
            )
        raise AssertionError(f"unexpected call #{n}")

    logger.async_send_compressed_data = AsyncMock(side_effect=fake_send)

    await logger.async_send_batch()

    assert logger.async_send_compressed_data.await_count == 5
    assert logger.log_queue == []
    assert delivered_batches == [['{"event": 2}', '{"event": 3}']]


@pytest.mark.asyncio
async def test_non_413_transport_error_preserves_full_batch(datadog_env):
    """Non-413 errors must still re-queue the entire slice (current contract)."""
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(3)
    logger.async_send_compressed_data = AsyncMock(side_effect=RuntimeError("boom"))

    await logger.async_send_batch()

    assert logger.async_send_compressed_data.await_count == 1
    assert [p["message"] for p in logger.log_queue] == [
        '{"event": 0}',
        '{"event": 1}',
        '{"event": 2}',
    ]


@pytest.mark.asyncio
async def test_413_response_without_exception_also_chunks(datadog_env):
    """
    Defensive coverage for clients that return a 413 Response instead of
    raising (sync clients, mocks, future httpx refactors).
    """
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(2)
    logger.async_send_compressed_data = AsyncMock(
        return_value=Response(
            413, request=Request("POST", "https://example.com"), text="too big"
        )
    )

    await logger.async_send_batch()

    # 1 full batch + 2 single-event retries = 3 calls
    assert logger.async_send_compressed_data.await_count == 3
    assert logger.log_queue == []
