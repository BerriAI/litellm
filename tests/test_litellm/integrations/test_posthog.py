"""
Regression tests for GH #38904: the PostHog batch callback logger silently
lost events in two independent ways:

1. ``flush_queue()`` cleared the *entire* live queue after a successful send,
   including events appended by concurrent requests while the batch POST was
   in flight (fixed by opting in to
   ``CustomBatchLogger.preserve_events_added_during_flush``).
2. ``PostHogLogger.async_send_batch()`` caught send failures and only logged
   them instead of re-raising, so the base ``flush_queue()`` treated a failed
   send as success and cleared the queue anyway instead of preserving events
   for retry.

Mirrors the equivalent coverage in ``tests/test_litellm/integrations/test_rubrik.py``.
"""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from litellm.integrations.posthog import PostHogLogger


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"POSTHOG_API_KEY": "test-api-key"}, clear=False):
        yield


@pytest.fixture
def handler(mock_env):
    return PostHogLogger()


def _queue_item(msg: str) -> dict:
    return {
        "event": {"event": "$ai_generation", "properties": {"msg": msg}},
        "api_key": "test-api-key",
        "api_url": "https://us.i.posthog.com",
    }


@pytest.mark.asyncio
class TestPostHogBatchFlush:
    async def test_flush_queue_sends_batch_and_drains_on_success(self, handler):
        handler.log_queue = [_queue_item("a"), _queue_item("b")]
        handler.flush_lock = asyncio.Lock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        handler.async_client = AsyncMock()
        handler.async_client.post = AsyncMock(return_value=mock_response)

        await handler.flush_queue()

        handler.async_client.post.assert_called_once()
        assert handler.log_queue == []

    async def test_flush_queue_preserves_events_added_during_send(self, handler):
        """Regression for bug 1: an event queued while the batch POST is
        in flight must survive the flush, not be wiped by the post-send
        ``self.log_queue.clear()``."""
        handler.log_queue = [_queue_item("a"), _queue_item("b")]
        handler.flush_lock = asyncio.Lock()

        async def mock_post(*_args, **_kwargs):
            handler.log_queue.append(_queue_item("c"))
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            return mock_response

        handler.async_client = AsyncMock()
        handler.async_client.post = mock_post

        await handler.flush_queue()

        assert len(handler.log_queue) == 1
        assert handler.log_queue[0]["event"]["properties"]["msg"] == "c"

    async def test_async_send_batch_does_not_drain_events(self, handler):
        """``async_send_batch`` itself must never mutate the queue - queue
        draining is ``flush_queue``'s job, using the pre-send length so
        concurrently appended events are preserved."""
        handler.log_queue = [_queue_item("a"), _queue_item("b")]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        handler.async_client = AsyncMock()
        handler.async_client.post = AsyncMock(return_value=mock_response)

        await handler.async_send_batch()

        assert len(handler.log_queue) == 2

    async def test_async_send_batch_raises_on_http_error(self, handler):
        """Regression for bug 2: a failed send must propagate instead of
        being swallowed, so the caller knows not to treat it as delivered."""
        handler.log_queue = [_queue_item("a")]

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("err", request=Mock(), response=mock_response)
        )
        handler.async_client = AsyncMock()
        handler.async_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPStatusError):
            await handler.async_send_batch()

        # async_send_batch itself still must not have drained the queue.
        assert handler.log_queue == [_queue_item("a")]

    async def test_flush_queue_preserves_events_on_http_error(self, handler):
        """End-to-end: a failed batch send must leave the original events in
        the queue for retry on the next flush, not silently drop them."""
        handler.log_queue = [_queue_item("a"), _queue_item("b")]
        handler.flush_lock = asyncio.Lock()

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("err", request=Mock(), response=mock_response)
        )
        handler.async_client = AsyncMock()
        handler.async_client.post = AsyncMock(return_value=mock_response)

        await handler.flush_queue()

        assert handler.log_queue == [_queue_item("a"), _queue_item("b")]

    async def test_flush_queue_preserves_events_on_network_error(self, handler):
        """Network/timeout errors must also preserve the in-flight events."""
        handler.log_queue = [_queue_item("a")]
        handler.flush_lock = asyncio.Lock()

        handler.async_client = AsyncMock()
        handler.async_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        await handler.flush_queue()

        assert handler.log_queue == [_queue_item("a")]

    async def test_async_send_batch_noop_when_queue_empty(self, handler):
        """An empty queue must not attempt any network call."""
        handler.log_queue = []
        handler.async_client = AsyncMock()

        await handler.async_send_batch()

        handler.async_client.post.assert_not_called()

    async def test_async_send_batch_mock_mode_logs_and_sends(self, handler):
        """Mock mode takes the same send path, just with extra debug logging -
        exercise it so both mock-mode branches (pre-send and post-success) run."""
        handler.is_mock_mode = True
        handler.log_queue = [_queue_item("a")]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        handler.async_client = AsyncMock()
        handler.async_client.post = AsyncMock(return_value=mock_response)

        await handler.async_send_batch()

        handler.async_client.post.assert_called_once()

    async def test_flush_queue_preserves_events_added_during_failed_send(self, handler):
        """Combined case: a send that both fails AND has a concurrent append
        mid-flight must preserve both the original snapshot and the new event."""
        handler.log_queue = [_queue_item("a"), _queue_item("b")]
        handler.flush_lock = asyncio.Lock()

        async def mock_post(*_args, **_kwargs):
            handler.log_queue.append(_queue_item("c"))
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "boom"
            mock_response.raise_for_status = Mock(
                side_effect=httpx.HTTPStatusError("err", request=Mock(), response=mock_response)
            )
            return mock_response

        handler.async_client = AsyncMock()
        handler.async_client.post = mock_post

        await handler.flush_queue()

        assert len(handler.log_queue) == 3
        assert [item["event"]["properties"]["msg"] for item in handler.log_queue] == [
            "a",
            "b",
            "c",
        ]
