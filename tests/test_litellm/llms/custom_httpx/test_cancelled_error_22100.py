"""
Tests for asyncio.CancelledError propagation through aiohttp transport/handler.

Regression tests for issue #22100:
  asyncio.CancelledError in aiohttp transport bypasses retry logic, logging,
  and exception mapping.

In Python 3.8+, asyncio.CancelledError inherits from BaseException (not
Exception), so a bare `except Exception` block will NOT catch it — it
propagates silently past all retry/logging/mapping code.  The fix adds
explicit `except asyncio.CancelledError: raise` guards in the critical
catch sites so that:

1. CancelledError is never accidentally mapped to an httpx exception.
2. CancelledError is never silently swallowed inside a retry loop.
3. Task-cancellation semantics are preserved end-to-end.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.aiohttp_transport import (
    AiohttpResponseStream,
    LiteLLMAiohttpTransport,
    map_aiohttp_exceptions,
)


# ---------------------------------------------------------------------------
# map_aiohttp_exceptions
# ---------------------------------------------------------------------------


def test_map_aiohttp_exceptions_reraises_cancelled_error():
    """map_aiohttp_exceptions must not swallow or remap CancelledError."""
    with pytest.raises(asyncio.CancelledError):
        with map_aiohttp_exceptions():
            raise asyncio.CancelledError()


def test_map_aiohttp_exceptions_still_maps_aiohttp_errors():
    """Normal aiohttp exceptions must still be mapped to httpx equivalents."""
    with pytest.raises(httpx.TimeoutException):
        with map_aiohttp_exceptions():
            raise aiohttp.ServerTimeoutError()


# ---------------------------------------------------------------------------
# AiohttpResponseStream.__aiter__
# ---------------------------------------------------------------------------


class _CancellingContent:
    """Async iterator that raises CancelledError after the first chunk."""

    async def iter_chunked(self, chunk_size: int):
        yield b"first-chunk"
        raise asyncio.CancelledError()


class _MockCancelResponse:
    content = _CancellingContent()

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_aiohttp_response_stream_propagates_cancelled_error():
    """
    AiohttpResponseStream.__aiter__ must propagate CancelledError rather than
    silently swallowing it through the bare `except Exception` branch.
    """
    stream = AiohttpResponseStream(_MockCancelResponse())  # type: ignore
    with pytest.raises(asyncio.CancelledError):
        async for _ in stream:
            pass


# ---------------------------------------------------------------------------
# LiteLLMAiohttpTransport.handle_async_request
# ---------------------------------------------------------------------------


def _make_mock_session_raising(exc: BaseException):
    """Build a fake aiohttp ClientSession whose request() raises *exc*."""

    class _FakeResp:
        async def __aenter__(self_inner):
            raise exc

        async def __aexit__(self_inner, *args):
            pass

    class _FakeSession:
        closed = False

        def __init__(self):
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, *args, **kwargs):
            return _FakeResp()

    return _FakeSession()


@pytest.mark.asyncio
async def test_handle_async_request_propagates_cancelled_error():
    """
    LiteLLMAiohttpTransport.handle_async_request must not catch CancelledError —
    it should propagate unchanged so that asyncio task-cancellation works.
    """
    session = _make_mock_session_raising(asyncio.CancelledError())
    transport = LiteLLMAiohttpTransport(client=session)  # type: ignore

    request = httpx.Request("GET", "http://example.com")
    with pytest.raises(asyncio.CancelledError):
        await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_handle_async_request_cancelled_error_not_mapped_to_httpx():
    """
    CancelledError must never be mapped to an httpx exception type.
    """
    session = _make_mock_session_raising(asyncio.CancelledError())
    transport = LiteLLMAiohttpTransport(client=session)  # type: ignore

    request = httpx.Request("GET", "http://example.com")
    try:
        await transport.handle_async_request(request)
        pytest.fail("Expected CancelledError to be raised")
    except asyncio.CancelledError:
        pass  # Correct — CancelledError propagated untouched
    except httpx.RequestError as exc:
        pytest.fail(f"CancelledError was incorrectly mapped to httpx exception: {exc}")


@pytest.mark.asyncio
async def test_cancelled_error_propagates_from_asyncio_task():
    """
    End-to-end: cancelling an asyncio.Task that awaits handle_async_request
    must result in the task finishing with CancelledError, not silently
    completing or raising an unexpected exception.
    """

    async def _slow_request():
        await asyncio.sleep(10)  # Will be cancelled before completing

    async def _patched_make_request(*args, **kwargs):
        await asyncio.sleep(10)  # Simulates a slow in-flight aiohttp request

    session = MagicMock()
    session.closed = False
    session._loop = asyncio.get_running_loop()

    transport = LiteLLMAiohttpTransport(client=session)  # type: ignore

    with patch.object(transport, "_make_aiohttp_request", side_effect=_patched_make_request):
        request = httpx.Request("GET", "http://example.com")
        task = asyncio.create_task(transport.handle_async_request(request))

        # Let the task start
        await asyncio.sleep(0)

        # Cancel it
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# BaseLLMAIOHTTPHandler._make_common_async_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_common_async_call_propagates_cancelled_error():
    """
    BaseLLMAIOHTTPHandler._make_common_async_call must not catch CancelledError
    via its `except Exception` branch — it should propagate unchanged.
    """
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    handler = BaseLLMAIOHTTPHandler()

    # Build a fake ClientSession whose post() raises CancelledError
    mock_session = MagicMock()
    mock_session.__class__ = aiohttp.ClientSession  # satisfy isinstance checks

    async def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    mock_session.post = _raise_cancelled

    mock_provider_config = MagicMock()
    mock_provider_config.max_retry_on_unprocessable_entity_error = 1

    with pytest.raises(asyncio.CancelledError):
        await handler._make_common_async_call(
            async_client_session=mock_session,
            provider_config=mock_provider_config,
            api_base="http://example.com",
            headers={},
            data={},
            timeout=30.0,
            litellm_params={},
        )


@pytest.mark.asyncio
async def test_make_common_async_call_cancelled_error_not_wrapped():
    """
    _make_common_async_call must not wrap CancelledError in a provider error
    class — it must propagate as-is.
    """
    from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler

    handler = BaseLLMAIOHTTPHandler()

    mock_session = MagicMock()
    mock_session.__class__ = aiohttp.ClientSession

    async def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    mock_session.post = _raise_cancelled

    mock_provider_config = MagicMock()
    mock_provider_config.max_retry_on_unprocessable_entity_error = 1
    # Make get_error_class return a generic Exception so we can detect if it's called
    mock_provider_config.get_error_class = MagicMock(return_value=Exception("mapped"))

    try:
        await handler._make_common_async_call(
            async_client_session=mock_session,
            provider_config=mock_provider_config,
            api_base="http://example.com",
            headers={},
            data={},
            timeout=30.0,
            litellm_params={},
        )
        pytest.fail("Expected CancelledError")
    except asyncio.CancelledError:
        # Correct: CancelledError propagated, get_error_class was NOT called
        mock_provider_config.get_error_class.assert_not_called()
    except Exception as exc:
        pytest.fail(f"CancelledError was converted to: {type(exc).__name__}: {exc}")
