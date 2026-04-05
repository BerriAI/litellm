import asyncio
import os
import sys

import aiohttp
import aiohttp.client_exceptions
import aiohttp.http_exceptions
import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

from litellm.llms.custom_httpx import aiohttp_transport as aiohttp_transport_module
from litellm.llms.custom_httpx.aiohttp_transport import (
    AiohttpResponseStream,
    AiohttpTransport,
    LiteLLMAiohttpTransport,
)


@pytest.mark.asyncio
async def test_aclose_does_not_close_shared_session():
    """Test that aclose() does not close a session it does not own (shared session)."""
    session = aiohttp.ClientSession()
    try:
        transport = LiteLLMAiohttpTransport(client=session, owns_session=False)
        await transport.aclose()
        assert not session.closed, "Shared session should not be closed by transport"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_aclose_closes_owned_session():
    """Test that aclose() closes a session it owns."""
    session = aiohttp.ClientSession()
    transport = LiteLLMAiohttpTransport(client=session, owns_session=True)
    await transport.aclose()
    assert session.closed, "Owned session should be closed by transport"


@pytest.mark.asyncio
async def test_owns_session_defaults_to_true():
    """Test that owns_session defaults to True for backwards compatibility."""
    session = aiohttp.ClientSession()
    transport = AiohttpTransport(client=session)
    assert transport._owns_session is True
    await transport.aclose()
    assert session.closed


class MockAiohttpResponse:
    """Mock aiohttp ClientResponse for testing"""

    def __init__(
        self,
        status=200,
        headers=None,
        content_chunks=None,
        exception_to_raise=None,
        exception_at_chunk=None,
    ):
        self.status = status
        self.headers = headers or {}
        self.content = MockContent(content_chunks, exception_to_raise, exception_at_chunk)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockContent:
    """Mock aiohttp response content for testing"""

    def __init__(self, chunks=None, exception_to_raise=None, exception_at_chunk=None):
        self.chunks = chunks or [b"chunk1", b"chunk2", b"chunk3"]
        self.exception_to_raise = exception_to_raise
        self.exception_at_chunk = exception_at_chunk or (len(self.chunks) - 1)
        self.chunk_index = 0

    async def iter_chunked(self, chunk_size):
        for i, chunk in enumerate(self.chunks):
            if self.exception_to_raise and i == self.exception_at_chunk:
                # Raise exception at specified chunk to simulate partial transfer
                raise self.exception_to_raise
            yield chunk


@pytest.mark.asyncio
async def test_aiohttp_response_stream_normal_flow():
    """Test normal flow of AiohttpResponseStream without exceptions"""
    mock_response = MockAiohttpResponse(content_chunks=[b"hello", b"world", b"test"])

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    chunks = []

    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [b"hello", b"world", b"test"]


@pytest.mark.asyncio
async def test_transfer_encoding_error_no_httpx_read_error():
    """Test that TransferEncodingError doesn't get converted to httpx.ReadError"""

    # Create a TransferEncodingError wrapped in ClientPayloadError (like in real scenarios)
    transfer_error = aiohttp.http_exceptions.TransferEncodingError(
        message="400, message: Not enough data for satisfy transfer length header."
    )

    # Wrap it in ClientPayloadError as aiohttp does
    client_payload_error = aiohttp.ClientPayloadError("Response payload is not completed")
    client_payload_error.__cause__ = transfer_error

    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2", b"chunk3"],
        exception_to_raise=client_payload_error,
        exception_at_chunk=1,  # Error occurs at chunk 1
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    received_chunks = []

    # This should NOT raise httpx.ReadError or any other exception
    # It should handle the error gracefully and just return what was received
    async for chunk in stream:
        received_chunks.append(chunk)
    print(f"received_chunks: {received_chunks}")

    # Should have received the first chunk before the error
    assert received_chunks == [b"chunk1"]
    assert len(received_chunks) == 1


@pytest.mark.asyncio
async def test_client_payload_error_graceful_handling():
    """Test that ClientPayloadError is handled gracefully without stacktrace"""
    # Create a ClientPayloadError directly
    client_error = aiohttp.client_exceptions.ClientPayloadError("Response payload is not completed")

    mock_response = MockAiohttpResponse(
        content_chunks=[b"data1", b"data2", b"data3"],
        exception_to_raise=client_error,
        exception_at_chunk=2,  # Error occurs at chunk 2
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    received_chunks = []

    # This should handle the error gracefully without raising
    async for chunk in stream:
        received_chunks.append(chunk)

    # Should have received chunks before the error
    assert received_chunks == [b"data1", b"data2"]
    assert len(received_chunks) == 2


@pytest.mark.asyncio
async def test_unknown_aiohttp_exception_gets_mapped():
    """Test that unknown aiohttp exceptions still get mapped to httpx exceptions"""
    # Create an aiohttp exception that's not specifically handled
    # Using InvalidURL which should map to httpx.InvalidURL
    invalid_url_error = aiohttp.InvalidURL("Invalid URL format")

    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2"],
        exception_to_raise=invalid_url_error,
        exception_at_chunk=0,  # Error occurs immediately
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore

    # This should raise httpx.InvalidURL (mapped from aiohttp.InvalidURL)
    with pytest.raises(httpx.InvalidURL):
        async for chunk in stream:
            pass


@pytest.mark.asyncio
async def test_timeout_exception_gets_mapped():
    """Test that aiohttp timeout exceptions get mapped to httpx timeout exceptions"""
    # Create an aiohttp timeout exception
    timeout_error = aiohttp.ServerTimeoutError("Server timeout")

    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2"],
        exception_to_raise=timeout_error,
        exception_at_chunk=1,  # Error occurs at chunk 1
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    received_chunks = []

    # This should raise httpx.TimeoutException (mapped from aiohttp.ServerTimeoutError)
    with pytest.raises(httpx.TimeoutException):
        async for chunk in stream:
            received_chunks.append(chunk)

    # Should have received the first chunk before the error
    assert received_chunks == [b"chunk1"]


@pytest.mark.asyncio
async def test_handle_async_request_uses_env_proxy(monkeypatch):
    """Aiohttp transport should honor HTTP(S)_PROXY env vars"""
    proxy_url = "http://proxy.local:3128"
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("http_proxy", proxy_url)
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("https_proxy", proxy_url)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRUST_ENV", raising=False)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {"http": proxy_url, "https": proxy_url})
    monkeypatch.setattr("urllib.request.proxy_bypass", lambda host: False)

    captured = {}

    class FakeSession:
        def __init__(self):
            self.closed = False
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, *args, **kwargs):
            captured["proxy"] = kwargs.get("proxy")

            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, size):
                            yield b""

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())  # type: ignore
    request = httpx.Request("GET", "http://example.com")
    await transport.handle_async_request(request)

    assert captured["proxy"] == proxy_url


@pytest.mark.asyncio
async def test_handle_async_request_uses_env_proxy_per_url(monkeypatch):
    """Aiohttp transport should honor HTTP(S)_PROXY env vars unless NO_PROXY matches"""
    proxy_url = "http://proxy.local:3128"
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("http_proxy", proxy_url)
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("https_proxy", proxy_url)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRUST_ENV", raising=False)

    request_count = 0
    proxied_count = 0

    class FakeSession:
        def __init__(self):
            self.closed = False
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, *args, **kwargs):
            nonlocal request_count
            nonlocal proxied_count
            request_count += 1

            if kwargs.get("proxy") is not None:
                proxied_count += 1

            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, size):
                            yield b""

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())  # type: ignore
    request = httpx.Request("GET", "http://example.com")
    await transport.handle_async_request(request)

    request = httpx.Request("GET", "http://foo.com")
    await transport.handle_async_request(request)

    assert request_count == 2
    assert proxied_count == 1


@pytest.mark.asyncio
async def test_handle_async_request_proxy_cache_per_host(monkeypatch):
    """Aiohttp transport should only cache a proxy per host rather than full URL"""
    proxy_url = "http://proxy.local:3128"
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("http_proxy", proxy_url)
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("https_proxy", proxy_url)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRUST_ENV", raising=False)

    def factory():
        return _make_mock_session()

    transport = LiteLLMAiohttpTransport(client=factory)  # type: ignore
    request = httpx.Request("GET", "http://foo.com/path1")
    await transport.handle_async_request(request)

    request = httpx.Request("GET", "http://foo.com/path2")
    await transport.handle_async_request(request)

    assert len(transport.proxy_cache) == 1


def _make_mock_response(should_fail=False, fail_count={"count": 0}):
    """Helper to create a mock aiohttp response"""

    class MockResp:
        status = 200
        headers = {}

        async def __aenter__(self):
            if should_fail and fail_count["count"] < 1:
                fail_count["count"] += 1
                raise RuntimeError("Session is closed")
            return self

        async def __aexit__(self, *args):
            pass

        @property
        def content(self):
            class C:
                async def iter_chunked(self, size):
                    yield b"test"

            return C()

    return MockResp()


@pytest.mark.asyncio
async def test_handle_async_request_sock_read_timeout_triggers():
    """
    Ensure that LiteLLMAiohttpTransport raises httpx.TimeoutException
    when the sock_read timeout duration elapses (individual read operation timeout).
    This is the correct behavior for stream_timeout - it should timeout on slow reads,
    not on the total duration of the stream.
    """
    import asyncio
    from aiohttp import web

    async def slow_handler(request):
        # Sleep longer than the sock_read timeout
        await asyncio.sleep(0.3)
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", slow_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]

    def factory():
        return aiohttp.ClientSession()

    transport = LiteLLMAiohttpTransport(client=factory)  # type: ignore

    request = httpx.Request("GET", f"http://127.0.0.1:{port}/")

    # Set a short sock_read timeout - this should trigger
    # Note: total timeout is NOT set, allowing long-running streams
    request.extensions["timeout"] = {
        "connect": 5.0,
        "read": 0.1,  # Short timeout for individual reads
        "pool": 5.0,
    }

    try:
        with pytest.raises(httpx.TimeoutException):
            await transport.handle_async_request(request)
    finally:
        await transport.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_handle_async_request_streaming_does_not_timeout_on_total_duration():
    """
    Ensure that LiteLLMAiohttpTransport does NOT timeout on long-running
    streaming responses as long as individual chunks arrive within the sock_read timeout.
    This is the fix for issue #19184 - stream_timeout should only control the timeout
    for individual chunks, not the total stream duration.
    """
    import asyncio
    from aiohttp import web

    async def streaming_handler(request):
        # Simulate a streaming response that takes longer than a single timeout
        # but each chunk arrives quickly
        response = web.StreamResponse()
        await response.prepare(request)
        
        # Send 5 chunks over 0.5 seconds total (0.1s between chunks)
        for i in range(5):
            await asyncio.sleep(0.05)  # Less than sock_read timeout
            await response.write(f"chunk{i}\n".encode())
        
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/stream", streaming_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]

    def factory():
        return aiohttp.ClientSession()

    transport = LiteLLMAiohttpTransport(client=factory)  # type: ignore

    request = httpx.Request("GET", f"http://127.0.0.1:{port}/stream")

    # Set sock_read timeout that's longer than individual chunk delays
    # but shorter than total stream duration
    # Total duration: ~0.25s, sock_read timeout: 0.15s per chunk
    # This should NOT timeout because each chunk arrives within 0.15s
    request.extensions["timeout"] = {
        "connect": 5.0,
        "read": 0.15,  # Timeout for individual reads
        "pool": 5.0,
        # Note: total is NOT set - this is the fix!
    }

    try:
        # This should succeed without timing out
        response = await transport.handle_async_request(request)
        assert response.status_code == 200
        
        # Read the streaming response
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        
        # Verify we got all chunks
        full_response = b"".join(chunks).decode()
        assert "chunk0" in full_response
        assert "chunk4" in full_response
    finally:
        await transport.aclose()
        await runner.cleanup()


def _make_mock_session(closed=False):
    """Helper to create a mock aiohttp session"""

    class MockSession:
        def __init__(self):
            self.closed = closed
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, *args, **kwargs):
            return _make_mock_response()

    return MockSession()


@pytest.mark.asyncio
async def test_handle_closed_session_before_request():
    """Test that closed sessions are detected and recreated"""
    counts = {"sessions": 0}

    def factory():
        counts["sessions"] += 1
        return _make_mock_session(closed=counts["sessions"] == 1)

    transport = LiteLLMAiohttpTransport(client=factory)  # type: ignore
    response = await transport.handle_async_request(httpx.Request("GET", "http://example.com"))

    assert counts["sessions"] == 2  # Created 2 sessions: closed one, then open one
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_schedule_session_close_handles_missing_running_loop(monkeypatch):
    """If create_task cannot run, the close coroutine should still be explicitly closed."""

    coro_closed = {"value": False}

    class MockCloseCoro:
        def close(self):
            coro_closed["value"] = True

    class MockSession:
        closed = False

        def close(self):
            return MockCloseCoro()

    session = MockSession()

    transport = LiteLLMAiohttpTransport(client=session)
    LiteLLMAiohttpTransport._background_close_tasks.clear()

    def raise_runtime_error(_coro):
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "create_task", raise_runtime_error)

    try:
        transport._schedule_session_close(session)

        assert coro_closed["value"] is True
    finally:
        LiteLLMAiohttpTransport._background_close_tasks.clear()


@pytest.mark.asyncio
async def test_discard_background_close_task_logs_failed_close(monkeypatch):
    """Failed background close tasks should be removed and logged."""

    debug_messages = []

    def capture_debug(message, *args, **kwargs):
        if args:
            debug_messages.append(message % args)
        else:
            debug_messages.append(message)

    async def fail_close():
        raise RuntimeError("close failed")

    monkeypatch.setattr(aiohttp_transport_module.verbose_logger, "debug", capture_debug)

    task = asyncio.create_task(fail_close())
    LiteLLMAiohttpTransport._background_close_tasks.clear()
    LiteLLMAiohttpTransport._background_close_tasks.add(task)

    try:
        await asyncio.gather(task, return_exceptions=True)
        LiteLLMAiohttpTransport._discard_background_close_task(task)

        assert task not in LiteLLMAiohttpTransport._background_close_tasks
        assert any("Error closing old session in background task: close failed" in msg for msg in debug_messages)
    finally:
        LiteLLMAiohttpTransport._background_close_tasks.clear()


@pytest.mark.asyncio
async def test_schedule_session_close_prunes_stale_done_tasks():
    """Scheduling a new close should prune completed stale tasks from the tracking set."""

    async def completed_task():
        return None

    stale_task = asyncio.create_task(completed_task())
    await stale_task

    session = aiohttp.ClientSession()
    transport = LiteLLMAiohttpTransport(client=session)

    LiteLLMAiohttpTransport._background_close_tasks.clear()
    LiteLLMAiohttpTransport._background_close_tasks.add(stale_task)

    try:
        transport._schedule_session_close(session)
        pending_tasks = list(LiteLLMAiohttpTransport._background_close_tasks)
        if pending_tasks:
            await asyncio.gather(*pending_tasks)

        assert stale_task not in LiteLLMAiohttpTransport._background_close_tasks
        assert LiteLLMAiohttpTransport._background_close_tasks == set()
        assert session.closed is True
    finally:
        LiteLLMAiohttpTransport._background_close_tasks.clear()
        if not session.closed:
            await session.close()


@pytest.mark.asyncio
async def test_get_valid_client_session_closes_old_session_on_loop_check_runtime_error(monkeypatch):
    """Loop-check exceptions should still attempt to close the old session before recreation."""

    old_session = aiohttp.ClientSession()
    new_session = aiohttp.ClientSession()
    scheduled_sessions = []

    transport = LiteLLMAiohttpTransport(client=lambda: new_session)
    transport.client = old_session

    def mock_schedule(session):
        scheduled_sessions.append(session)

    monkeypatch.setattr(
        transport,
        "_schedule_session_close",
        mock_schedule,
    )

    def raise_runtime_error():
        raise RuntimeError("loop unavailable")

    monkeypatch.setattr(asyncio, "get_running_loop", raise_runtime_error)

    try:
        session = transport._get_valid_client_session()

        assert session is new_session
        assert scheduled_sessions == [old_session]
    finally:
        await old_session.close()
        await new_session.close()


@pytest.mark.asyncio
async def test_handle_session_closed_during_request():
    """Test that sessions closed during request are handled with retry"""
    counts = {"sessions": 0}
    scheduled_sessions = []

    def factory():
        counts["sessions"] += 1
        return aiohttp.ClientSession()

    transport = LiteLLMAiohttpTransport(client=factory)  # type: ignore
    original_schedule = transport._schedule_session_close
    call_count = {"value": 0}

    def tracked_schedule(session):
        scheduled_sessions.append(session)
        return original_schedule(session)

    transport._schedule_session_close = tracked_schedule  # type: ignore[method-assign]

    async def tracked_make_request(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("Session is closed")
        return await _make_mock_response().__aenter__()

    transport._make_aiohttp_request = tracked_make_request  # type: ignore[method-assign]

    try:
        response = await transport.handle_async_request(httpx.Request("GET", "http://example.com"))

        assert call_count["value"] == 2  # First request failed, second succeeded
        assert counts["sessions"] == 2  # Created 2 sessions for retry
        assert response.status_code == 200
        assert len(scheduled_sessions) == 1
        assert scheduled_sessions[0].closed is False
    finally:
        if isinstance(transport.client, aiohttp.ClientSession) and not transport.client.closed:
            await transport.client.close()
