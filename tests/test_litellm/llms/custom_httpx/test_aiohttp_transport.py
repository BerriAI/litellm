import asyncio
import concurrent.futures
import os
import sys

import aiohttp
import aiohttp.client_exceptions
import aiohttp.http_exceptions
import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

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
        self.closed = False
        self.content = MockContent(
            content_chunks, exception_to_raise, exception_at_chunk
        )

    def close(self):
        self.closed = True

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
    client_payload_error = aiohttp.ClientPayloadError(
        "Response payload is not completed"
    )
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
    client_error = aiohttp.client_exceptions.ClientPayloadError(
        "Response payload is not completed"
    )

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
    monkeypatch.setattr(
        "urllib.request.getproxies", lambda: {"http": proxy_url, "https": proxy_url}
    )
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
async def test_handle_async_request_empty_body_sends_no_data():
    """
    A bodyless request (e.g. DELETE /responses/{id}) must reach aiohttp with
    data=None. Passing the empty `b""` httpx content makes aiohttp attach a
    `Content-Type: application/octet-stream` header, which providers like
    OpenAI reject with `unsupported_content_type`.
    """
    captured = {}

    class FakeSession:
        def __init__(self):
            self.closed = False
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, *args, **kwargs):
            captured["data"] = kwargs.get("data")

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

    empty_request = httpx.Request("DELETE", "http://example.com/responses/resp_123")
    await transport.handle_async_request(empty_request)
    assert captured["data"] is None

    body_request = httpx.Request(
        "POST", "http://example.com/responses", json={"input": "ping"}
    )
    await transport.handle_async_request(body_request)
    assert captured["data"] == body_request.content
    assert captured["data"]


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
    response = await transport.handle_async_request(
        httpx.Request("GET", "http://example.com")
    )

    assert counts["sessions"] == 2  # Created 2 sessions: closed one, then open one
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_handle_session_closed_during_request():
    """Test that sessions closed during request are handled with retry"""
    counts = {"sessions": 0, "requests": 0}
    fail_count = {"count": 0}

    class MockSession:
        def __init__(self):
            self.closed = False
            try:
                self._loop = __import__("asyncio").get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, *args, **kwargs):
            counts["requests"] += 1
            return _make_mock_response(should_fail=True, fail_count=fail_count)

    def factory():
        counts["sessions"] += 1
        return MockSession()

    transport = LiteLLMAiohttpTransport(client=factory)  # type: ignore
    response = await transport.handle_async_request(
        httpx.Request("GET", "http://example.com")
    )

    assert counts["requests"] == 2  # First request failed, second succeeded
    assert counts["sessions"] == 2  # Created 2 sessions for retry
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_response_stream_closes_response_on_error():
    """
    Regression test for #30192: when body iteration ends with an error, the
    underlying aiohttp response must be closed so its connector slot is
    released. Leaked slots exhaust the pool and every later request times
    out (408) until the proxy restarts, even after the backend recovers.
    """
    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2"],
        exception_to_raise=aiohttp.ServerTimeoutError("read timeout"),
        exception_at_chunk=1,
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    with pytest.raises(httpx.TimeoutException):
        async for _ in stream:
            pass

    assert mock_response.closed is True


@pytest.mark.asyncio
async def test_response_stream_closes_response_on_cancellation():
    """
    Regression test for #30192: a task cancelled mid-stream (e.g. the caller
    disconnects during a traffic spike) must not leak its aiohttp connection.
    """
    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2", b"chunk3"],
        exception_to_raise=asyncio.CancelledError(),
        exception_at_chunk=1,
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    with pytest.raises(asyncio.CancelledError):
        async for _ in stream:
            pass

    assert mock_response.closed is True


@pytest.mark.asyncio
async def test_response_stream_closes_response_on_generator_exit():
    """
    Regression test for #30192: when the consumer stops iterating early and the
    stream generator is closed (GeneratorExit), the underlying aiohttp response
    must still be closed so its connector slot is released.
    """
    mock_response = MockAiohttpResponse(
        content_chunks=[b"chunk1", b"chunk2", b"chunk3"],
    )

    stream = AiohttpResponseStream(mock_response)  # type: ignore
    iterator = stream.__aiter__()
    assert await iterator.__anext__() == b"chunk1"
    await iterator.aclose()

    assert mock_response.closed is True


# ---------------------------------------------------------------------------
# Recycled-session leak tests (#24230)
# ---------------------------------------------------------------------------


async def _new_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession()


def _make_session_on_dead_loop() -> aiohttp.ClientSession:
    """Create a ClientSession bound to an event loop that is then closed.

    Runs in a worker thread: the caller may already be inside a running
    event loop, where a nested run_until_complete is forbidden.
    """
    import threading

    result: dict = {}

    def build() -> None:
        loop = asyncio.new_event_loop()
        try:
            result["session"] = loop.run_until_complete(_new_session())
        finally:
            loop.close()

    thread = threading.Thread(target=build)
    thread.start()
    thread.join(5)
    return result["session"]


def _flaky_get_running_loop_factory():
    """get_running_loop stand-in that fails once, then delegates.

    Reproduces #24230: a transient loop-inspection failure sends
    _get_valid_client_session into its (RuntimeError, AttributeError)
    fallback branch.
    """
    real_get_running_loop = asyncio.get_running_loop
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated loop inspection failure")
        return real_get_running_loop()

    return flaky


@pytest.mark.asyncio
async def test_fallback_recreate_closes_previous_session():
    """
    Regression test for #24230: when loop inspection fails and the fallback
    branch recreates the session, the replaced session must still be closed -
    not silently abandoned to the garbage collector.
    """
    from unittest.mock import patch

    old_session = aiohttp.ClientSession()
    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())
    transport.client = old_session

    with patch(
        "litellm.llms.custom_httpx.aiohttp_transport.asyncio.get_running_loop",
        side_effect=_flaky_get_running_loop_factory(),
    ):
        new_session = transport._get_valid_client_session()

    try:
        assert new_session is not old_session
        for _ in range(3):
            await asyncio.sleep(0)
        assert old_session.closed, "replaced session must be closed, not leaked"
    finally:
        await new_session.close()
        if not old_session.closed:
            await old_session.close()


@pytest.mark.asyncio
async def test_replaced_session_emits_no_unclosed_warnings():
    """
    Regression test for #24230: a session replaced by the fallback branch must
    not surface "Unclosed client session" / "Unclosed connector" warnings when
    the garbage collector finalizes it.
    """
    import gc
    import warnings as warnings_mod
    from unittest.mock import patch

    old_session = aiohttp.ClientSession()
    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())
    transport.client = old_session

    with patch(
        "litellm.llms.custom_httpx.aiohttp_transport.asyncio.get_running_loop",
        side_effect=_flaky_get_running_loop_factory(),
    ):
        new_session = transport._get_valid_client_session()

    try:
        for _ in range(3):
            await asyncio.sleep(0)

        del old_session
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            gc.collect()

        unclosed = [
            str(w.message)
            for w in caught
            if "Unclosed client session" in str(w.message) or "Unclosed connector" in str(w.message)
        ]
        assert not unclosed, f"leaked session warnings: {unclosed}"
    finally:
        await new_session.close()


@pytest.mark.asyncio
async def test_dead_loop_session_closed_synchronously_on_recycle():
    """
    Regression test for #24230: a session whose event loop is already closed
    cannot run an async close anywhere. Recycling it must dispose of it
    deterministically, the session reads closed as soon as the recycle
    returns, so no finalizer warning window remains.
    """
    old_session = _make_session_on_dead_loop()
    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())
    transport.client = old_session

    new_session = transport._get_valid_client_session()

    try:
        assert new_session is not old_session
        assert old_session.closed, "session from a closed loop must be disposed synchronously at recycle"
    finally:
        await new_session.close()


@pytest.mark.asyncio
async def test_close_task_strongly_referenced_until_done():
    """
    Regression test for #24230: scheduled session-close tasks must be strongly
    referenced (and pruned on completion) so they cannot be garbage-collected
    before they run.
    """
    old_session = aiohttp.ClientSession()
    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())

    transport._close_recycled_session(old_session)

    assert LiteLLMAiohttpTransport._background_close_tasks, "close task must be strongly referenced while pending"
    for _ in range(5):
        await asyncio.sleep(0)
    assert old_session.closed
    assert not LiteLLMAiohttpTransport._background_close_tasks, "completed close tasks must be pruned from the registry"


@pytest.mark.asyncio
async def test_session_from_other_running_loop_closed_threadsafe():
    """
    Regression test for #24230: a session that belongs to a loop still running
    in another thread must be closed on its own loop (thread-safe), not driven
    from the current loop.
    """
    import threading
    import time

    ready = threading.Event()
    holder: dict = {}

    def worker() -> None:
        loop = asyncio.new_event_loop()
        holder["loop"] = loop

        async def make() -> None:
            holder["session"] = aiohttp.ClientSession()

        loop.run_until_complete(make())
        ready.set()
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert ready.wait(5), "worker loop failed to start"

    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())
    transport.client = holder["session"]

    new_session = transport._get_valid_client_session()

    try:
        deadline = time.monotonic() + 5
        while not holder["session"].closed and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert holder["session"].closed, "foreign-loop session was never closed"
    finally:
        holder["loop"].call_soon_threadsafe(holder["loop"].stop)
        thread.join(5)
        await new_session.close()


def test_threadsafe_close_done_callback_tolerates_cancelled_future():
    """
    Regression test for #24230 (review finding): when the foreign loop stops
    before the handed-off close coroutine runs, asyncio cancels the
    concurrent.futures.Future. The done-callback must return quietly instead
    of letting future.exception() raise CancelledError (a BaseException that
    escapes _invoke_callbacks and crashes the foreign loop's thread).
    """
    future: "concurrent.futures.Future[None]" = concurrent.futures.Future()
    future.cancel()

    LiteLLMAiohttpTransport._on_threadsafe_close_done(future)


@pytest.mark.asyncio
async def test_session_closed_retry_does_not_close_concurrent_replacement():
    """
    Regression test for #24230 (review finding): when the "Session is closed"
    retry fires, the handler must dispose the session that actually faulted,
    not self.client - a concurrent task may already have replaced self.client
    with a healthy session, which must stay open.
    """
    from unittest.mock import patch

    faulted_session = aiohttp.ClientSession()
    healthy_replacement = aiohttp.ClientSession()
    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())
    transport.client = faulted_session

    calls = {"n": 0}

    async def fake_make_request(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # simulate a concurrent task replacing the shared session between
            # the failed await and the exception handler
            transport.client = healthy_replacement
            raise RuntimeError("Session is closed")
        raise StopAsyncIteration("stop after retry dispatch")

    with patch.object(transport, "_make_aiohttp_request", side_effect=fake_make_request):
        with pytest.raises(Exception):
            await transport.handle_async_request(httpx.Request("GET", "http://example.com"))

    try:
        assert not healthy_replacement.closed, "concurrent replacement session must not be closed by the retry handler"
        for _ in range(3):
            await asyncio.sleep(0)
        assert faulted_session.closed, "the faulted session must be disposed"
    finally:
        await faulted_session.close()
        await healthy_replacement.close()
        new_session = transport.client
        if isinstance(new_session, aiohttp.ClientSession):
            await new_session.close()


@pytest.mark.asyncio
async def test_stopped_loop_session_disposed_synchronously_on_recycle():
    """
    Regression test for #24230 (review finding): a session whose loop is
    stopped but not yet closed cannot safely run an async close on another
    loop, and nothing will ever process a close handed to the stopped loop.
    Recycling must dispose it synchronously, like the closed-loop case.
    """
    import threading

    result: dict = {}

    def build() -> None:
        loop = asyncio.new_event_loop()

        async def make() -> None:
            result["session"] = aiohttp.ClientSession()

        loop.run_until_complete(make())
        result["loop"] = loop  # stopped, deliberately NOT closed

    thread = threading.Thread(target=build)
    thread.start()
    thread.join(5)

    old_session = result["session"]
    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())
    transport.client = old_session

    new_session = transport._get_valid_client_session()

    try:
        assert new_session is not old_session
        assert old_session.closed, "session from a stopped (not yet closed) loop must be disposed synchronously"
    finally:
        await new_session.close()
        result["loop"].close()
