import asyncio
import gc
import io
import os
import pathlib
import ssl
import threading
import weakref
from unittest.mock import MagicMock, patch

import certifi
import httpx
import pytest
from aiohttp import ClientSession, TCPConnector

import litellm
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.llms.custom_httpx.http_handler import (
    _CLIENT_REFCOUNT_WHEN_HANDLER_IS_SOLE_REFERRER,
    AsyncHTTPHandler,
    HTTPHandler,
    MaskedHTTPStatusError,
    _get_httpx_client,
    get_ssl_configuration,
)


@pytest.mark.asyncio
async def test_async_post_streaming_status_error_should_not_wait_forever_for_body(
    monkeypatch,
):
    """
    Vertex Anthropic streamRawPredict can return a pre-stream 4xx where the
    streamed error body never terminates. The handler must still surface the
    status promptly instead of blocking the downstream client.
    """

    class HangingErrorStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            if False:
                yield b""

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            request=request,
            headers={"content-type": "application/json"},
            stream=HangingErrorStream(),
        )

    monkeypatch.setattr(
        "litellm.llms.custom_httpx.http_handler._STREAMING_ERROR_BODY_READ_TIMEOUT_SECONDS",
        0.01,
    )

    litellm_handler = AsyncHTTPHandler()
    await litellm_handler.client.aclose()
    litellm_handler.client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    try:
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            await asyncio.wait_for(
                litellm_handler.post(
                    "https://vertex.example/streamRawPredict",
                    stream=True,
                ),
                timeout=0.2,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.response.status_code == 400
    finally:
        await litellm_handler.close()


def test_sync_post_streaming_status_error_should_not_wait_forever_for_body(
    monkeypatch,
):
    """
    Keep the sync streaming error path aligned with the async path so a
    non-terminating streamed error body cannot block a worker thread forever.
    """

    class HangingSyncErrorStream(httpx.SyncByteStream):
        def __init__(self):
            self.closed_event = threading.Event()

        def __iter__(self):
            self.closed_event.wait()
            if False:
                yield b""

        def close(self):
            self.closed_event.set()

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            request=request,
            headers={"content-type": "application/json"},
            stream=HangingSyncErrorStream(),
        )

    monkeypatch.setattr(
        "litellm.llms.custom_httpx.http_handler._STREAMING_ERROR_BODY_READ_TIMEOUT_SECONDS",
        0.01,
    )

    litellm_handler = HTTPHandler()
    litellm_handler.client.close()
    litellm_handler.client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    try:
        with pytest.raises(MaskedHTTPStatusError) as exc_info:
            litellm_handler.post(
                "https://vertex.example/streamRawPredict",
                stream=True,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.response.status_code == 400
    finally:
        litellm_handler.close()


@pytest.mark.asyncio
async def test_ssl_security_level(monkeypatch):
    # Ensure aiohttp transport is enabled for this test
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)

    with patch.dict(os.environ, clear=True):
        # Set environment variable for SSL security level
        monkeypatch.setenv("SSL_SECURITY_LEVEL", "DEFAULT@SECLEVEL=1")

        # Create async client with SSL verification disabled to isolate SSL context testing
        client = AsyncHTTPHandler()

        try:
            # Get the transport (should be LiteLLMAiohttpTransport)
            transport = client.client._transport
            assert isinstance(transport, LiteLLMAiohttpTransport)

            # Get the aiohttp ClientSession
            client_session = transport._get_valid_client_session()

            # Get the connector from the session
            connector = client_session.connector
            assert isinstance(connector, TCPConnector)

            # Get the SSL context from the connector
            ssl_context = connector._ssl

            # Verify that the SSL context exists and has the correct cipher string
            assert isinstance(ssl_context, ssl.SSLContext)
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_force_ipv4_transport(monkeypatch: pytest.MonkeyPatch):
    """Test transport creation with force_ipv4 enabled"""
    monkeypatch.setattr(litellm, "force_ipv4", True)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    transport = AsyncHTTPHandler._create_async_transport()

    # Should get an AsyncHTTPTransport (no real HTTP call — avoids CI hangs)
    assert isinstance(transport, httpx.AsyncHTTPTransport)


@pytest.mark.asyncio
async def test_aiohttp_disabled_transport(monkeypatch: pytest.MonkeyPatch):
    """Test transport creation with aiohttp disabled"""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)

    transport = AsyncHTTPHandler._create_async_transport()

    # Should get None when both aiohttp is disabled and force_ipv4 is False
    assert transport is None


@pytest.mark.asyncio
async def test_ssl_verification_with_aiohttp_transport(monkeypatch: pytest.MonkeyPatch):
    """
    Test aiohttp respects ssl_verify=False

    We validate that the ssl settings for a litellm transport match what a ssl verify=False aiohttp client would have.

    """
    import aiohttp

    # Ensure aiohttp transport is enabled for this test
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)

    litellm_async_client = AsyncHTTPHandler(ssl_verify=False)

    try:
        transport = litellm_async_client.client._transport
        assert isinstance(transport, LiteLLMAiohttpTransport)
        transport_connector = transport._get_valid_client_session().connector
        assert isinstance(transport_connector, TCPConnector)

        aiohttp_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        try:
            aiohttp_connector = aiohttp_session.connector
            assert isinstance(aiohttp_connector, aiohttp.TCPConnector)

            # assert both litellm transport and aiohttp session have ssl_verify=False
            assert transport_connector._ssl == aiohttp_connector._ssl
        finally:
            await aiohttp_session.close()
    finally:
        await litellm_async_client.close()


@pytest.mark.asyncio
async def test_ssl_verification_with_shared_session(monkeypatch: pytest.MonkeyPatch):
    """
    Test that ssl_verify=False is respected even with shared sessions.

    This was a bug where shared sessions bypassed SSL configuration because
    _create_aiohttp_transport returned immediately without passing ssl_verify
    to the LiteLLMAiohttpTransport constructor.

    The fix stores ssl_verify in the transport and passes it per-request.
    """
    import aiohttp

    # Ensure aiohttp transport is enabled for this test
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)

    shared_session = aiohttp.ClientSession()

    try:
        # Create transport with shared session and ssl_verify=False
        transport = AsyncHTTPHandler._create_aiohttp_transport(
            ssl_verify=False,
            shared_session=shared_session,
        )

        # Verify the transport uses the shared session
        assert transport.client is shared_session

        # Verify the SSL setting is stored in the transport for per-request use
        assert transport._ssl_verify is False
    finally:
        await shared_session.close()


@pytest.mark.asyncio
async def test_ssl_context_with_shared_session(monkeypatch: pytest.MonkeyPatch):
    """
    Test that ssl_context is respected even with shared sessions.
    """
    import aiohttp

    # Ensure aiohttp transport is enabled for this test
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)

    custom_ssl_context = ssl.create_default_context()

    # Create a shared session
    shared_session = aiohttp.ClientSession()

    try:
        # Create transport with shared session and custom ssl_context
        transport = AsyncHTTPHandler._create_aiohttp_transport(
            ssl_context=custom_ssl_context,
            shared_session=shared_session,
        )

        # Verify the transport uses the shared session
        assert transport.client is shared_session

        # Verify the SSL context is stored in the transport for per-request use
        assert transport._ssl_verify is custom_ssl_context
    finally:
        await shared_session.close()


def test_get_ssl_configuration():
    """Test that get_ssl_configuration() returns a proper SSL context with certifi CA bundle
    when no environment variables are set."""
    from litellm.llms.custom_httpx.http_handler import _ssl_context_cache

    # Clear cache to ensure ssl.create_default_context is called
    _ssl_context_cache.clear()

    with patch.dict(os.environ, clear=True):
        with patch("ssl.create_default_context") as mock_create_context:
            # Mock the return value
            mock_ssl_context = MagicMock(spec=ssl.SSLContext)
            mock_ssl_context.set_ciphers = MagicMock()
            mock_ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            mock_create_context.return_value = mock_ssl_context

            # Call the static method
            result = get_ssl_configuration()

            # Verify ssl.create_default_context was called with certifi's CA file
            expected_ca_file = certifi.where()
            mock_create_context.assert_called_once_with(cafile=expected_ca_file)

            # Verify it returns the mocked SSL context
            assert result == mock_ssl_context


def test_get_ssl_configuration_integration():
    """Integration test that _get_ssl_context() returns a working SSL context"""
    # Call the static method without mocking
    ssl_context = get_ssl_configuration()

    # Verify it returns an SSLContext instance
    assert isinstance(ssl_context, ssl.SSLContext)

    # Verify it has basic SSL context properties
    assert ssl_context.protocol is not None
    assert ssl_context.verify_mode is not None


# Session Reuse Tests
class MockClientSession:
    """Mock ClientSession that is not callable"""

    def __init__(self):
        self.closed = False


@pytest.mark.asyncio
async def test_create_aiohttp_transport_with_shared_session():
    """Test that _create_aiohttp_transport reuses shared session when provided"""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    # Create a mock shared session that's not callable
    mock_session = MockClientSession()

    # Test with shared session
    transport = AsyncHTTPHandler._create_aiohttp_transport(
        shared_session=mock_session  # type: ignore
    )

    # Verify the transport uses the shared session directly
    assert transport.client is mock_session
    assert not callable(transport.client)  # Should not be callable


@pytest.mark.asyncio
async def test_async_handler_with_shared_session():
    """Test AsyncHTTPHandler initialization with shared session"""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    # Create a mock shared session
    mock_session = MockClientSession()

    # Create handler with shared session
    handler = AsyncHTTPHandler(shared_session=mock_session)  # type: ignore

    # Verify the handler was created successfully
    assert handler is not None
    assert handler.client is not None


@pytest.mark.asyncio
async def test_get_async_httpx_client_with_shared_session():
    """Test get_async_httpx_client with shared session"""
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,
        AsyncHTTPHandler as AsyncHTTPHandlerReload,
    )
    from litellm.types.utils import LlmProviders

    # Create a mock shared session
    mock_session = MockClientSession()

    # Test with shared session
    client = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,
        shared_session=mock_session,  # type: ignore
    )

    # Verify the client was created successfully
    assert client is not None
    # Import locally to avoid stale reference after module reload in conftest
    assert isinstance(client, AsyncHTTPHandlerReload)


@pytest.mark.asyncio
async def test_get_async_httpx_client_without_shared_session():
    """Test get_async_httpx_client without shared session (backward compatibility)"""
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,
        AsyncHTTPHandler as AsyncHTTPHandlerReload,
    )
    from litellm.types.utils import LlmProviders

    # Test without shared session
    client = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC, shared_session=None)

    # Verify the client was created successfully
    assert client is not None
    # Import locally to avoid stale reference after module reload in conftest
    assert isinstance(client, AsyncHTTPHandlerReload)


@pytest.mark.asyncio
async def test_session_reuse_chain():
    """Test that session is properly passed through the entire call chain"""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    # Create a mock shared session
    mock_session = MockClientSession()

    # Test the entire chain
    transport = AsyncHTTPHandler._create_async_transport(
        shared_session=mock_session  # type: ignore
    )

    # Verify the transport was created
    assert transport is not None

    # Test AsyncHTTPHandler creation
    handler = AsyncHTTPHandler(shared_session=mock_session)  # type: ignore
    assert handler is not None


def test_shared_session_parameter_in_acompletion():
    """Test that acompletion function accepts shared_session parameter"""
    import inspect
    from litellm.main import acompletion

    # Get the function signature
    sig = inspect.signature(acompletion)
    params = list(sig.parameters.keys())

    # Verify shared_session parameter exists
    assert "shared_session" in params

    # Verify the parameter type annotation
    shared_session_param = sig.parameters["shared_session"]
    assert "ClientSession" in str(shared_session_param.annotation)


def test_shared_session_parameter_in_completion():
    """Test that completion function accepts shared_session parameter"""
    import inspect
    from litellm.main import completion

    # Get the function signature
    sig = inspect.signature(completion)
    params = list(sig.parameters.keys())

    # Verify shared_session parameter exists
    assert "shared_session" in params

    # Verify the parameter type annotation
    shared_session_param = sig.parameters["shared_session"]
    assert "ClientSession" in str(shared_session_param.annotation)


@pytest.mark.asyncio
async def test_session_reuse_integration():
    """Integration test for session reuse functionality"""
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,
        AsyncHTTPHandler as AsyncHTTPHandlerReload,
    )
    from litellm.types.utils import LlmProviders

    # Create a mock session
    mock_session = MockClientSession()

    # Create two clients with the same session
    client1 = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,
        shared_session=mock_session,  # type: ignore
    )

    client2 = get_async_httpx_client(
        llm_provider=LlmProviders.OPENAI,
        shared_session=mock_session,  # type: ignore
    )

    # Both clients should be created successfully
    assert client1 is not None
    assert client2 is not None

    # Both should be AsyncHTTPHandler instances
    # Import locally to avoid stale reference after module reload in conftest
    assert isinstance(client1, AsyncHTTPHandlerReload)
    assert isinstance(client2, AsyncHTTPHandlerReload)

    # Clean up
    await client1.close()
    await client2.close()


@pytest.mark.parametrize(
    "env_curve,litellm_curve,expected_curve,should_call",
    [
        # env_curve: SSL_ECDH_CURVE env var | litellm_curve: litellm.ssl_ecdh_curve variable
        # expected_curve: curve that should be set | should_call: whether set_ecdh_curve() should be called
        # Valid configurations
        ("X25519", None, "X25519", True),  # Env var only
        ("prime256v1", None, "prime256v1", True),  # Different valid curve
        (None, "secp384r1", "secp384r1", True),  # litellm variable only
        ("X25519", "secp521r1", "X25519", True),  # Env var takes precedence
        # Empty/None configurations - should skip
        ("", None, None, False),  # Empty string - skip configuration
        (None, None, None, False),  # None value - skip configuration
    ],
)
def test_ssl_ecdh_curve(env_curve, litellm_curve, expected_curve, should_call, monkeypatch):
    """Test SSL ECDH curve configuration with valid curves and precedence"""
    from litellm.llms.custom_httpx.http_handler import _ssl_context_cache

    # Clear cache to ensure fresh SSL context creation
    _ssl_context_cache.clear()

    with patch.dict(os.environ, clear=True):
        if env_curve:
            monkeypatch.setenv("SSL_ECDH_CURVE", env_curve)

        monkeypatch.setattr(litellm, "ssl_ecdh_curve", litellm_curve)

        # Create a real SSL context and patch set_ecdh_curve on it
        # We need a real SSLContext instance (not a MagicMock) because _create_ssl_context
        # calls methods like set_ciphers() and minimum_version that require a real context.
        # We patch set_ecdh_curve specifically to verify it's called with the correct curve.
        real_ssl_context = ssl.create_default_context()
        with patch("ssl.create_default_context", return_value=real_ssl_context):
            with patch.object(real_ssl_context, "set_ecdh_curve") as mock_set_curve:
                ssl_context = get_ssl_configuration()

                if should_call:
                    mock_set_curve.assert_called_once_with(expected_curve)
                else:
                    mock_set_curve.assert_not_called()
                assert isinstance(ssl_context, ssl.SSLContext)


def test_default_user_agent_is_litellm_version(monkeypatch):
    from litellm._version import version
    from litellm.llms.custom_httpx.http_handler import get_default_headers

    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)

    assert get_default_headers()["User-Agent"] == f"litellm/{version}"


def test_user_agent_can_be_overridden_via_env_var(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import get_default_headers

    monkeypatch.setenv("LITELLM_USER_AGENT", "Claude Code")

    assert get_default_headers()["User-Agent"] == "Claude Code"


def test_user_agent_env_var_can_be_empty_string(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import get_default_headers

    monkeypatch.setenv("LITELLM_USER_AGENT", "")

    assert get_default_headers()["User-Agent"] == ""


def test_user_agent_override_is_not_appended_to_default(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)

    handler = HTTPHandler()
    try:
        req = handler.client.build_request(
            "GET",
            "https://example.com",
            headers={"user-agent": "Claude Code"},
        )

        assert req.headers.get_list("User-Agent") == ["Claude Code"]
    finally:
        handler.close()


def test_sync_http_handler_uses_env_user_agent(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    monkeypatch.setenv("LITELLM_USER_AGENT", "Claude Code")

    handler = HTTPHandler()
    try:
        req = handler.client.build_request("GET", "https://example.com")
        assert req.headers.get("User-Agent") == "Claude Code"
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_async_http_handler_uses_env_user_agent(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    monkeypatch.setenv("LITELLM_USER_AGENT", "Claude Code")

    handler = AsyncHTTPHandler()
    try:
        req = handler.client.build_request("GET", "https://example.com")
        assert req.headers.get("User-Agent") == "Claude Code"
    finally:
        await handler.close()


@pytest.mark.asyncio
async def test_httpx_handler_uses_env_user_agent(monkeypatch):
    from litellm.llms.custom_httpx.httpx_handler import HTTPHandler

    monkeypatch.setenv("LITELLM_USER_AGENT", "Claude Code")

    handler = HTTPHandler()
    try:
        req = handler.client.build_request("GET", "https://example.com")
        assert req.headers.get("User-Agent") == "Claude Code"
    finally:
        await handler.close()


def test_get_httpx_client_applies_float_timeout_without_mocking_handler():
    """
    Exercise real _get_httpx_client + HTTPHandler: params={'timeout': x} must reach httpx.Client(timeout=...).
    Uses an uncommon timeout value to avoid colliding with other cached clients in-process.
    """
    timeout = 3847.291
    handler = _get_httpx_client(params={"timeout": timeout})
    try:
        assert isinstance(handler, HTTPHandler)
        assert handler.client.timeout == httpx.Timeout(timeout)
    finally:
        handler.close()


def test_get_httpx_client_applies_httpx_timeout_object_without_mocking_handler():
    t = httpx.Timeout(40.0, connect=5.0)
    handler = _get_httpx_client(params={"timeout": t})
    try:
        assert handler.client.timeout == t
    finally:
        handler.close()


def test_sync_get_forwards_per_request_timeout():
    """HTTPHandler.get(timeout=...) must apply the timeout to that request,
    overriding the client default rather than silently ignoring it."""
    captured = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, request=request, json={"ok": True})

    handler = HTTPHandler()
    handler.client.close()
    handler.client = httpx.Client(
        transport=httpx.MockTransport(mock_handler),
        timeout=httpx.Timeout(5.0),
    )
    try:
        handler.get("https://example.com/poll", timeout=99.0)
        assert captured["timeout"] == {
            "connect": 99.0,
            "read": 99.0,
            "write": 99.0,
            "pool": 99.0,
        }
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_async_get_forwards_per_request_timeout():
    captured = {}

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, request=request, json={"ok": True})

    handler = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(
        transport=httpx.MockTransport(mock_handler),
        timeout=httpx.Timeout(5.0),
    )
    try:
        await handler.get("https://example.com/poll", timeout=99.0)
        assert captured["timeout"] == {
            "connect": 99.0,
            "read": 99.0,
            "write": 99.0,
            "pool": 99.0,
        }
    finally:
        await handler.close()


class TestDefaultCachedClientTimeoutHonorsRequestTimeout:
    """Cached default httpx clients must fall back to an explicit litellm.request_timeout.

    Regression for LIT-2369: get_async_httpx_client / _get_httpx_client hardcoded a
    600s default and never consulted litellm.request_timeout, so provider calls with
    no per-model timeout (e.g. Bedrock) hung for 600s.
    """

    def test_default_when_request_timeout_unset(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.custom_httpx.http_handler import (
            _DEFAULT_TIMEOUT,
            _default_cached_client_timeout,
        )

        monkeypatch.setattr(litellm, "request_timeout", litellm.constants.DEFAULT_REQUEST_TIMEOUT_SECONDS)
        monkeypatch.setattr(litellm, "request_timeout_explicitly_set", False)
        assert _default_cached_client_timeout() is _DEFAULT_TIMEOUT

    def test_uses_explicit_request_timeout(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.llms.custom_httpx.http_handler import (
            _default_cached_client_timeout,
        )

        monkeypatch.setattr(litellm, "request_timeout", 300)
        monkeypatch.setattr(litellm, "request_timeout_explicitly_set", True)
        resolved = _default_cached_client_timeout()
        assert resolved.read == 300.0
        assert resolved.connect == 5.0

    def test_cached_async_client_built_with_explicit_request_timeout(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.caching.llm_caching_handler import LLMClientCache
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        from litellm.types.utils import LlmProviders

        monkeypatch.setattr(litellm, "request_timeout", 300)
        monkeypatch.setattr(litellm, "request_timeout_explicitly_set", True)
        litellm.in_memory_llm_clients_cache = LLMClientCache()
        client = get_async_httpx_client(llm_provider=LlmProviders.BEDROCK)
        assert client.timeout.read == 300.0


async def _read_http_request(reader: asyncio.StreamReader) -> None:
    raw = b""
    while b"\r\n\r\n" not in raw:
        chunk = await reader.read(1024)
        if not chunk:
            return
        raw += chunk
    head, _, body = raw.partition(b"\r\n\r\n")
    content_length = next(
        (int(line.split(b":", 1)[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")),
        0,
    )
    while len(body) < content_length:
        body += await reader.read(content_length - len(body))


@pytest.mark.asyncio
async def test_init_held_async_handler_survives_external_client_close():
    handler = AsyncHTTPHandler(timeout=42.5)
    held_client = handler.client
    await held_client.aclose()
    assert held_client.is_closed

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _read_http_request(reader)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        response = await handler.post(f"http://127.0.0.1:{port}/v1/compress", json={"messages": []})
    finally:
        server.close()
        await server.wait_closed()

    assert response.status_code == 200
    assert handler.client is not held_client
    assert handler.client.timeout == httpx.Timeout(42.5)
    await handler.close()


@pytest.mark.asyncio
async def test_init_held_async_handler_survives_evicted_client_close():
    from litellm.caching.evicted_client_closer import EvictedClientCloser
    from litellm.caching.llm_caching_handler import LLMClientCache

    cache = LLMClientCache(evicted_client_closer=EvictedClientCloser(grace_seconds=0))
    handler = AsyncHTTPHandler(timeout=42.5)
    held_client = handler.client
    cache.set_cache("init-held-handler", handler, litellm_owned_client=True, ttl=0)
    await asyncio.sleep(0.02)
    assert cache.get_cache("init-held-handler") is None
    await asyncio.sleep(0.05)
    assert held_client.is_closed

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _read_http_request(reader)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        response = await handler.post(f"http://127.0.0.1:{port}/v1/compress", json={"messages": []})
    finally:
        server.close()
        await server.wait_closed()

    assert response.status_code == 200
    assert handler.client is not held_client
    assert handler.client.timeout == httpx.Timeout(42.5)
    await handler.close()


def test_init_held_sync_handler_recreates_closed_client():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class OkRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    handler = HTTPHandler(timeout=7)
    held_client = handler.client
    held_client.close()

    server = HTTPServer(("127.0.0.1", 0), OkRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = handler.get(f"http://127.0.0.1:{server.server_port}/")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status_code == 200
    assert handler.client is not held_client
    assert handler.client.timeout == httpx.Timeout(7)
    handler.close()


def test_caller_supplied_sync_client_is_not_replaced_when_closed():
    supplied = httpx.Client()
    handler = HTTPHandler(client=supplied)
    supplied.close()
    assert handler.client is supplied


@pytest.mark.asyncio
async def test_assigned_async_client_is_not_replaced():
    handler = AsyncHTTPHandler()
    await handler.client.aclose()
    replacement = MagicMock()
    handler.client = replacement
    assert handler.client is replacement


def test_concurrent_sync_heal_creates_exactly_one_replacement():
    class GatedHealHandler(HTTPHandler):
        def __init__(self):
            self.heal_started = threading.Event()
            self.release_heal = threading.Event()
            self.heal_calls = 0
            super().__init__(timeout=7)

        def create_client(self) -> httpx.Client:
            if hasattr(self, "_client"):
                self.heal_calls += 1
                self.heal_started.set()
                assert self.release_heal.wait(timeout=5)
            return super().create_client()

    handler = GatedHealHandler()
    handler.client.close()

    seen = []

    def grab_client():
        seen.append(handler.client)

    first = threading.Thread(target=grab_client)
    second = threading.Thread(target=grab_client)
    first.start()
    assert handler.heal_started.wait(timeout=5)
    second.start()
    second.join(timeout=0.3)
    handler.release_heal.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert handler.heal_calls == 1
    assert seen[0] is seen[1]
    assert not seen[0].is_closed
    handler.close()


@pytest.fixture
def fresh_llm_client_cache():
    from litellm.caching.llm_caching_handler import LLMClientCache

    previous = getattr(litellm, "in_memory_llm_clients_cache", None)
    litellm.in_memory_llm_clients_cache = LLMClientCache()
    try:
        yield
    finally:
        litellm.in_memory_llm_clients_cache = previous


def test_sole_referrer_handler_may_close_but_a_sharing_one_may_not():
    from litellm.llms.custom_httpx.http_handler import _handler_may_close_client

    assert _handler_may_close_client(_CLIENT_REFCOUNT_WHEN_HANDLER_IS_SOLE_REFERRER, owns_client=True)
    assert not _handler_may_close_client(_CLIENT_REFCOUNT_WHEN_HANDLER_IS_SOLE_REFERRER + 1, owns_client=True)
    assert not _handler_may_close_client(_CLIENT_REFCOUNT_WHEN_HANDLER_IS_SOLE_REFERRER, owns_client=False)


@pytest.fixture
def keepalive_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    class OkRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedServer(("127.0.0.1", 0), OkRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_exclusively_owned_sync_client_pool_is_closed_when_handler_is_collected(keepalive_server):
    handler = HTTPHandler()
    pool = handler._client._transport._pool
    client_ref = weakref.ref(handler._client)
    handler.get(keepalive_server)

    assert pool._connections, "setup failed: no pooled connection to release"

    del handler
    gc.collect()

    assert client_ref() is None
    assert pool._connections == []


@pytest.mark.asyncio
async def test_exclusively_owned_async_client_pool_is_closed_when_handler_is_collected(keepalive_server, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)

    handler = AsyncHTTPHandler()
    pool = handler.client._transport._pool
    await handler.get(keepalive_server)

    assert pool._connections, "setup failed: no pooled connection to release"

    del handler
    gc.collect()
    await asyncio.sleep(0.25)

    assert pool._connections == []


@pytest.mark.asyncio
async def test_handed_out_async_client_pool_survives_handler_collection(keepalive_server, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)

    handler = AsyncHTTPHandler()
    consumer_client = handler.client
    pool = consumer_client._transport._pool
    await handler.get(keepalive_server)

    assert pool._connections, "setup failed: no pooled connection to observe"

    del handler
    gc.collect()
    await asyncio.sleep(0.25)

    assert pool._connections != []
    assert not consumer_client.is_closed
    await consumer_client.aclose()


def test_handed_out_sync_client_pool_survives_handler_collection(keepalive_server):
    handler = HTTPHandler()
    consumer_client = handler.client
    pool = consumer_client._transport._pool
    handler.get(keepalive_server)

    assert pool._connections, "setup failed: no pooled connection to observe"

    del handler
    gc.collect()

    assert pool._connections != []
    assert not consumer_client.is_closed
    consumer_client.close()


def test_sync_close_leaves_caller_supplied_client_open():
    supplied = httpx.Client()
    handler = HTTPHandler(client=supplied)

    handler.close()

    assert not supplied.is_closed
    supplied.close()


@pytest.mark.asyncio
async def test_async_aexit_closes_an_owned_client_but_not_an_assigned_one():
    owning = AsyncHTTPHandler()
    owned = owning.client

    await owning.__aexit__()

    assert owned.is_closed

    borrowing = AsyncHTTPHandler()
    original = borrowing.client
    assigned = httpx.AsyncClient()
    borrowing.client = assigned

    await borrowing.__aexit__()

    assert not assigned.is_closed
    await assigned.aclose()
    await original.aclose()


@pytest.mark.asyncio
async def test_async_close_leaves_assigned_client_open():
    handler = AsyncHTTPHandler()
    owned = handler.client
    assigned = httpx.AsyncClient()
    handler.client = assigned

    await handler.close()

    assert not assigned.is_closed
    await assigned.aclose()
    await owned.aclose()


def test_client_handed_out_by_sync_cache_survives_eviction_and_collection(fresh_llm_client_cache):
    from litellm.caching.llm_caching_handler import LLMClientCache

    handler = _get_httpx_client()
    consumer_client = handler.client
    handler_ref = weakref.ref(handler)

    assert not consumer_client.is_closed
    assert litellm.in_memory_llm_clients_cache.get_cache("httpx_client") is handler

    litellm.in_memory_llm_clients_cache = LLMClientCache()
    del handler
    gc.collect()

    assert litellm.in_memory_llm_clients_cache.get_cache("httpx_client") is None
    assert handler_ref() is None
    assert not consumer_client.is_closed

    consumer_client.close()


@pytest.mark.asyncio
async def test_client_handed_out_by_async_cache_survives_eviction_and_collection(fresh_llm_client_cache):
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.utils import LlmProviders

    handler = get_async_httpx_client(llm_provider=LlmProviders.OPENAI)
    consumer_client = handler.client
    handler_ref = weakref.ref(handler)

    assert not consumer_client.is_closed

    litellm.in_memory_llm_clients_cache = LLMClientCache()
    del handler
    gc.collect()
    await asyncio.sleep(0.1)

    assert handler_ref() is None
    assert not consumer_client.is_closed

    await consumer_client.aclose()


_SET_COOKIE = "SESSION=upstream-a-secret; Path=/"


def _cookie_recorder():
    """A transport that hands out a Set-Cookie once, and records what comes back."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("cookie"))
        if request.url.path == "/set":
            return httpx.Response(200, headers={"set-cookie": _SET_COOKIE})
        return httpx.Response(200)

    return handler, seen


@pytest.mark.asyncio
async def test_async_client_never_replays_one_upstreams_cookie_to_another():
    """LiteLLM's async clients are pooled and shared by every caller, so a cookie one
    upstream sets would be attached to every later request on a matching domain, reaching
    a different tenant's upstream. The client must persist no response cookie."""
    handler, seen = _cookie_recorder()
    http_handler = AsyncHTTPHandler()
    client = http_handler.client
    client._transport = httpx.MockTransport(handler)

    await client.get("https://upstream-a.example.com/set")
    await client.get("https://upstream-b.example.com/rpc")
    await client.aclose()

    assert dict(client.cookies) == {}, "the shared client stored an upstream's cookie"
    assert seen == [None, None]


def test_sync_client_never_replays_one_upstreams_cookie_to_another():
    """Same invariant on the sync client, which is pooled the same way."""
    handler, seen = _cookie_recorder()
    http_handler = HTTPHandler()
    client = http_handler.client
    client._transport = httpx.MockTransport(handler)

    client.get("https://upstream-a.example.com/set")
    client.get("https://upstream-b.example.com/rpc")
    client.close()

    assert dict(client.cookies) == {}
    assert seen == [None, None]


@pytest.mark.asyncio
async def test_aiohttp_session_never_replays_one_upstreams_cookie_to_another():
    """The httpx jar is not the only one. AiohttpTransport is litellm's default transport
    and the aiohttp ClientSession keeps its own cookie jar, which httpx-level assertions
    cannot see, so blocking only the httpx jar leaves the leak intact on the real path.

    aiohttp's default jar refuses cookies for IP hosts, so this drives a hostname. An
    IP-addressed check passes whether or not the session jar is blocked."""
    from aiohttp import DummyCookieJar
    from yarl import URL

    http_handler = AsyncHTTPHandler(timeout=61.0)
    transport = http_handler.client._transport
    assert isinstance(transport, LiteLLMAiohttpTransport), "aiohttp is no longer the default transport"

    session = transport.client() if callable(transport.client) else transport.client
    jar = session.cookie_jar
    assert isinstance(jar, DummyCookieJar)

    jar.update_cookies({"SESSION": "upstream-a-secret"}, URL("https://upstream-a.example.com"))
    assert len(jar) == 0
    assert dict(jar.filter_cookies(URL("https://upstream-a.example.com"))) == {}
    await session.close()


def _mint_session_on_dead_loop(handler: AsyncHTTPHandler) -> ClientSession:
    """Create the transport's real ClientSession on a loop that then closes.

    This is the lifecycle of every client minted for a short-lived event loop
    (the loop-id-keyed LLM client cache creates one handler per loop): the
    session outlives its loop and can only ever be disposed loop-lessly.
    """
    transport = handler.client._transport
    assert isinstance(transport, LiteLLMAiohttpTransport)
    loop = asyncio.new_event_loop()

    async def _create() -> ClientSession:
        return transport._get_valid_client_session()

    session = loop.run_until_complete(_create())
    loop.close()
    return session


def test_finalizer_without_running_loop_closes_dead_loop_session():
    """A handler finalized with no running event loop must still dispose its
    aiohttp session.

    The async close can never run in that context; without the synchronous
    fallback the session and its connector are abandoned to GC and emit
    "Unclosed client session" / "Unclosed connector" warnings."""
    handler = AsyncHTTPHandler(timeout=61.0)
    session = _mint_session_on_dead_loop(handler)
    assert not session.closed

    del handler
    gc.collect()

    assert session.closed


@pytest.mark.asyncio
async def test_finalizer_with_running_loop_schedules_close_and_holds_task_ref():
    """With a running loop, finalization schedules an async close and must keep
    a strong reference to the task until it completes — a bare create_task()
    result may be collected before it runs, leaving the session unclosed."""
    handler = AsyncHTTPHandler(timeout=61.0)
    transport = handler.client._transport
    assert isinstance(transport, LiteLLMAiohttpTransport)
    session = transport._get_valid_client_session()
    assert not session.closed
    del transport

    baseline_tasks = set(AsyncHTTPHandler._finalizer_close_tasks)
    del handler
    gc.collect()

    scheduled = AsyncHTTPHandler._finalizer_close_tasks - baseline_tasks
    assert len(scheduled) == 1

    await asyncio.gather(*scheduled)
    assert session.closed
    assert not (AsyncHTTPHandler._finalizer_close_tasks & scheduled)


@pytest.mark.asyncio
async def test_sync_close_helper_respects_session_ownership():
    """The loop-less fallback closes only sessions the transport owns; a
    shared session (e.g. the proxy's) must never be closed by a handler."""
    owned_handler = AsyncHTTPHandler(timeout=61.0)
    owned_transport = owned_handler.client._transport
    assert isinstance(owned_transport, LiteLLMAiohttpTransport)
    owned_session = owned_transport._get_valid_client_session()

    baseline = set(LiteLLMAiohttpTransport._background_close_tasks)
    owned_handler._dispose_wrapped_aiohttp_session()
    scheduled = LiteLLMAiohttpTransport._background_close_tasks - baseline
    await asyncio.gather(*scheduled)
    assert owned_session.closed

    shared_session = ClientSession()
    shared_handler = AsyncHTTPHandler(timeout=61.0, shared_session=shared_session)
    shared_transport = shared_handler.client._transport
    assert isinstance(shared_transport, LiteLLMAiohttpTransport)
    assert shared_transport._owns_session is False

    shared_handler._dispose_wrapped_aiohttp_session()
    assert not shared_session.closed

    await shared_session.close()
    await shared_handler.close()
    await owned_handler.close()


@pytest.mark.asyncio
async def test_finalizer_close_done_consumes_exception():
    """A failing finalizer close must have its exception retrieved by the done
    callback, or asyncio emits "Task exception was never retrieved" at GC —
    the same log noise the finalizer path exists to eliminate."""

    async def failing_close() -> None:
        raise RuntimeError("close failed")

    task = asyncio.get_running_loop().create_task(failing_close())
    AsyncHTTPHandler._finalizer_close_tasks.add(task)
    await asyncio.sleep(0)

    AsyncHTTPHandler._on_finalizer_close_done(task)
    assert task not in AsyncHTTPHandler._finalizer_close_tasks

    cancelled = asyncio.get_running_loop().create_task(asyncio.sleep(30))
    cancelled.cancel()
    await asyncio.sleep(0)
    AsyncHTTPHandler._on_finalizer_close_done(cancelled)


@pytest.mark.asyncio
async def test_finalizer_on_live_loop_disposes_foreign_loop_session_without_scheduling():
    """GC on a live loop (e.g. the app's) of a handler whose session belongs to
    another, dead loop must not schedule aclose() here — that is the cross-loop
    path the transport refuses — and must still dispose the session."""
    handler = AsyncHTTPHandler(timeout=61.0)
    session = await asyncio.to_thread(_mint_session_on_dead_loop, handler)
    assert not session.closed

    baseline_tasks = set(AsyncHTTPHandler._finalizer_close_tasks)
    del handler
    gc.collect()

    assert AsyncHTTPHandler._finalizer_close_tasks == baseline_tasks
    assert session.closed
