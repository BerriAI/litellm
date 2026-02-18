import io
import os
import pathlib
import ssl
import sys
from unittest.mock import MagicMock, patch

import certifi
import httpx
import pytest
from aiohttp import ClientSession, TCPConnector

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_ssl_configuration


@pytest.mark.asyncio
async def test_ssl_security_level(monkeypatch):
    # Ensure aiohttp transport is enabled for this test
    original_disable = litellm.disable_aiohttp_transport
    litellm.disable_aiohttp_transport = False
    
    try:
        with patch.dict(os.environ, clear=True):
            # Set environment variable for SSL security level
            monkeypatch.setenv("SSL_SECURITY_LEVEL", "DEFAULT@SECLEVEL=1")

            # Create async client with SSL verification disabled to isolate SSL context testing
            client = AsyncHTTPHandler()

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
            # Optionally, check the ciphers string if needed
            # assert "DEFAULT@SECLEVEL=1" in ssl_context.get_ciphers()
    finally:
        # Restore original setting
        litellm.disable_aiohttp_transport = original_disable


@pytest.mark.asyncio
async def test_force_ipv4_transport():
    """Test transport creation with force_ipv4 enabled"""
    litellm.force_ipv4 = True
    litellm.disable_aiohttp_transport = True

    transport = AsyncHTTPHandler._create_async_transport()

    # Should get an AsyncHTTPTransport
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    # Verify IPv4 configuration through a request
    client = httpx.AsyncClient(transport=transport)
    try:
        response = await client.get("http://example.com")
        assert response.status_code == 200
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ssl_context_transport():
    """Test transport creation with SSL context"""
    # Create a test SSL context
    ssl_context = ssl.create_default_context()

    transport = AsyncHTTPHandler._create_async_transport(ssl_context=ssl_context)
    assert transport is not None

    if isinstance(transport, LiteLLMAiohttpTransport):
        # Get the client session and verify SSL context is passed through
        client_session = transport._get_valid_client_session()
        assert isinstance(client_session, ClientSession)
        assert isinstance(client_session.connector, TCPConnector)
        # Verify the connector has SSL context set by checking if it's using SSL
        assert client_session.connector._ssl is not None


@pytest.mark.asyncio
async def test_aiohttp_disabled_transport():
    """Test transport creation with aiohttp disabled"""
    litellm.disable_aiohttp_transport = True
    litellm.force_ipv4 = False

    transport = AsyncHTTPHandler._create_async_transport()

    # Should get None when both aiohttp is disabled and force_ipv4 is False
    assert transport is None


@pytest.mark.asyncio
async def test_ssl_verification_with_aiohttp_transport():
    """
    Test aiohttp respects ssl_verify=False

    We validate that the ssl settings for a litellm transport match what a ssl verify=False aiohttp client would have.

    """
    import aiohttp

    # Ensure aiohttp transport is enabled for this test
    original_disable = litellm.disable_aiohttp_transport
    litellm.disable_aiohttp_transport = False
    
    try:
        # Create a test SSL context
        litellm_async_client = AsyncHTTPHandler(ssl_verify=False)

        transport = litellm_async_client.client._transport
        assert isinstance(transport, LiteLLMAiohttpTransport)
        transport_connector = transport._get_valid_client_session().connector
        assert isinstance(transport_connector, TCPConnector)

        aiohttp_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False)
        )
        aiohttp_connector = aiohttp_session.connector
        assert isinstance(aiohttp_connector, aiohttp.TCPConnector)

        # assert both litellm transport and aiohttp session have ssl_verify=False
        assert transport_connector._ssl == aiohttp_connector._ssl
    finally:
        # Restore original setting
        litellm.disable_aiohttp_transport = original_disable


@pytest.mark.asyncio
async def test_ssl_verification_with_shared_session():
    """
    Test that ssl_verify=False is respected even with shared sessions.

    This was a bug where shared sessions bypassed SSL configuration because
    _create_aiohttp_transport returned immediately without passing ssl_verify
    to the LiteLLMAiohttpTransport constructor.

    The fix stores ssl_verify in the transport and passes it per-request.
    """
    import aiohttp

    # Ensure aiohttp transport is enabled for this test
    original_disable = litellm.disable_aiohttp_transport
    litellm.disable_aiohttp_transport = False

    try:
        # Create a shared session (simulating what happens in production)
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
    finally:
        # Restore original setting
        litellm.disable_aiohttp_transport = original_disable


@pytest.mark.asyncio
async def test_ssl_context_with_shared_session():
    """
    Test that ssl_context is respected even with shared sessions.
    """
    import aiohttp

    # Ensure aiohttp transport is enabled for this test
    original_disable = litellm.disable_aiohttp_transport
    litellm.disable_aiohttp_transport = False

    try:
        # Create a custom SSL context
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
    finally:
        # Restore original setting
        litellm.disable_aiohttp_transport = original_disable


@pytest.mark.asyncio
async def test_aiohttp_transport_trust_env_setting(monkeypatch):
    """Test that trust_env setting is properly configured in aiohttp transport"""
    # Test 1: Default trust_env behavior
    transport = AsyncHTTPHandler._create_aiohttp_transport()
    client_session = transport._get_valid_client_session()
    
    # Default should be False (litellm.aiohttp_trust_env default)
    default_trust_env = getattr(litellm, 'aiohttp_trust_env', False)
    assert client_session._trust_env == default_trust_env
    
    # Test 2: Environment variable override
    monkeypatch.setenv("AIOHTTP_TRUST_ENV", "True")
    transport_with_env = AsyncHTTPHandler._create_aiohttp_transport()
    client_session_with_env = transport_with_env._get_valid_client_session()
    
    # Should be True when environment variable is set
    assert client_session_with_env._trust_env is True
    
    # Test 3: Verify environment variable with False value
    monkeypatch.setenv("AIOHTTP_TRUST_ENV", "False")
    transport_with_false_env = AsyncHTTPHandler._create_aiohttp_transport()
    client_session_with_false_env = transport_with_false_env._get_valid_client_session()
    
    # Should respect the litellm.aiohttp_trust_env setting when env var is False
    assert client_session_with_false_env._trust_env == default_trust_env


def test_get_ssl_configuration():
    """Test that get_ssl_configuration() returns a proper SSL context with certifi CA bundle
    when no environment variables are set."""
    from litellm.llms.custom_httpx.http_handler import _ssl_context_cache
    
    # Clear cache to ensure ssl.create_default_context is called
    _ssl_context_cache.clear()
    
    with patch.dict(os.environ, clear=True):
        with patch('ssl.create_default_context') as mock_create_context:
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
async def test_create_aiohttp_transport_without_shared_session():
    """Test that _create_aiohttp_transport creates new session when none provided"""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    
    # Test without shared session
    transport = AsyncHTTPHandler._create_aiohttp_transport(shared_session=None)
    
    # Verify the transport uses a lambda function (for backward compatibility)
    assert callable(transport.client)  # Should be a lambda function


@pytest.mark.asyncio
async def test_create_aiohttp_transport_with_closed_session():
    """Test that _create_aiohttp_transport creates new session when shared session is closed"""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    
    # Create a mock closed session
    mock_session = MockClientSession()
    mock_session.closed = True
    
    # Test with closed session
    transport = AsyncHTTPHandler._create_aiohttp_transport(
        shared_session=mock_session  # type: ignore
    )
    
    # Verify the transport creates a new session (lambda function)
    assert callable(transport.client)  # Should be a lambda function


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
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, AsyncHTTPHandler as AsyncHTTPHandlerReload
    from litellm.types.utils import LlmProviders

    # Create a mock shared session
    mock_session = MockClientSession()

    # Test with shared session
    client = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,
        shared_session=mock_session  # type: ignore
    )

    # Verify the client was created successfully
    assert client is not None
    # Import locally to avoid stale reference after module reload in conftest
    assert isinstance(client, AsyncHTTPHandlerReload)


@pytest.mark.asyncio
async def test_get_async_httpx_client_without_shared_session():
    """Test get_async_httpx_client without shared session (backward compatibility)"""
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, AsyncHTTPHandler as AsyncHTTPHandlerReload
    from litellm.types.utils import LlmProviders

    # Test without shared session
    client = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,
        shared_session=None
    )

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
    assert 'shared_session' in params
    
    # Verify the parameter type annotation
    shared_session_param = sig.parameters['shared_session']
    assert 'ClientSession' in str(shared_session_param.annotation)


def test_shared_session_parameter_in_completion():
    """Test that completion function accepts shared_session parameter"""
    import inspect
    from litellm.main import completion
    
    # Get the function signature
    sig = inspect.signature(completion)
    params = list(sig.parameters.keys())
    
    # Verify shared_session parameter exists
    assert 'shared_session' in params
    
    # Verify the parameter type annotation
    shared_session_param = sig.parameters['shared_session']
    assert 'ClientSession' in str(shared_session_param.annotation)


@pytest.mark.asyncio
async def test_session_reuse_integration():
    """Integration test for session reuse functionality"""
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, AsyncHTTPHandler as AsyncHTTPHandlerReload
    from litellm.types.utils import LlmProviders

    # Create a mock session
    mock_session = MockClientSession()

    # Create two clients with the same session
    client1 = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,
        shared_session=mock_session  # type: ignore
    )

    client2 = get_async_httpx_client(
        llm_provider=LlmProviders.OPENAI,
        shared_session=mock_session  # type: ignore
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


@pytest.mark.asyncio
async def test_shared_session_bypasses_cache():
    """
    Test that when shared_session is provided, the cache is bypassed.
    
    This is critical for aiohttp tracing support - users need their custom
    ClientSession (with trace_configs) to be used, not a cached session.
    
    Related: GitHub issue #20174
    """
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.utils import LlmProviders
    
    # First, get a cached client without shared_session
    cached_client = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,
        shared_session=None
    )
    
    # Now create a mock shared session
    mock_session = MockClientSession()
    
    # Get a client WITH shared_session - this should NOT return the cached client
    client_with_session = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,  # Same provider!
        shared_session=mock_session  # type: ignore
    )
    
    # The clients should be DIFFERENT - cache should be bypassed when shared_session is provided
    assert client_with_session is not cached_client, \
        "Cache should be bypassed when shared_session is provided"
    
    # Verify the shared_session handler is using our mock session
    # The transport should have our mock_session as its client
    transport = client_with_session.client._transport
    if hasattr(transport, 'client'):
        assert transport.client is mock_session, \
            "Handler should use the provided shared_session"
    
    # Clean up
    await cached_client.close()
    await client_with_session.close()


@pytest.mark.asyncio  
async def test_shared_session_each_call_gets_new_handler():
    """
    Test that each call with shared_session creates a new handler.
    
    This ensures user sessions (with their trace_configs, etc.) are always
    used and not affected by caching.
    """
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.utils import LlmProviders
    
    # Create two different mock sessions
    mock_session1 = MockClientSession()
    mock_session2 = MockClientSession()
    
    # Get clients with different sessions for the same provider
    client1 = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,
        shared_session=mock_session1  # type: ignore
    )
    
    client2 = get_async_httpx_client(
        llm_provider=LlmProviders.ANTHROPIC,  # Same provider
        shared_session=mock_session2  # type: ignore  # Different session
    )
    
    # Should be different clients, each using their own session
    assert client1 is not client2, \
        "Different shared_sessions should create different handlers"
    
    # Clean up
    await client1.close()
    await client2.close()


@pytest.mark.asyncio
async def test_session_validation():
    """Test that session validation works correctly"""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    
    # Test with None session
    transport1 = AsyncHTTPHandler._create_aiohttp_transport(shared_session=None)
    assert callable(transport1.client)  # Should create lambda
    
    # Test with closed session
    mock_closed_session = MockClientSession()
    mock_closed_session.closed = True
    transport2 = AsyncHTTPHandler._create_aiohttp_transport(shared_session=mock_closed_session)  # type: ignore
    assert callable(transport2.client)  # Should create lambda
    
    # Test with valid session
    mock_valid_session = MockClientSession()
    transport3 = AsyncHTTPHandler._create_aiohttp_transport(shared_session=mock_valid_session)  # type: ignore
    assert transport3.client is mock_valid_session  # Should reuse session


@pytest.mark.parametrize(
    "env_curve,litellm_curve,expected_curve,should_call",
    [
        # env_curve: SSL_ECDH_CURVE env var | litellm_curve: litellm.ssl_ecdh_curve variable
        # expected_curve: curve that should be set | should_call: whether set_ecdh_curve() should be called
        
        # Valid configurations
        ("X25519", None, "X25519", True),           # Env var only
        ("prime256v1", None, "prime256v1", True),   # Different valid curve
        (None, "secp384r1", "secp384r1", True),     # litellm variable only
        ("X25519", "secp521r1", "X25519", True),    # Env var takes precedence
        # Empty/None configurations - should skip
        ("", None, None, False),                     # Empty string - skip configuration
        (None, None, None, False),                   # None value - skip configuration
    ]
)
def test_ssl_ecdh_curve(env_curve, litellm_curve, expected_curve, should_call, monkeypatch):
    """Test SSL ECDH curve configuration with valid curves and precedence"""
    from litellm.llms.custom_httpx.http_handler import _ssl_context_cache
    
    # Clear cache to ensure fresh SSL context creation
    _ssl_context_cache.clear()
    
    with patch.dict(os.environ, clear=True):
        if env_curve:
            monkeypatch.setenv("SSL_ECDH_CURVE", env_curve)
        
        original_value = litellm.ssl_ecdh_curve
        try:
            litellm.ssl_ecdh_curve = litellm_curve
            
            # Create a real SSL context and patch set_ecdh_curve on it
            # We need a real SSLContext instance (not a MagicMock) because _create_ssl_context
            # calls methods like set_ciphers() and minimum_version that require a real context.
            # We patch set_ecdh_curve specifically to verify it's called with the correct curve.
            real_ssl_context = ssl.create_default_context()
            with patch('ssl.create_default_context', return_value=real_ssl_context):
                with patch.object(real_ssl_context, 'set_ecdh_curve') as mock_set_curve:
                    ssl_context = get_ssl_configuration()
                    
                    if should_call:
                        mock_set_curve.assert_called_once_with(expected_curve)
                    else:
                        mock_set_curve.assert_not_called()
                    assert isinstance(ssl_context, ssl.SSLContext)
        finally:
            litellm.ssl_ecdh_curve = original_value


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
