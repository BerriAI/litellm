import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport


class TestBaseLLMAIOHTTPHandler:
    """Test cases for BaseLLMAIOHTTPHandler dependency injection functionality"""

    def test_init_with_no_client_session(self):
        """Test handler initialization with no client session"""
        handler = BaseLLMAIOHTTPHandler()

        assert handler.client_session is None
        assert handler._owns_session is True

    def test_init_with_provided_client_session(self):
        """Test handler initialization with provided client session"""
        # Create a mock client session
        mock_session = Mock()

        handler = BaseLLMAIOHTTPHandler(client_session=mock_session)

        assert handler.client_session is mock_session
        assert handler._owns_session is False

    def test_get_async_client_session_with_dynamic_session(self):
        """Test _get_async_client_session with dynamic session parameter"""
        handler = BaseLLMAIOHTTPHandler()

        dynamic_session = Mock()

        result = handler._get_async_client_session(
            dynamic_client_session=dynamic_session
        )

        assert result is dynamic_session

    def test_get_async_client_session_with_instance_session(self):
        """Test _get_async_client_session with instance session"""
        instance_session = Mock()
        handler = BaseLLMAIOHTTPHandler(client_session=instance_session)

        result = handler._get_async_client_session()

        assert result is instance_session

    @patch("aiohttp.ClientSession")
    def test_get_async_client_session_create_new(self, mock_client_session):
        """Test _get_async_client_session creates new session when none provided"""
        handler = BaseLLMAIOHTTPHandler()
        mock_session_instance = Mock()
        mock_client_session.return_value = mock_session_instance

        result = handler._get_async_client_session()

        # Verify new session was created
        mock_client_session.assert_called_once()
        assert handler.client_session is mock_session_instance
        assert handler._owns_session is True
        assert result is mock_session_instance

    @pytest.mark.asyncio
    async def test_close_with_owned_session(self):
        """Test close() method with owned session"""
        # Create a mock session that we own
        mock_session = Mock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        # Create handler that owns the session
        handler = BaseLLMAIOHTTPHandler()
        handler.client_session = mock_session
        handler._owns_session = True

        await handler.close()

        # Verify close was called
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_non_owned_session(self):
        """Test close() method with non-owned session (should not close)"""
        # Create a mock session that we don't own
        mock_session = Mock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        handler = BaseLLMAIOHTTPHandler(client_session=mock_session)

        await handler.close()

        # Verify close was NOT called since we don't own this session
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_already_closed_session(self):
        """Test close() method with already closed session"""
        mock_session = Mock()
        mock_session.closed = True
        mock_session.close = AsyncMock()

        handler = BaseLLMAIOHTTPHandler()
        handler.client_session = mock_session
        handler._owns_session = True

        await handler.close()

        # Verify close was NOT called since session is already closed
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_no_session(self):
        """Test close() method with no session"""
        handler = BaseLLMAIOHTTPHandler()

        # Should not raise any exceptions
        await handler.close()

    def test_session_priority_dynamic_over_instance(self):
        """Test that dynamic session takes priority over instance session"""
        instance_session = Mock()
        dynamic_session = Mock()

        handler = BaseLLMAIOHTTPHandler(client_session=instance_session)

        result = handler._get_async_client_session(
            dynamic_client_session=dynamic_session
        )

        assert result is dynamic_session
        assert result is not instance_session

    def test_session_ownership_tracking(self):
        """Test proper session ownership tracking in various scenarios"""
        # Scenario 1: Provided session - not owned
        provided_session = Mock()
        handler1 = BaseLLMAIOHTTPHandler(client_session=provided_session)
        assert not handler1._owns_session

        # Scenario 2: No session initially - becomes owned when created
        handler2 = BaseLLMAIOHTTPHandler()
        assert handler2._owns_session

        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_session_instance = Mock()
            mock_client_session.return_value = mock_session_instance

            handler2._get_async_client_session()
            assert handler2._owns_session

    @pytest.mark.asyncio
    async def test_context_manager_pattern_compatibility(self):
        """Test that the handler works well with context manager pattern"""
        mock_session = Mock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        # Test as context manager style usage
        handler = BaseLLMAIOHTTPHandler()
        handler.client_session = mock_session
        handler._owns_session = True

        try:
            # Simulate some work
            session = handler._get_async_client_session()
            assert session is mock_session
        finally:
            await handler.close()

        # Verify cleanup happened
        mock_session.close.assert_called_once()

    @patch("litellm.llms.custom_httpx.aiohttp_handler.aiohttp.ClientSession")
    def test_lazy_session_creation(self, mock_client_session):
        """Test that session is created lazily only when needed"""
        handler = BaseLLMAIOHTTPHandler()

        # Session should not be created on init
        mock_client_session.assert_not_called()
        assert handler.client_session is None

        # Session should be created when requested
        mock_session_instance = Mock()
        mock_client_session.return_value = mock_session_instance

        session = handler._get_async_client_session()

        mock_client_session.assert_called_once()
        assert session is mock_session_instance
        assert handler.client_session is mock_session_instance

    def test_session_reuse(self):
        """Test that the same session is reused across multiple calls"""
        instance_session = Mock()
        handler = BaseLLMAIOHTTPHandler(client_session=instance_session)

        # Multiple calls should return the same session
        session1 = handler._get_async_client_session()
        session2 = handler._get_async_client_session()
        session3 = handler._get_async_client_session()

        assert session1 is session2 is session3 is instance_session

    # ===============================
    # TRANSPORT INJECTION TESTS
    # ===============================

    def test_init_with_transport(self):
        """Test handler initialization with provided transport"""
        mock_transport = Mock(spec=LiteLLMAiohttpTransport)

        handler = BaseLLMAIOHTTPHandler(transport=mock_transport)

        assert handler.transport is mock_transport
        assert handler._owns_transport is False

    def test_init_with_connector(self):
        """Test handler initialization with provided connector"""
        mock_connector = Mock(spec=aiohttp.BaseConnector)

        handler = BaseLLMAIOHTTPHandler(connector=mock_connector)

        assert handler.connector is mock_connector
        assert handler._owns_connector is False

    def test_init_with_transport_and_session(self):
        """Test handler initialization with both transport and session"""
        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        mock_session = Mock()

        handler = BaseLLMAIOHTTPHandler(
            client_session=mock_session, transport=mock_transport
        )

        assert handler.transport is mock_transport
        assert handler._owns_transport is False
        assert handler.client_session is mock_session
        assert handler._owns_session is False

    def test_get_connector_from_provided_connector(self):
        """Test _get_connector returns provided connector"""
        mock_connector = Mock(spec=aiohttp.BaseConnector)
        handler = BaseLLMAIOHTTPHandler(connector=mock_connector)

        result = handler._get_connector()

        assert result is mock_connector

    def test_get_connector_from_transport(self):
        """Test _get_connector extracts connector from transport"""
        mock_connector = Mock(spec=aiohttp.BaseConnector)

        # Use a simple object instead of Mock to avoid callable issues
        class MockSession:
            def __init__(self):
                self.connector = mock_connector

        mock_session = MockSession()

        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        mock_transport.client = mock_session

        handler = BaseLLMAIOHTTPHandler(transport=mock_transport)

        result = handler._get_connector()

        assert result is mock_connector

    def test_get_connector_from_transport_with_callable_client(self):
        """Test _get_connector with transport that has callable client"""
        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        mock_transport.client = lambda: Mock()  # Callable client

        handler = BaseLLMAIOHTTPHandler(transport=mock_transport)

        result = handler._get_connector()

        assert result is None

    @patch("aiohttp.ClientSession")
    def test_create_client_session_with_transport(self, mock_client_session):
        """Test session creation using transport"""
        mock_session_from_transport = Mock()

        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        mock_transport._get_valid_client_session = Mock(
            return_value=mock_session_from_transport
        )

        handler = BaseLLMAIOHTTPHandler(transport=mock_transport)

        result = handler._create_client_session_with_transport()

        # Should use transport's session creation method
        mock_transport._get_valid_client_session.assert_called_once()
        assert result is mock_session_from_transport

        # Should not call aiohttp.ClientSession directly
        mock_client_session.assert_not_called()

    @patch("aiohttp.ClientSession")
    def test_create_client_session_with_connector(self, mock_client_session):
        """Test session creation using connector"""
        mock_connector = Mock(spec=aiohttp.BaseConnector)
        mock_session_instance = Mock()
        mock_client_session.return_value = mock_session_instance

        handler = BaseLLMAIOHTTPHandler(connector=mock_connector)

        result = handler._create_client_session_with_transport()

        # Should create session with connector
        mock_client_session.assert_called_once_with(connector=mock_connector)
        assert result is mock_session_instance

    @patch("aiohttp.ClientSession")
    def test_create_client_session_default(self, mock_client_session):
        """Test default session creation when no transport/connector provided"""
        mock_session_instance = Mock()
        mock_client_session.return_value = mock_session_instance

        handler = BaseLLMAIOHTTPHandler()

        result = handler._create_client_session_with_transport()

        # Should create default session
        mock_client_session.assert_called_once_with()
        assert result is mock_session_instance

    @patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler._create_aiohttp_transport"
    )
    def test_get_or_create_transport(self, mock_create_transport):
        """Test transport creation when none provided"""
        mock_transport_instance = Mock(spec=LiteLLMAiohttpTransport)
        mock_create_transport.return_value = mock_transport_instance

        handler = BaseLLMAIOHTTPHandler()

        result = handler._get_or_create_transport()

        mock_create_transport.assert_called_once()
        assert result is mock_transport_instance
        assert handler.transport is mock_transport_instance
        assert handler._owns_transport is True

    def test_get_or_create_transport_with_existing(self):
        """Test _get_or_create_transport returns existing transport"""
        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        handler = BaseLLMAIOHTTPHandler(transport=mock_transport)

        result = handler._get_or_create_transport()

        assert result is mock_transport

    @pytest.mark.asyncio
    async def test_close_with_owned_transport(self):
        """Test close() method with owned transport"""
        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        mock_transport.aclose = AsyncMock()

        handler = BaseLLMAIOHTTPHandler()
        handler.transport = mock_transport
        handler._owns_transport = True

        await handler.close()

        # Verify transport close was called
        mock_transport.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_non_owned_transport(self):
        """Test close() method with non-owned transport (should not close)"""
        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        mock_transport.aclose = AsyncMock()

        handler = BaseLLMAIOHTTPHandler(transport=mock_transport)

        await handler.close()

        # Verify transport close was NOT called since we don't own this transport
        mock_transport.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_transport_without_aclose_method(self):
        """Test close() handles transport without aclose method gracefully"""
        mock_transport = Mock()  # No aclose method

        handler = BaseLLMAIOHTTPHandler()
        handler.transport = mock_transport
        handler._owns_transport = True

        # Should not raise any exceptions
        await handler.close()

    def test_transport_priority_hierarchy(self):
        """Test that session creation follows the right priority: transport > connector > default"""
        # Test with transport having _get_valid_client_session
        mock_transport = Mock(spec=LiteLLMAiohttpTransport)
        mock_session_from_transport = Mock()
        mock_transport._get_valid_client_session = Mock(
            return_value=mock_session_from_transport
        )

        mock_connector = Mock(spec=aiohttp.BaseConnector)

        handler = BaseLLMAIOHTTPHandler(
            transport=mock_transport, connector=mock_connector
        )

        result = handler._create_client_session_with_transport()

        # Should use transport, not connector
        mock_transport._get_valid_client_session.assert_called_once()
        assert result is mock_session_from_transport
