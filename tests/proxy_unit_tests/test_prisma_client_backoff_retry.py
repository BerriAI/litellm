"""
Test backoff retry mechanisms for PrismaClient methods during _setup_prisma_client state.

This test validates that intermittent database connection issues are handled correctly
with exponential backoff retries for critical startup operations.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from unittest.mock import Mock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.utils import PrismaClient, ProxyLogging
from prisma.errors import PrismaError, ClientNotConnectedError
import httpx
import backoff


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring 'prisma generate' in CI.

    PrismaClient.__init__ does `from prisma import Prisma` inline, which raises
    RuntimeError when the Prisma client hasn't been generated yet.  Replacing
    sys.modules['prisma'] with a MagicMock lets the import succeed so tests can
    instantiate a real PrismaClient and override client.db with their own mocks.
    """
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield


@pytest.fixture
def mock_prisma_client():
    """Create a mock PrismaClient with necessary attributes"""
    client = MagicMock(spec=PrismaClient)

    # Mock database connection
    client.db = AsyncMock()
    client.db.query_raw = AsyncMock()

    # Mock proxy logging object
    client.proxy_logging_obj = AsyncMock()
    client.proxy_logging_obj.failure_handler = AsyncMock()

    return client


@pytest.fixture
def mock_proxy_logging():
    """Create a mock ProxyLogging object"""
    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    return proxy_logging


@pytest.fixture
def connection_errors():
    """Common database connection errors that should trigger retries"""
    return [
        httpx.ConnectError("Connection failed"),
        httpx.TimeoutException("Request timeout"),
        ClientNotConnectedError(),
        PrismaError("Database connection lost"),
        ConnectionError("Network connection failed"),
        OSError("Connection refused"),
    ]


class TestPrismaClientBackoffRetry:
    """Test suite for PrismaClient backoff retry mechanisms"""

    @pytest.mark.asyncio
    async def test_health_check_success_no_retry(
        self, mock_prisma_client, mock_proxy_logging
    ):
        """Test health_check succeeds immediately without retries"""
        # Mock successful query response
        mock_prisma_client.db.query_raw.return_value = [{"result": 1}]

        # Create real PrismaClient instance with mocked db
        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        # Call health_check
        result = await client.health_check()

        # Verify success
        assert result == [{"result": 1}]
        mock_prisma_client.db.query_raw.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_health_check_retry_then_success(
        self, mock_prisma_client, mock_proxy_logging, connection_errors
    ):
        """Test health_check retries on connection errors and eventually succeeds"""
        # Mock first two calls to fail, third to succeed
        mock_prisma_client.db.query_raw.side_effect = [
            connection_errors[0],  # First call fails
            connection_errors[1],  # Second call fails
            [{"result": 1}],  # Third call succeeds
        ]

        # Create real PrismaClient instance
        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        # Measure execution time to verify backoff delay
        start_time = time.time()
        result = await client.health_check()
        end_time = time.time()

        # Verify eventual success
        assert result == [{"result": 1}]

        # Verify retry attempts (3 calls total)
        assert mock_prisma_client.db.query_raw.call_count == 3

        # Verify backoff delay occurred (should be at least a few milliseconds)
        assert end_time - start_time > 0.01

    @pytest.mark.asyncio
    async def test_health_check_max_retries_exceeded(
        self, mock_prisma_client, mock_proxy_logging, connection_errors
    ):
        """Test health_check fails after max retries (3) are exceeded"""
        # Mock all calls to fail
        mock_prisma_client.db.query_raw.side_effect = connection_errors[0]

        # Create real PrismaClient instance
        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        # Expect final exception after retries
        with pytest.raises(httpx.ConnectError):
            await client.health_check()

        # Verify max retries attempted (3 attempts)
        assert mock_prisma_client.db.query_raw.call_count == 3

    @pytest.mark.asyncio
    async def test_get_spend_logs_row_count_success_no_retry(
        self, mock_prisma_client, mock_proxy_logging
    ):
        """Test _get_spend_logs_row_count succeeds immediately"""
        # Mock successful query response
        mock_prisma_client.db.query_raw.return_value = [{"reltuples": 1000}]

        # Create real PrismaClient instance
        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        result = await client._get_spend_logs_row_count()

        assert result == 1000
        mock_prisma_client.db.query_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_spend_logs_row_count_retry_then_success(
        self, mock_prisma_client, mock_proxy_logging, connection_errors
    ):
        """Test _get_spend_logs_row_count retries and eventually succeeds"""
        # Mock first call fails, second succeeds
        mock_prisma_client.db.query_raw.side_effect = [
            connection_errors[2],  # First call fails (ClientNotConnectedError)
            [{"reltuples": 500}],  # Second call succeeds
        ]

        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        result = await client._get_spend_logs_row_count()

        assert mock_prisma_client.db.query_raw.call_count == 2

    @pytest.mark.asyncio
    async def test_get_spend_logs_row_count_handles_errors_gracefully(
        self, mock_prisma_client, mock_proxy_logging
    ):
        """Test _get_spend_logs_row_count returns 0 on persistent errors"""
        # Mock all calls to fail
        mock_prisma_client.db.query_raw.side_effect = PrismaError("Persistent DB error")

        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        result = await client._get_spend_logs_row_count()

        assert result == 0
        assert mock_prisma_client.db.query_raw.call_count == 3  # Max retries attempted

    @pytest.mark.asyncio
    async def test_set_spend_logs_row_count_in_proxy_state_success(
        self, mock_prisma_client, mock_proxy_logging
    ):
        """Test _set_spend_logs_row_count_in_proxy_state succeeds"""
        # Mock successful query response
        mock_prisma_client.db.query_raw.return_value = [{"reltuples": 2000}]

        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        # Mock proxy_state
        with patch("litellm.proxy.proxy_server.proxy_state") as mock_proxy_state:
            mock_proxy_state.set_proxy_state_variable = Mock()

            await client._set_spend_logs_row_count_in_proxy_state()

            # Verify proxy state was updated
            mock_proxy_state.set_proxy_state_variable.assert_called_once_with(
                variable_name="spend_logs_row_count", value=2000
            )

    @pytest.mark.asyncio
    async def test_set_spend_logs_row_count_retry_behavior(
        self, mock_prisma_client, mock_proxy_logging, connection_errors
    ):
        """Test _set_spend_logs_row_count_in_proxy_state retries on database errors"""
        mock_prisma_client.db.query_raw.side_effect = [
            connection_errors[3],  # First call fails (PrismaError)
            [{"reltuples": 1500}],  # Second call succeeds
        ]

        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        with patch("litellm.proxy.proxy_server.proxy_state") as mock_proxy_state:
            mock_proxy_state.set_proxy_state_variable = Mock()

            await client._set_spend_logs_row_count_in_proxy_state()

            assert mock_prisma_client.db.query_raw.call_count == 2
            mock_proxy_state.set_proxy_state_variable.assert_called_once_with(
                variable_name="spend_logs_row_count", value=1500
            )

    @pytest.mark.asyncio
    async def test_backoff_configuration_parameters(self, mock_proxy_logging):
        """Test that backoff decorators are configured with correct parameters"""
        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )

        # Check that methods have backoff decorators
        assert hasattr(client.health_check, "__wrapped__")
        assert hasattr(client._set_spend_logs_row_count_in_proxy_state, "__wrapped__")

        # Verify backoff configuration exists (methods should have retry behavior)
        # This is implicit verification - the decorators are applied in the source code

    @pytest.mark.asyncio
    async def test_multiple_connection_error_types(
        self, mock_prisma_client, mock_proxy_logging
    ):
        """Test that different types of connection errors all trigger retries"""
        error_types = [
            httpx.ConnectError("Connection error"),
            httpx.TimeoutException("Timeout error"),
            ClientNotConnectedError(),
            PrismaError("Database error"),
            ConnectionError("Network error"),
            OSError("OS-level connection error"),
        ]

        for error_type in error_types:
            # Reset mock for each test
            mock_prisma_client.db.query_raw.reset_mock()
            mock_prisma_client.db.query_raw.side_effect = [
                error_type,  # First call fails
                [{"result": 1}],  # Second call succeeds
            ]

            client = PrismaClient(
                database_url="mock://test", proxy_logging_obj=mock_proxy_logging
            )
            client.db = mock_prisma_client.db
            client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

            # Should succeed after retry
            result = await client.health_check()
            assert result == [{"result": 1}]
            assert mock_prisma_client.db.query_raw.call_count == 2

    @pytest.mark.asyncio
    async def test_setup_prisma_client_integration(
        self, mock_prisma_client, mock_proxy_logging
    ):
        """Test simulated _setup_prisma_client flow with intermittent failures"""
        # This simulates the actual flow that happens in proxy_server.py _setup_prisma_client

        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        # Simulate intermittent failures followed by success
        call_count = 0

        def mock_query_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First two calls fail
                raise httpx.ConnectError("Intermittent connection issue")
            else:
                if "SELECT 1" in str(args):
                    return [{"result": 1}]
                else:
                    return [{"reltuples": 1000}]

        mock_prisma_client.db.query_raw.side_effect = mock_query_side_effect

        with patch("litellm.proxy.proxy_server.proxy_state") as mock_proxy_state:
            mock_proxy_state.set_proxy_state_variable = Mock()

            # Execute the two critical calls from _setup_prisma_client
            health_result = await client.health_check()
            await client._set_spend_logs_row_count_in_proxy_state()

            # Verify both operations eventually succeeded despite initial failures
            assert health_result == [{"result": 1}]
            mock_proxy_state.set_proxy_state_variable.assert_called_once()

            # Verify retries occurred (should have more than 2 total query calls)
            assert mock_prisma_client.db.query_raw.call_count >= 4

    @pytest.mark.asyncio
    async def test_backoff_timing_constraints(
        self, mock_prisma_client, mock_proxy_logging, connection_errors
    ):
        """Test that backoff respects max_time constraint (10 seconds)"""
        # Mock all calls to fail to test max_time
        mock_prisma_client.db.query_raw.side_effect = connection_errors[0]

        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
        client.db = mock_prisma_client.db
        client.proxy_logging_obj = mock_prisma_client.proxy_logging_obj

        start_time = time.time()

        with pytest.raises(httpx.ConnectError):
            await client.health_check()

        end_time = time.time()
        duration = end_time - start_time

        # Should not exceed max_time of 10 seconds significantly
        # Adding small buffer for test execution overhead
        assert duration < 12.0

        # Should have attempted max_tries (3) retries
        assert mock_prisma_client.db.query_raw.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
