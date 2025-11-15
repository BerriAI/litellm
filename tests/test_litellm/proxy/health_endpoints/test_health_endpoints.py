import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
from prisma.errors import ClientNotConnectedError, HTTPClientClosedError, PrismaError

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.health_endpoints._health_endpoints import (
    _db_health_readiness_check,
    db_health_cache,
    health_services_endpoint,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError(),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_readiness_check_with_prisma_error(prisma_error):
    """
    Test that when prisma_client.health_check() raises a PrismaError and
    allow_requests_on_db_unavailable is True, the function should not raise an error
    and return the cached health status.
    """
    # Mock the prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.health_check.side_effect = prisma_error

    # Reset the health cache to a known state
    global db_health_cache
    db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(minutes=5),
    }

    # Patch the imports and general_settings
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        # Call the function
        result = await _db_health_readiness_check()

        # Verify that the function called health_check
        mock_prisma_client.health_check.assert_called_once()

        # Verify that the function returned the cache
        assert result is not None
        assert result["status"] == "unknown"  # Should retain the status from the cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError(),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_readiness_check_with_error_and_flag_off(prisma_error):
    """
    Test that when prisma_client.health_check() raises a DB error but
    allow_requests_on_db_unavailable is False, the exception should be raised.
    """
    # Mock the prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.health_check.side_effect = prisma_error

    # Reset the health cache
    global db_health_cache
    db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(minutes=5),
    }

    # Patch the imports and general_settings where the flag is False
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ):
        # The function should raise the exception
        with pytest.raises(Exception) as excinfo:
            await _db_health_readiness_check()

        # Verify that the raised exception is the same
        assert excinfo.value == prisma_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_message",
    [
        ("healthy", ""),
        ("unhealthy", "queue not reachable"),
    ],
)
async def test_health_services_endpoint_sqs(status, error_message):
    """
    Verify the /health/services SQS branch returns expected status and message
    based on SQSLogger.async_health_check().
    """
    with patch("litellm.integrations.sqs.SQSLogger") as MockSQSLogger:
        mock_instance = MagicMock()
        mock_instance.async_health_check = AsyncMock(
            return_value={"status": status, "error_message": error_message}
        )
        MockSQSLogger.return_value = mock_instance

        result = await health_services_endpoint(service="sqs")

        assert result["status"] == status
        assert result["message"] == error_message
        mock_instance.async_health_check.assert_awaited_once()

