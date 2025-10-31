"""
Unit tests for cache settings management endpoints
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.cache_settings_endpoints import (
    CacheTestRequest,
    test_cache_connection,
)


@pytest.mark.asyncio
async def test_test_cache_connection_calls_cache_test_connection_with_params():
    """
    Test that test_cache_connection endpoint calls cache.test_connection()
    with the correct parameters from the request body.
    """
    # Mock cache settings from request
    cache_settings = {
        "type": "redis",
        "host": "test-redis-host",
        "port": "6379",
        "password": "test-password",
    }

    request = CacheTestRequest(cache_settings=cache_settings)
    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test", user_id="test-user"
    )

    # Mock Cache class and its test_connection method
    mock_cache_instance = MagicMock()
    mock_cache_instance.cache = MagicMock()
    mock_cache_instance.cache.test_connection = AsyncMock(
        return_value={
            "status": "success",
            "message": "Redis connection test successful",
        }
    )

    # Patch Cache class at the import location (litellm module)
    with patch("litellm.Cache") as mock_cache_class:
        mock_cache_class.return_value = mock_cache_instance

        # Call the endpoint
        result = await test_cache_connection(
            request=request, user_api_key_dict=user_api_key_dict
        )

        # Verify Cache was instantiated with correct params
        mock_cache_class.assert_called_once_with(**cache_settings)

        # Verify test_connection was called on the cache instance
        mock_cache_instance.cache.test_connection.assert_called_once()

        # Verify response
        assert result.status == "success"
        assert result.message == "Redis connection test successful"
        assert result.error is None

