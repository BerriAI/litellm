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
    CacheSettingsManager,
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


class TestCacheSettingsManager:
    """Tests for CacheSettingsManager class"""

    def test_cache_params_equal_identical_params(self):
        """
        Test that _cache_params_equal returns True for identical params.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
            "password": "test-password",
        }
        params2 = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
            "password": "test-password",
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is True

    def test_cache_params_equal_different_params(self):
        """
        Test that _cache_params_equal returns False for different params.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
        }
        params2 = {
            "type": "redis",
            "host": "different-host",
            "port": "6379",
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is False

    def test_cache_params_equal_filters_redis_type(self):
        """
        Test that _cache_params_equal filters out redis_type field.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "redis_type": "node",  # Should be ignored
        }
        params2 = {
            "type": "redis",
            "host": "localhost",
            "redis_type": "cluster",  # Different value, but should be ignored
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is True

    def test_cache_params_equal_filters_none_values(self):
        """
        Test that _cache_params_equal filters out None values.
        """
        params1 = {
            "type": "redis",
            "host": "localhost",
            "port": None,
        }
        params2 = {
            "type": "redis",
            "host": "localhost",
        }

        result = CacheSettingsManager._cache_params_equal(params1, params2)
        assert result is True

    def test_update_cache_params(self):
        """
        Test that update_cache_params updates the tracked cache params.
        """
        # Reset the class variable
        CacheSettingsManager._last_cache_params = None

        cache_params = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
        }

        CacheSettingsManager.update_cache_params(cache_params)

        assert CacheSettingsManager._last_cache_params == cache_params
        # Verify it's a copy, not a reference
        assert CacheSettingsManager._last_cache_params is not cache_params

    @pytest.mark.asyncio
    async def test_init_cache_settings_in_db_initializes_when_params_changed(self):
        """
        Test that init_cache_settings_in_db initializes cache when params change.
        """
        # Reset the class variable
        CacheSettingsManager._last_cache_params = None

        # Mock prisma client
        mock_prisma_client = MagicMock()
        mock_cache_config = MagicMock()
        mock_cache_config.cache_settings = '{"type": "redis", "host": "localhost", "port": "6379"}'
        mock_prisma_client.db.litellm_cacheconfig.find_unique = AsyncMock(
            return_value=mock_cache_config
        )

        # Mock proxy_config
        mock_proxy_config = MagicMock()
        mock_proxy_config._decrypt_db_variables = MagicMock(
            return_value={
                "type": "redis",
                "host": "localhost",
                "port": "6379",
            }
        )
        mock_proxy_config._init_cache = MagicMock()
        mock_proxy_config.switch_on_llm_response_caching = MagicMock()

        # Call the method
        await CacheSettingsManager.init_cache_settings_in_db(
            prisma_client=mock_prisma_client, proxy_config=mock_proxy_config
        )

        # Verify cache was initialized
        mock_proxy_config._init_cache.assert_called_once()
        mock_proxy_config.switch_on_llm_response_caching.assert_called_once()

        # Verify params were stored
        assert CacheSettingsManager._last_cache_params is not None
        assert "type" in CacheSettingsManager._last_cache_params
        assert "redis_type" not in CacheSettingsManager._last_cache_params

    @pytest.mark.asyncio
    async def test_init_cache_settings_in_db_skips_when_params_unchanged(self):
        """
        Test that init_cache_settings_in_db skips initialization when params unchanged.
        """
        # Set existing params
        existing_params = {
            "type": "redis",
            "host": "localhost",
            "port": "6379",
        }
        CacheSettingsManager._last_cache_params = existing_params.copy()

        # Mock prisma client
        mock_prisma_client = MagicMock()
        mock_cache_config = MagicMock()
        mock_cache_config.cache_settings = '{"type": "redis", "host": "localhost", "port": "6379"}'
        mock_prisma_client.db.litellm_cacheconfig.find_unique = AsyncMock(
            return_value=mock_cache_config
        )

        # Mock proxy_config
        mock_proxy_config = MagicMock()
        mock_proxy_config._decrypt_db_variables = MagicMock(
            return_value={
                "type": "redis",
                "host": "localhost",
                "port": "6379",
            }
        )
        mock_proxy_config._init_cache = MagicMock()
        mock_proxy_config.switch_on_llm_response_caching = MagicMock()

        # Call the method
        await CacheSettingsManager.init_cache_settings_in_db(
            prisma_client=mock_prisma_client, proxy_config=mock_proxy_config
        )

        # Verify cache was NOT initialized (params unchanged)
        mock_proxy_config._init_cache.assert_not_called()
        mock_proxy_config.switch_on_llm_response_caching.assert_not_called()
