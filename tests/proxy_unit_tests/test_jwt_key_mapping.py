import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm.proxy.auth.user_api_key_auth import (
    _resolve_jwt_to_virtual_key,
)
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy._types import LiteLLM_JWTAuth, UserAPIKeyAuth
from litellm.caching.caching import DualCache


@pytest.mark.asyncio
async def test_jwt_to_virtual_key_mapping_resolution():
    """
    Test that a JWT claim is correctly resolved to a virtual key token.
    """
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(
        virtual_key_claim_field="email", virtual_key_mapping_cache_ttl=3600
    )

    jwt_claims = {"email": "user@example.com", "sub": "123"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock()

    # Mock finding a mapping
    mock_mapping = MagicMock()
    mock_mapping.token = "sk-1234"
    mock_mapping.is_active = True
    prisma_client.db.litellm_jwtkeymapping.find_first.return_value = mock_mapping

    # Mock getting the key object
    mock_key_obj = UserAPIKeyAuth(token="sk-1234", team_id="team1")

    user_api_key_cache = DualCache()

    # Use patch to mock get_key_object in the module where it's used
    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object", new_callable=AsyncMock
    ) as mock_get_key:
        mock_get_key.return_value = mock_key_obj

        result = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        assert result == mock_key_obj
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_called_once()

        # Test Cache hit
        prisma_client.db.litellm_jwtkeymapping.find_first.reset_mock()
        result_cached = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
        assert result_cached == mock_key_obj
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_jwt_to_virtual_key_mapping_no_mapping():
    """
    Test that when no mapping exists, resolve returns None.
    """
    jwt_handler = JWTHandler()
    jwt_handler.litellm_jwtauth = LiteLLM_JWTAuth(virtual_key_claim_field="email")
    jwt_claims = {"email": "unknown@example.com"}

    prisma_client = MagicMock()
    prisma_client.db.litellm_jwtkeymapping.find_first = AsyncMock()
    prisma_client.db.litellm_jwtkeymapping.find_first.return_value = None

    # Mock get_key_object just in case
    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_key_object", new_callable=AsyncMock
    ):
        user_api_key_cache = DualCache()

        result = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )

        assert result is None

        # Test Negative Cache hit
        prisma_client.db.litellm_jwtkeymapping.find_first.reset_mock()
        result_cached = await _resolve_jwt_to_virtual_key(
            jwt_claims=jwt_claims,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        )
        assert result_cached is None
        prisma_client.db.litellm_jwtkeymapping.find_first.assert_not_called()
