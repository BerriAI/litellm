"""
Test that object_permission is automatically loaded when fetching keys and teams.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import get_key_object, get_team_object
from tests.unit.proxy.rust_auth_stores import RustAuthStores, install_rust_auth_stores


@pytest.fixture
def rust_auth_stores(monkeypatch: pytest.MonkeyPatch) -> RustAuthStores:
    return install_rust_auth_stores(monkeypatch)


@pytest.mark.asyncio
async def test_get_key_object_loads_object_permission(rust_auth_stores: RustAuthStores):
    """
    Test that get_key_object automatically loads object_permission when object_permission_id exists.
    """
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)  # Not in cache

    rust_auth_stores.key.rows["test_token_hash"] = {
        "token": "test_token_hash",
        "user_id": "test_user",
        "object_permission_id": "test_perm_id",
        "object_permission": None,
    }

    # Mock the object_permission that should be loaded
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="test_perm_id",
        mcp_servers=["server1", "server2"],
        vector_stores=["store1"],
    )

    # Mock proxy_logging_obj to handle async service hooks
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    mock_proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    # Mock get_object_permission to return the permission
    with (
        patch(
            "litellm.proxy.auth.auth_checks.get_object_permission",
            AsyncMock(return_value=mock_object_permission),
        ),
        patch("litellm.proxy.auth.auth_checks.cache_key_object", AsyncMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj),
    ):
        result = await get_key_object(
            hashed_token="test_token_hash",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )

        # Verify that object_permission was loaded
        assert result.object_permission is not None
        assert result.object_permission.object_permission_id == "test_perm_id"
        assert result.object_permission.mcp_servers == ["server1", "server2"]


@pytest.mark.asyncio
async def test_get_key_object_no_permission_id(rust_auth_stores: RustAuthStores):
    """
    Test that get_key_object works correctly when no object_permission_id exists.
    """
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)  # Not in cache

    rust_auth_stores.key.rows["test_token_hash"] = {
        "token": "test_token_hash",
        "user_id": "test_user",
        "object_permission_id": None,
        "object_permission": None,
    }

    # Mock proxy_logging_obj to handle async service hooks
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    mock_proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    with (
        patch("litellm.proxy.auth.auth_checks.cache_key_object", AsyncMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj),
    ):
        result = await get_key_object(
            hashed_token="test_token_hash",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )

        # Verify that object_permission is None
        assert result.object_permission is None


@pytest.mark.asyncio
async def test_get_team_object_loads_object_permission():
    """
    Test that get_team_object automatically loads object_permission when object_permission_id exists.
    """
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)  # Not in cache

    # Mock team data with object_permission_id
    mock_team = LiteLLM_TeamTableCachedObj(
        team_id="test_team", team_alias="Test Team", object_permission_id="test_perm_id", object_permission=None
    )

    # Mock the object_permission that should be loaded
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="test_perm_id",
        mcp_servers=["team_server1"],
        vector_stores=["team_store1"],
    )

    # Mock proxy_logging_obj to handle async service hooks
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    mock_proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    with (
        patch(
            "litellm.proxy.auth.auth_checks._get_team_db_check",
            AsyncMock(return_value=mock_team),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_object_permission",
            AsyncMock(return_value=mock_object_permission),
        ),
        patch("litellm.proxy.auth.auth_checks.cache_team_object", AsyncMock()),
        patch("litellm.proxy.auth.auth_checks._should_check_db", return_value=True),
        patch("litellm.proxy.auth.auth_checks._update_last_db_access_time"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj),
    ):
        result = await get_team_object(
            team_id="test_team",
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
        )

        # Verify that object_permission was loaded
        assert result.object_permission is not None
        assert result.object_permission.object_permission_id == "test_perm_id"
        assert result.object_permission.mcp_servers == ["team_server1"]
