"""
Test that object_permission is automatically loaded when fetching keys and teams.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import get_key_object, get_team_object


@pytest.mark.asyncio
async def test_get_key_object_loads_object_permission():
    """
    Test that get_key_object automatically loads object_permission when object_permission_id exists.
    """
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)  # Not in cache
    
    # Mock the DB response with object_permission_id but no object_permission
    mock_token_data = MagicMock()
    mock_token_data.model_dump.return_value = {
        "token": "test_token_hash",
        "user_id": "test_user",
        "object_permission_id": "test_perm_id",
        "object_permission": None,
    }
    mock_prisma_client.get_data = AsyncMock(return_value=mock_token_data)
    
    # Mock the object_permission that should be loaded
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="test_perm_id",
        mcp_servers=["server1", "server2"],
        vector_stores=["store1"],
    )
    
    # Mock get_object_permission to return the permission
    with patch(
        "litellm.proxy.auth.auth_checks.get_object_permission",
        AsyncMock(return_value=mock_object_permission)
    ), patch(
        "litellm.proxy.auth.auth_checks._cache_key_object",
        AsyncMock()
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
async def test_get_key_object_no_permission_id():
    """
    Test that get_key_object works correctly when no object_permission_id exists.
    """
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)  # Not in cache
    
    # Mock the DB response without object_permission_id
    mock_token_data = MagicMock()
    mock_token_data.model_dump.return_value = {
        "token": "test_token_hash",
        "user_id": "test_user",
        "object_permission_id": None,
        "object_permission": None,
    }
    mock_prisma_client.get_data = AsyncMock(return_value=mock_token_data)
    
    with patch(
        "litellm.proxy.auth.auth_checks._cache_key_object",
        AsyncMock()
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
    mock_team = MagicMock()
    mock_team.dict.return_value = {
        "team_id": "test_team",
        "team_alias": "Test Team",
        "object_permission_id": "test_perm_id",
        "object_permission": None,
    }
    
    # Mock the object_permission that should be loaded
    mock_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="test_perm_id",
        mcp_servers=["team_server1"],
        vector_stores=["team_store1"],
    )
    
    with patch(
        "litellm.proxy.auth.auth_checks._get_team_db_check",
        AsyncMock(return_value=mock_team)
    ), patch(
        "litellm.proxy.auth.auth_checks.get_object_permission",
        AsyncMock(return_value=mock_object_permission)
    ), patch(
        "litellm.proxy.auth.auth_checks._cache_team_object",
        AsyncMock()
    ), patch(
        "litellm.proxy.auth.auth_checks._should_check_db",
        return_value=True
    ), patch(
        "litellm.proxy.auth.auth_checks._update_last_db_access_time"
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
