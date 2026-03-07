"""
Test vector store access control based on team membership and object_permission.

Core tests:
1. Access control logic works correctly for different team scenarios
2. Delete endpoint enforces team access control
3. Virtual keys with object_permission.vector_stores can access team-scoped stores
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.management_endpoints import (
    _check_vector_store_access,
)
from litellm.proxy.vector_store_endpoints.endpoints import (
    _check_vector_store_access as _check_vector_store_access_endpoints,
)
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore


def test_check_vector_store_access():
    """Test core access control logic for team-based vector store access"""
    
    # Test 1: Legacy vector stores (no team_id) are accessible to all
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_legacy",
        "custom_llm_provider": "openai",
        "team_id": None,
    }
    user = UserAPIKeyAuth(team_id="team_456")
    assert _check_vector_store_access(vector_store, user) is True
    
    # Test 2: User can access their team's vector stores
    vector_store = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(team_id="team_456")
    assert _check_vector_store_access(vector_store, user) is True
    
    # Test 3: User cannot access other teams' vector stores
    vector_store = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(team_id="team_789")
    assert _check_vector_store_access(vector_store, user) is False


@pytest.mark.asyncio
async def test_delete_vector_store_checks_access():
    """Test that delete endpoint enforces team access control"""
    from litellm.proxy.vector_store_endpoints.management_endpoints import (
        delete_vector_store,
    )
    from litellm.types.vector_stores import VectorStoreDeleteRequest

    mock_prisma = MagicMock()
    mock_vector_store = MagicMock(
        model_dump=lambda: {
            "vector_store_id": "vs_123",
            "custom_llm_provider": "openai",
            "team_id": "team_456",
        }
    )
    mock_prisma.db.litellm_managedvectorstorestable.find_unique = AsyncMock(
        return_value=mock_vector_store
    )

    # User from different team should get 403
    user_api_key_dict = UserAPIKeyAuth(team_id="team_789")
    request = VectorStoreDeleteRequest(vector_store_id="vs_123")

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ):
        with patch("litellm.vector_store_registry", None):
            with pytest.raises(HTTPException) as exc_info:
                await delete_vector_store(
                    data=request, user_api_key_dict=user_api_key_dict
                )

            assert exc_info.value.status_code == 403
            assert "Access denied" in exc_info.value.detail


# --- object_permission tests ---
# Applied to both management_endpoints and endpoints versions since the logic
# is identical in both.


def _make_object_permission_user(team_id: str, permitted_store_ids: list) -> UserAPIKeyAuth:
    perm = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm_test",
        vector_stores=permitted_store_ids,
    )
    return UserAPIKeyAuth(team_id=team_id, object_permission=perm)


def test_object_permission_grants_access_to_team_scoped_store():
    """Virtual key with explicit vector_store_id in object_permission can access
    a store owned by a different team."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_123",
        "custom_llm_provider": "openai",
        "team_id": "team_other",
    }
    user = _make_object_permission_user(team_id="team_mine", permitted_store_ids=["vs_123"])

    # management_endpoints version
    assert _check_vector_store_access(vector_store, user) is True
    # endpoints version
    assert _check_vector_store_access_endpoints(vector_store, user) is True


def test_object_permission_denies_access_when_store_id_not_in_list():
    """Virtual key's object_permission does not include the requested store — deny."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_456",
        "custom_llm_provider": "openai",
        "team_id": "team_other",
    }
    user = _make_object_permission_user(team_id="team_mine", permitted_store_ids=["vs_123"])

    # management_endpoints version
    assert _check_vector_store_access(vector_store, user) is False
    # endpoints version
    assert _check_vector_store_access_endpoints(vector_store, user) is False


def test_object_permission_none_falls_back_to_team_check():
    """When object_permission is None, team_id matching is still the gate."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_789",
        "custom_llm_provider": "openai",
        "team_id": "team_abc",
    }
    user_same_team = UserAPIKeyAuth(team_id="team_abc", object_permission=None)
    user_diff_team = UserAPIKeyAuth(team_id="team_xyz", object_permission=None)

    # management_endpoints version
    assert _check_vector_store_access(vector_store, user_same_team) is True
    assert _check_vector_store_access(vector_store, user_diff_team) is False
    # endpoints version
    assert _check_vector_store_access_endpoints(vector_store, user_same_team) is True
    assert _check_vector_store_access_endpoints(vector_store, user_diff_team) is False
