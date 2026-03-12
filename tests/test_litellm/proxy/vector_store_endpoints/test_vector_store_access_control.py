"""
Test vector store access control based on team membership, object permissions,
and admin role.

Core tests:
1. Access control logic works correctly for different team scenarios
2. Delete endpoint enforces team access control
3. Admin users always have access
4. Virtual keys with object_permission.vector_stores have access (issue #22577)
5. Both endpoints.py and management_endpoints.py copies behave identically
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.vector_store_endpoints.management_endpoints import (
    _check_vector_store_access,
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


def test_check_vector_store_access_admin_bypass():
    """Test that proxy admin users always have access regardless of team_id.

    Regression test for https://github.com/BerriAI/litellm/issues/22577
    """
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    admin_user = UserAPIKeyAuth(
        team_id="different_team",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    assert _check_vector_store_access(vector_store, admin_user) is True


def test_check_vector_store_access_object_permission_grants_access():
    """Test that a virtual key with object_permission.vector_stores can access
    the listed vector store even when the team_id does not match.

    This is the core fix for https://github.com/BerriAI/litellm/issues/22577
    """
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_123",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(
        team_id="team_789",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_1",
            vector_stores=["vs_123", "vs_other"],
        ),
    )
    assert _check_vector_store_access(vector_store, user) is True


def test_check_vector_store_access_object_permission_empty_list_grants_all():
    """Test that an empty object_permission.vector_stores list means
    unrestricted access (can access all vector stores)."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_any",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(
        team_id="team_789",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_1",
            vector_stores=[],
        ),
    )
    assert _check_vector_store_access(vector_store, user) is True


def test_check_vector_store_access_object_permission_does_not_include_store():
    """Test that access is denied when the key has object_permission but the
    specific vector store is not listed."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_restricted",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(
        team_id="team_789",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_1",
            vector_stores=["vs_other"],
        ),
    )
    assert _check_vector_store_access(vector_store, user) is False


def test_check_vector_store_access_no_object_permission():
    """Test that access is denied when there is no object_permission
    and the team_id does not match."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_123",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(
        team_id="team_789",
        object_permission=None,
    )
    assert _check_vector_store_access(vector_store, user) is False


def test_endpoints_check_vector_store_access_matches_management():
    """Verify that the endpoints.py copy of _check_vector_store_access
    behaves identically to the management_endpoints.py copy."""
    from litellm.proxy.vector_store_endpoints.endpoints import (
        _check_vector_store_access as endpoints_check,
    )

    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_123",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }

    # Virtual key with object_permission should be granted access
    user_with_perm = UserAPIKeyAuth(
        team_id="team_789",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_1",
            vector_stores=["vs_123"],
        ),
    )
    assert endpoints_check(vector_store, user_with_perm) is True

    # Admin should be granted access
    admin_user = UserAPIKeyAuth(
        team_id="other_team",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    assert endpoints_check(vector_store, admin_user) is True

    # Wrong team, no permission → denied
    user_denied = UserAPIKeyAuth(team_id="team_789")
    assert endpoints_check(vector_store, user_denied) is False


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


def test_search_endpoint_check_with_object_permission():
    """Test that _update_request_data_with_litellm_managed_vector_store_registry
    allows access when a key has object_permission granting vector store access,
    even when team_id doesn't match.

    End-to-end test for the search path from issue #22577.
    """
    import litellm
    from litellm.proxy.vector_store_endpoints.endpoints import (
        _update_request_data_with_litellm_managed_vector_store_registry,
    )

    mock_vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_pgvector_1",
        "custom_llm_provider": "openai",
        "team_id": "team_owner",
        "litellm_params": {"some_param": "value"},
    }

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = (
        mock_vector_store
    )

    user_with_perm = UserAPIKeyAuth(
        team_id="team_consumer",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm_1",
            vector_stores=["vs_pgvector_1"],
        ),
    )

    with patch.object(litellm, "vector_store_registry", mock_registry):
        result = _update_request_data_with_litellm_managed_vector_store_registry(
            data={"query": "test"},
            vector_store_id="vs_pgvector_1",
            user_api_key_dict=user_with_perm,
        )

    assert result["custom_llm_provider"] == "openai"
    assert result["some_param"] == "value"


def test_search_endpoint_denied_without_permission():
    """Test that _update_request_data_with_litellm_managed_vector_store_registry
    denies access when the key has no matching object_permission and wrong team."""
    import litellm
    from litellm.proxy.vector_store_endpoints.endpoints import (
        _update_request_data_with_litellm_managed_vector_store_registry,
    )

    mock_vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_pgvector_1",
        "custom_llm_provider": "openai",
        "team_id": "team_owner",
    }

    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = (
        mock_vector_store
    )

    user_no_perm = UserAPIKeyAuth(team_id="team_consumer")

    with patch.object(litellm, "vector_store_registry", mock_registry):
        with pytest.raises(HTTPException) as exc_info:
            _update_request_data_with_litellm_managed_vector_store_registry(
                data={"query": "test"},
                vector_store_id="vs_pgvector_1",
                user_api_key_dict=user_no_perm,
            )

    assert exc_info.value.status_code == 403
