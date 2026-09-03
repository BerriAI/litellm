"""
Test vector store access control based on team membership.

Core tests:
1. Access control logic works correctly for different team scenarios
2. Delete endpoint enforces team access control
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


@pytest.mark.asyncio
async def test_check_vector_store_access():
    """Test core access control logic for team-based vector store access"""

    # Test 1: Legacy vector stores (no team_id) are accessible to all
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_legacy",
        "custom_llm_provider": "openai",
        "team_id": None,
    }
    user = UserAPIKeyAuth(team_id="team_456")
    assert await _check_vector_store_access(vector_store, user) is True

    # Test 2: User can access their team's vector stores
    vector_store = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(team_id="team_456")
    assert await _check_vector_store_access(vector_store, user) is True

    # Test 3: User cannot access other teams' vector stores
    vector_store = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(team_id="team_789")
    assert await _check_vector_store_access(vector_store, user) is False


@pytest.mark.asyncio
async def test_check_vector_store_access_proxy_admin_bypass():
    """PROXY_ADMIN can access a vector store even if teams don't match."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_team",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    admin = UserAPIKeyAuth(team_id="team_999", user_role=LitellmUserRoles.PROXY_ADMIN)
    assert await _check_vector_store_access(vector_store, admin) is True


@pytest.mark.asyncio
async def test_check_vector_store_access_key_object_permission_grants_access():
    """A key whose object_permission.vector_stores allowlists the store can access it
    even if its team_id does not match the store's team_id."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_explicit",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(
        team_id="team_789",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-1",
            vector_stores=["vs_explicit"],
        ),
    )
    assert await _check_vector_store_access(vector_store, user) is True


@pytest.mark.asyncio
async def test_check_vector_store_access_key_object_permission_wrong_store_denied():
    """A key whose object_permission.vector_stores lists *other* stores is still denied
    when the key has no other reason to access this store."""
    vector_store: LiteLLM_ManagedVectorStore = {
        "vector_store_id": "vs_target",
        "custom_llm_provider": "openai",
        "team_id": "team_456",
    }
    user = UserAPIKeyAuth(
        team_id="team_789",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-1",
            vector_stores=["vs_other"],
        ),
    )
    assert await _check_vector_store_access(vector_store, user) is False


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
    mock_prisma.db.litellm_managedvectorstorestable.find_unique = AsyncMock(return_value=mock_vector_store)

    # User from different team should get 403
    user_api_key_dict = UserAPIKeyAuth(team_id="team_789")
    request = VectorStoreDeleteRequest(vector_store_id="vs_123")

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma,
    ):
        with patch("litellm.vector_store_registry", None):
            with pytest.raises(HTTPException) as exc_info:
                await delete_vector_store(data=request, user_api_key_dict=user_api_key_dict)

            assert exc_info.value.status_code == 403
            assert "Access denied" in exc_info.value.detail


_UNSCOPED: LiteLLM_ManagedVectorStore = {
    "vector_store_id": "vs_unscoped",
    "custom_llm_provider": "openai",
    "team_id": None,
}
_TEAM_A_OWNED: LiteLLM_ManagedVectorStore = {
    "vector_store_id": "vs_team_a",
    "custom_llm_provider": "openai",
    "team_id": "team_a",
}
_UI_CREATED: LiteLLM_ManagedVectorStore = {
    "vector_store_id": "vs_ui_created",
    "custom_llm_provider": "openai",
    "team_id": "litellm-dashboard",
}


async def _listed_ids(user_api_key_dict: UserAPIKeyAuth) -> list[str]:
    from litellm.proxy.vector_store_endpoints.management_endpoints import (
        list_vector_stores,
    )

    with patch(  # test-quality-ok: the list route reads rows through this module-level DB helper, no injection seam
        "litellm.proxy.vector_store_endpoints.management_endpoints.VectorStoreRegistry._get_vector_stores_from_db",
        new=AsyncMock(return_value=[_UNSCOPED, _TEAM_A_OWNED, _UI_CREATED]),
    ):
        response = await list_vector_stores(user_api_key_dict=user_api_key_dict)
    return sorted(vs["vector_store_id"] for vs in response["data"])


@pytest.mark.asyncio
async def test_list_vector_stores_hides_ungranted_stores_from_non_admin_keys():
    """A store with no team_id and no allowlist entry is not listed for a key it was never granted to;
    only team ownership or an explicit object_permission grant makes a store visible."""
    assert await _listed_ids(UserAPIKeyAuth()) == []
    assert await _listed_ids(UserAPIKeyAuth(team_id="team_a")) == ["vs_team_a"]
    assert await _listed_ids(
        UserAPIKeyAuth(
            team_id="team_b",
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-1", vector_stores=["vs_unscoped"]),
        )
    ) == ["vs_unscoped"]
    assert await _listed_ids(
        UserAPIKeyAuth(
            team_id="team_b",
            team_object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="op-2", vector_stores=["vs_unscoped"]
            ),
        )
    ) == ["vs_unscoped"]
    assert await _listed_ids(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)) == [
        "vs_team_a",
        "vs_ui_created",
        "vs_unscoped",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_team_ids", "session_key_grants", "expected"),
    [
        ([], None, []),
        ([], ["vs_unscoped"], ["vs_unscoped"]),
        (["team_a"], None, ["vs_team_a"]),
        (["team_a", "team_granted"], None, ["vs_team_a", "vs_unscoped"]),
    ],
)
async def test_list_vector_stores_dashboard_session_resolves_real_teams(
    user_team_ids: list[str], session_key_grants: list[str] | None, expected: list[str]
):
    """A dashboard session lists through the user's real teams plus the session key's own grants: stores created
    from the dashboard (team_id litellm-dashboard) are not visible just because every session shares that team id,
    while stores owned by or granted to one of the user's teams, or granted to the session key itself, are."""
    from litellm.models.team import LiteLLM_TeamTableCachedObj

    alice = UserAPIKeyAuth(
        team_id="litellm-dashboard",
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=(
            LiteLLM_ObjectPermissionTable(object_permission_id="op-4", vector_stores=session_key_grants)
            if session_key_grants is not None
            else None
        ),
    )
    teams = {
        "team_a": LiteLLM_TeamTableCachedObj(team_id="team_a"),
        "team_granted": LiteLLM_TeamTableCachedObj(
            team_id="team_granted",
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-3", vector_stores=["vs_unscoped"]),
        ),
    }

    async def fake_get_team_object(team_id: str, **_kwargs: object) -> LiteLLM_TeamTableCachedObj:
        return teams[team_id]

    with (
        patch(  # test-quality-ok: team rows come from the module-level prisma client, no injection seam
            "litellm.proxy.auth.auth_checks.get_team_object", new=fake_get_team_object
        ),
        patch(  # test-quality-ok: the user row comes from the module-level prisma client, no injection seam
            "litellm.proxy.vector_store_endpoints.utils.resolve_ui_session_team_ids",
            new=AsyncMock(return_value=user_team_ids),
        ),
    ):
        assert await _listed_ids(alice) == expected
