"""
Tests for RBAC enforcement on vector store management endpoints.

Verifies that check_feature_access_for_user is called and that a 403 is
raised when vector stores are disabled for internal users.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _make_internal_user(user_id: str = "user-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        user_id=user_id,
    )


_DISABLED_GS = {
    "disable_vector_stores_for_internal_users": True,
    "allow_vector_stores_for_team_admins": False,
}

_ENABLED_GS: dict = {}


# ---------------------------------------------------------------------------
# list_vector_stores
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_vector_stores_blocked_when_disabled():
    from litellm.proxy.vector_store_endpoints.management_endpoints import list_vector_stores

    user = _make_internal_user()
    with patch.dict("litellm.proxy.proxy_server.general_settings", _DISABLED_GS, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await list_vector_stores(user_api_key_dict=user)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_list_vector_stores_allowed_when_not_disabled():
    """list_vector_stores should not raise 403 when vector stores are not disabled."""
    from litellm.proxy.vector_store_endpoints.management_endpoints import list_vector_stores

    import litellm
    user = _make_internal_user()
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedvectorstorestable.find_many = AsyncMock(return_value=[])

    raised_403 = False
    with patch.dict("litellm.proxy.proxy_server.general_settings", _ENABLED_GS, clear=True):
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            with patch.object(litellm, "vector_store_registry", None):
                with patch(
                    "litellm.proxy.vector_store_endpoints.management_endpoints.VectorStoreRegistry._get_vector_stores_from_db",
                    new=AsyncMock(return_value=[]),
                ):
                    try:
                        await list_vector_stores(user_api_key_dict=user)
                    except HTTPException as e:
                        if e.status_code == 403:
                            raised_403 = True
    assert not raised_403, "Should not raise 403 when vector stores are not disabled"


# ---------------------------------------------------------------------------
# new_vector_store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_new_vector_store_blocked_when_disabled():
    from litellm.proxy.vector_store_endpoints.management_endpoints import new_vector_store
    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    user = _make_internal_user()
    vs = LiteLLM_ManagedVectorStore(vector_store_id="vs-1", custom_llm_provider="openai")  # type: ignore[call-arg]

    with patch.dict("litellm.proxy.proxy_server.general_settings", _DISABLED_GS, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await new_vector_store(vector_store=vs, user_api_key_dict=user)
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Admin user is never blocked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_vector_stores_admin_not_blocked():
    """Proxy admin should never be blocked, even when vector stores are disabled."""
    from litellm.proxy.vector_store_endpoints.management_endpoints import list_vector_stores

    import litellm
    admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
        user_id="admin-1",
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedvectorstorestable.find_many = AsyncMock(return_value=[])

    raised_403 = False
    with patch.dict("litellm.proxy.proxy_server.general_settings", _DISABLED_GS, clear=True):
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            with patch.object(litellm, "vector_store_registry", None):
                with patch(
                    "litellm.proxy.vector_store_endpoints.management_endpoints.VectorStoreRegistry._get_vector_stores_from_db",
                    new=AsyncMock(return_value=[]),
                ):
                    try:
                        await list_vector_stores(user_api_key_dict=admin)
                    except HTTPException as e:
                        if e.status_code == 403:
                            raised_403 = True
    assert not raised_403, "Admin should not be blocked even when vector stores are disabled"
