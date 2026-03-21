"""
Tests for search tool access control.

Covers:
- _can_object_call_search_tools() with least-privilege semantics
- search_tool_access_check() for key and team level permissions
- ProxyErrorTypes for search tool access denied
- Ensures vector store access check semantics are unchanged
"""

from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _can_object_call_search_tools,
    _can_object_call_vector_stores,
    search_tool_access_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_object_permission(
    search_tools: Optional[List[str]] = None,
    vector_stores: Optional[List[str]] = None,
) -> LiteLLM_ObjectPermissionTable:
    """Create a minimal object permission for testing."""
    return LiteLLM_ObjectPermissionTable(
        object_permission_id="test-perm-id",
        search_tools=search_tools,
        vector_stores=vector_stores if vector_stores is not None else [],
    )


# ===========================================================================
# _can_object_call_search_tools — least-privilege semantics
# ===========================================================================


class TestCanObjectCallSearchTools:
    """should enforce least-privilege semantics for search tools."""

    def test_should_allow_when_permissions_are_none(self):
        """None object_permissions → allow (no permission record)."""
        result = _can_object_call_search_tools(
            object_type="key",
            search_tool_name="any-tool",
            object_permissions=None,
        )
        assert result is True

    def test_should_allow_when_search_tools_field_is_none(self):
        """search_tools=None → allow (field not configured)."""
        perm = _make_object_permission(search_tools=None)
        result = _can_object_call_search_tools(
            object_type="key",
            search_tool_name="any-tool",
            object_permissions=perm,
        )
        assert result is True

    def test_should_deny_when_search_tools_is_empty_list(self):
        """search_tools=[] → DENY (principle of least privilege)."""
        perm = _make_object_permission(search_tools=[])
        with pytest.raises(ProxyException) as exc_info:
            _can_object_call_search_tools(
                object_type="key",
                search_tool_name="any-tool",
                object_permissions=perm,
            )
        assert exc_info.value.type == ProxyErrorTypes.key_search_tool_access_denied

    def test_should_allow_tool_in_allowed_list(self):
        """Requesting a tool that is in the allowed list → allow."""
        perm = _make_object_permission(search_tools=["tool-a", "tool-b"])
        result = _can_object_call_search_tools(
            object_type="key",
            search_tool_name="tool-a",
            object_permissions=perm,
        )
        assert result is True

    def test_should_deny_tool_not_in_allowed_list(self):
        """Requesting a tool NOT in the allowed list → deny."""
        perm = _make_object_permission(search_tools=["tool-a"])
        with pytest.raises(ProxyException) as exc_info:
            _can_object_call_search_tools(
                object_type="key",
                search_tool_name="tool-b",
                object_permissions=perm,
            )
        assert exc_info.value.type == ProxyErrorTypes.key_search_tool_access_denied

    def test_should_use_team_error_type_for_team_object(self):
        """Team denial uses team-specific error type."""
        perm = _make_object_permission(search_tools=[])
        with pytest.raises(ProxyException) as exc_info:
            _can_object_call_search_tools(
                object_type="team",
                search_tool_name="any-tool",
                object_permissions=perm,
            )
        assert exc_info.value.type == ProxyErrorTypes.team_search_tool_access_denied

    def test_should_use_org_error_type_for_org_object(self):
        """Org denial uses org-specific error type."""
        perm = _make_object_permission(search_tools=["other"])
        with pytest.raises(ProxyException) as exc_info:
            _can_object_call_search_tools(
                object_type="org",
                search_tool_name="not-other",
                object_permissions=perm,
            )
        assert exc_info.value.type == ProxyErrorTypes.org_search_tool_access_denied

    def test_should_allow_all_tools_in_allowed_list(self):
        """Multiple tools in allowed list all pass."""
        perm = _make_object_permission(search_tools=["a", "b", "c"])
        for name in ["a", "b", "c"]:
            result = _can_object_call_search_tools(
                object_type="key",
                search_tool_name=name,
                object_permissions=perm,
            )
            assert result is True
        with pytest.raises(ProxyException):
            _can_object_call_search_tools(
                object_type="key",
                search_tool_name="d",
                object_permissions=perm,
            )


# ===========================================================================
# search_tool_access_check — async DB lookups
# ===========================================================================


_PROXY_SERVER = "litellm.proxy.proxy_server"
_GET_OBJ_PERM = "litellm.proxy.auth.auth_checks.get_object_permission"


@pytest.mark.asyncio
async def test_should_allow_when_no_prisma_client():
    """No prisma client → allow."""
    with patch(f"{_PROXY_SERVER}.prisma_client", None):
        result = await search_tool_access_check(
            search_tool_name="any-tool",
            valid_token=None,
        )
    assert result is True


@pytest.mark.asyncio
async def test_should_allow_when_no_token():
    """No valid_token → allow."""
    with patch(f"{_PROXY_SERVER}.prisma_client", MagicMock()), \
         patch(f"{_PROXY_SERVER}.proxy_logging_obj", MagicMock()), \
         patch(f"{_PROXY_SERVER}.user_api_key_cache", MagicMock()):
        result = await search_tool_access_check(
            search_tool_name="any-tool",
            valid_token=None,
        )
    assert result is True


@pytest.mark.asyncio
async def test_should_allow_when_no_permission_ids():
    """Token with no object_permission_id or team_object_permission_id → allow."""
    token = UserAPIKeyAuth(
        object_permission_id=None,
        team_object_permission_id=None,
    )
    with patch(f"{_PROXY_SERVER}.prisma_client", MagicMock()), \
         patch(f"{_PROXY_SERVER}.proxy_logging_obj", MagicMock()), \
         patch(f"{_PROXY_SERVER}.user_api_key_cache", MagicMock()):
        result = await search_tool_access_check(
            search_tool_name="any-tool",
            valid_token=token,
        )
    assert result is True


@pytest.mark.asyncio
async def test_should_allow_key_with_matching_permission():
    """Key with object_permission that includes the tool → allow."""
    mock_perm = MagicMock()
    mock_perm.search_tools = ["my-tool"]

    token = UserAPIKeyAuth(
        object_permission_id="key-perm-id",
        team_object_permission_id=None,
    )
    with patch(f"{_PROXY_SERVER}.prisma_client", MagicMock()), \
         patch(f"{_PROXY_SERVER}.proxy_logging_obj", MagicMock()), \
         patch(f"{_PROXY_SERVER}.user_api_key_cache", MagicMock()), \
         patch(_GET_OBJ_PERM, AsyncMock(return_value=mock_perm)):
        result = await search_tool_access_check(
            search_tool_name="my-tool",
            valid_token=token,
        )
    assert result is True


@pytest.mark.asyncio
async def test_should_deny_key_with_empty_search_tools():
    """Key with empty search_tools → deny."""
    mock_perm = MagicMock()
    mock_perm.search_tools = []

    token = UserAPIKeyAuth(
        object_permission_id="key-perm-id",
        team_object_permission_id=None,
    )
    with patch(f"{_PROXY_SERVER}.prisma_client", MagicMock()), \
         patch(f"{_PROXY_SERVER}.proxy_logging_obj", MagicMock()), \
         patch(f"{_PROXY_SERVER}.user_api_key_cache", MagicMock()), \
         patch(_GET_OBJ_PERM, AsyncMock(return_value=mock_perm)):
        with pytest.raises(ProxyException) as exc_info:
            await search_tool_access_check(
                search_tool_name="any-tool",
                valid_token=token,
            )
    assert exc_info.value.type == ProxyErrorTypes.key_search_tool_access_denied


@pytest.mark.asyncio
async def test_should_deny_team_with_empty_search_tools():
    """Team with empty search_tools → deny."""
    mock_team_perm = MagicMock()
    mock_team_perm.search_tools = []

    token = UserAPIKeyAuth(
        object_permission_id=None,
        team_object_permission_id="team-perm-id",
    )
    with patch(f"{_PROXY_SERVER}.prisma_client", MagicMock()), \
         patch(f"{_PROXY_SERVER}.proxy_logging_obj", MagicMock()), \
         patch(f"{_PROXY_SERVER}.user_api_key_cache", MagicMock()), \
         patch(_GET_OBJ_PERM, AsyncMock(return_value=mock_team_perm)):
        with pytest.raises(ProxyException) as exc_info:
            await search_tool_access_check(
                search_tool_name="any-tool",
                valid_token=token,
            )
    assert exc_info.value.type == ProxyErrorTypes.team_search_tool_access_denied


@pytest.mark.asyncio
async def test_should_deny_when_key_allows_but_team_denies():
    """Key allows, team denies → deny (team check second)."""
    key_perm = MagicMock()
    key_perm.search_tools = ["the-tool"]

    team_perm = MagicMock()
    team_perm.search_tools = []

    async def mock_get_perm(object_permission_id, **kwargs):
        if object_permission_id == "key-perm-id":
            return key_perm
        elif object_permission_id == "team-perm-id":
            return team_perm
        return None

    token = UserAPIKeyAuth(
        object_permission_id="key-perm-id",
        team_object_permission_id="team-perm-id",
    )
    with patch(f"{_PROXY_SERVER}.prisma_client", MagicMock()), \
         patch(f"{_PROXY_SERVER}.proxy_logging_obj", MagicMock()), \
         patch(f"{_PROXY_SERVER}.user_api_key_cache", MagicMock()), \
         patch(_GET_OBJ_PERM, AsyncMock(side_effect=mock_get_perm)):
        with pytest.raises(ProxyException) as exc_info:
            await search_tool_access_check(
                search_tool_name="the-tool",
                valid_token=token,
            )
    assert exc_info.value.type == ProxyErrorTypes.team_search_tool_access_denied


@pytest.mark.asyncio
async def test_should_allow_when_both_key_and_team_allow():
    """Both key and team allow → allow."""
    key_perm = MagicMock()
    key_perm.search_tools = ["tool-x"]

    team_perm = MagicMock()
    team_perm.search_tools = ["tool-x", "tool-y"]

    async def mock_get_perm(object_permission_id, **kwargs):
        if object_permission_id == "key-perm-id":
            return key_perm
        elif object_permission_id == "team-perm-id":
            return team_perm
        return None

    token = UserAPIKeyAuth(
        object_permission_id="key-perm-id",
        team_object_permission_id="team-perm-id",
    )
    with patch(f"{_PROXY_SERVER}.prisma_client", MagicMock()), \
         patch(f"{_PROXY_SERVER}.proxy_logging_obj", MagicMock()), \
         patch(f"{_PROXY_SERVER}.user_api_key_cache", MagicMock()), \
         patch(_GET_OBJ_PERM, AsyncMock(side_effect=mock_get_perm)):
        result = await search_tool_access_check(
            search_tool_name="tool-x",
            valid_token=token,
        )
    assert result is True


# ===========================================================================
# ProxyErrorTypes classmethod
# ===========================================================================


class TestSearchToolErrorTypes:
    def test_should_return_key_error_type(self):
        assert (
            ProxyErrorTypes.get_search_tool_access_error_type_for_object("key")
            == ProxyErrorTypes.key_search_tool_access_denied
        )

    def test_should_return_team_error_type(self):
        assert (
            ProxyErrorTypes.get_search_tool_access_error_type_for_object("team")
            == ProxyErrorTypes.team_search_tool_access_denied
        )

    def test_should_return_org_error_type(self):
        assert (
            ProxyErrorTypes.get_search_tool_access_error_type_for_object("org")
            == ProxyErrorTypes.org_search_tool_access_denied
        )


# ===========================================================================
# Regression: vector store semantics unchanged (empty = allow all)
# ===========================================================================


class TestVectorStoreAccessNotBroken:
    """Preserves existing vector store semantics: empty list = allow ALL."""

    def test_should_allow_all_when_vector_stores_is_empty(self):
        """Vector stores: empty list = access to ALL (existing behavior)."""
        perm = MagicMock()
        perm.vector_stores = []
        result = _can_object_call_vector_stores(
            object_type="key",
            vector_store_ids_to_run=["store-1"],
            object_permissions=perm,
        )
        assert result is True

    def test_should_allow_when_vector_stores_is_none(self):
        perm = MagicMock()
        perm.vector_stores = None
        result = _can_object_call_vector_stores(
            object_type="key",
            vector_store_ids_to_run=["store-1"],
            object_permissions=perm,
        )
        assert result is True

    def test_should_deny_unlisted_vector_store(self):
        perm = MagicMock()
        perm.vector_stores = ["store-1"]
        with pytest.raises(ProxyException) as exc_info:
            _can_object_call_vector_stores(
                object_type="key",
                vector_store_ids_to_run=["store-99"],
                object_permissions=perm,
            )
        assert exc_info.value.type == ProxyErrorTypes.key_vector_store_access_denied
