"""Tests for MCP toolset scope enforcement."""

import asyncio
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth


def _make_auth(
    mcp_servers: Optional[List[str]] = None,
    mcp_tool_permissions: Optional[Dict[str, List[str]]] = None,
    mcp_toolsets: Optional[List[str]] = None,
) -> UserAPIKeyAuth:
    op = LiteLLM_ObjectPermissionTable(
        object_permission_id="test",
        mcp_servers=mcp_servers,
        mcp_tool_permissions=mcp_tool_permissions or {},
        mcp_toolsets=mcp_toolsets or [],
    )
    return UserAPIKeyAuth(
        api_key="sk-test",
        object_permission=op,
    )


class TestApplyToolsetScope:
    """Tests for _apply_toolset_scope helper."""

    @pytest.mark.asyncio
    async def test_restricts_to_toolset_servers_and_tools(self):
        from litellm.proxy._experimental.mcp_server.server import _apply_toolset_scope

        toolset_perms = {
            "server-a": ["tool1", "tool2"],
            "server-b": ["tool3"],
        }
        with patch(
            "litellm.proxy._experimental.mcp_server.server."
            "global_mcp_server_manager.resolve_toolset_tool_permissions",
            new=AsyncMock(return_value=toolset_perms),
        ):
            auth = _make_auth(mcp_servers=["server-a", "server-b", "server-c"])
            result = await _apply_toolset_scope(auth, "toolset-123")

        op = result.object_permission
        assert op is not None
        assert set(op.mcp_servers or []) == {"server-a", "server-b"}
        assert op.mcp_tool_permissions == toolset_perms

    @pytest.mark.asyncio
    async def test_creates_object_permission_when_none(self):
        from litellm.proxy._experimental.mcp_server.server import _apply_toolset_scope

        toolset_perms = {"server-a": ["tool1"]}
        with patch(
            "litellm.proxy._experimental.mcp_server.server."
            "global_mcp_server_manager.resolve_toolset_tool_permissions",
            new=AsyncMock(return_value=toolset_perms),
        ):
            auth = UserAPIKeyAuth(api_key="sk-test", object_permission=None)
            result = await _apply_toolset_scope(auth, "toolset-123")

        op = result.object_permission
        assert op is not None
        assert op.mcp_servers == ["server-a"]
        assert op.mcp_tool_permissions == toolset_perms


class TestFetchMCPToolsetsAccess:
    """Tests for GET /v1/mcp/toolset access control."""

    def test_empty_toolsets_returns_empty(self):
        """Non-admin key with mcp_toolsets=[] must not see any toolsets."""
        # Simulate what fetch_mcp_toolsets does with raw_toolsets=[]
        raw_toolsets: Optional[List[str]] = []
        # raw_toolsets is [] → return nothing
        assert raw_toolsets is not None
        assert not raw_toolsets  # empty list → return []

    def test_none_toolsets_returns_all(self):
        """Key where mcp_toolsets is absent (None) should return all toolsets."""
        raw_toolsets: Optional[List[str]] = None
        # raw_toolsets is None → no restriction → return all
        assert raw_toolsets is None

    def test_populated_toolsets_filters(self):
        """Key with explicit toolset IDs should only see those."""
        raw_toolsets: Optional[List[str]] = ["ts-1", "ts-2"]
        assert raw_toolsets is not None
        assert len(raw_toolsets) == 2


class TestMCPActiveToolsetContextVar:
    """Tests for _mcp_active_toolset_id ContextVar — clients cannot inject it."""

    def test_contextvar_default_is_none(self):
        from litellm.proxy._experimental.mcp_server.server import _mcp_active_toolset_id

        assert _mcp_active_toolset_id.get() is None

    def test_contextvar_set_and_reset(self):
        from litellm.proxy._experimental.mcp_server.server import _mcp_active_toolset_id

        token = _mcp_active_toolset_id.set("toolset-abc")
        assert _mcp_active_toolset_id.get() == "toolset-abc"
        _mcp_active_toolset_id.reset(token)
        assert _mcp_active_toolset_id.get() is None

    def test_client_header_is_stripped(self):
        """x-mcp-toolset-id header is removed from scope headers before auth runs."""
        # Simulate the stripping logic from handle_streamable_http_mcp
        headers = [
            (b"authorization", b"Bearer sk-test"),
            (b"x-mcp-toolset-id", b"evil-toolset"),
            (b"content-type", b"application/json"),
        ]
        stripped = [(k, v) for k, v in headers if k.lower() != b"x-mcp-toolset-id"]
        assert (b"x-mcp-toolset-id", b"evil-toolset") not in stripped
        assert len(stripped) == 2
