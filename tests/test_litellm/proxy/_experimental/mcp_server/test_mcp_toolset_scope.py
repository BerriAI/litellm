"""Tests for MCP toolset scope enforcement."""

import asyncio
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)


def _make_auth(
    mcp_servers: Optional[List[str]] = None,
    mcp_tool_permissions: Optional[Dict[str, List[str]]] = None,
    mcp_toolsets: Optional[List[str]] = None,
) -> UserAPIKeyAuth:
    op = LiteLLM_ObjectPermissionTable(
        object_permission_id="test",
        mcp_servers=mcp_servers,
        mcp_tool_permissions=mcp_tool_permissions or {},
        mcp_toolsets=mcp_toolsets,
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
            # Key has been explicitly granted toolset-123 — access check passes.
            auth = _make_auth(
                mcp_servers=["server-a", "server-b", "server-c"],
                mcp_toolsets=["toolset-123"],
            )
            result = await _apply_toolset_scope(auth, "toolset-123")

        op = result.object_permission
        assert op is not None
        assert set(op.mcp_servers or []) == {"server-a", "server-b"}
        assert op.mcp_tool_permissions == toolset_perms

    @pytest.mark.asyncio
    async def test_admin_creates_object_permission_when_none(self):
        """Admin key with object_permission=None can access any toolset."""
        from litellm.proxy._experimental.mcp_server.server import _apply_toolset_scope

        toolset_perms = {"server-a": ["tool1"]}
        with patch(
            "litellm.proxy._experimental.mcp_server.server."
            "global_mcp_server_manager.resolve_toolset_tool_permissions",
            new=AsyncMock(return_value=toolset_perms),
        ):
            auth = UserAPIKeyAuth(
                api_key="sk-test",
                user_role=LitellmUserRoles.PROXY_ADMIN,
                object_permission=None,
            )
            result = await _apply_toolset_scope(auth, "toolset-123")

        op = result.object_permission
        assert op is not None
        assert op.mcp_servers == ["server-a"]
        assert op.mcp_tool_permissions == toolset_perms

    @pytest.mark.asyncio
    async def test_non_admin_no_object_permission_raises_403(self):
        """Non-admin key with object_permission=None is denied (no grants configured)."""
        from starlette.exceptions import HTTPException

        from litellm.proxy._experimental.mcp_server.server import _apply_toolset_scope

        auth = UserAPIKeyAuth(api_key="sk-test", object_permission=None)
        with pytest.raises(HTTPException) as exc_info:
            await _apply_toolset_scope(auth, "toolset-123")
        assert exc_info.value.status_code == 403


class TestFetchMCPToolsetsAccess:
    """Tests for GET /v1/mcp/toolset access control."""

    @pytest.mark.asyncio
    async def test_non_admin_empty_grants_returns_empty(self):
        """Non-admin key with mcp_toolsets=[] must not see any toolsets."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            fetch_mcp_toolsets,
        )

        auth = _make_auth(mcp_toolsets=[])
        mock_client = MagicMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.list_mcp_toolsets",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
        ):
            result = await fetch_mcp_toolsets(user_api_key_dict=auth)

        assert result == []
        mock_list.assert_not_called()

    @pytest.mark.asyncio
    async def test_admin_unrestricted_returns_all(self):
        """Admin key with mcp_toolsets absent (None) gets all toolsets."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            fetch_mcp_toolsets,
        )

        auth = UserAPIKeyAuth(
            api_key="sk-test",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            object_permission=None,
        )
        fake_toolsets = [MagicMock(), MagicMock()]
        mock_client = MagicMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.list_mcp_toolsets",
                new=AsyncMock(return_value=fake_toolsets),
            ) as mock_list,
        ):
            result = await fetch_mcp_toolsets(user_api_key_dict=auth)

        assert result == fake_toolsets
        mock_list.assert_called_once_with(mock_client)

    @pytest.mark.asyncio
    async def test_non_admin_none_grants_returns_empty(self):
        """Non-admin key with no object_permission (field absent) gets no toolsets."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            fetch_mcp_toolsets,
        )

        auth = UserAPIKeyAuth(api_key="sk-test", object_permission=None)
        mock_client = MagicMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.list_mcp_toolsets",
                new=AsyncMock(return_value=[]),
            ) as mock_list,
        ):
            result = await fetch_mcp_toolsets(user_api_key_dict=auth)

        assert result == []
        mock_list.assert_not_called()

    @pytest.mark.asyncio
    async def test_populated_grants_filters_toolsets(self):
        """Key with explicit toolset IDs fetches only those IDs from the DB."""
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            fetch_mcp_toolsets,
        )

        auth = _make_auth(mcp_toolsets=["ts-1", "ts-2"])
        fake_toolsets = [MagicMock(toolset_id="ts-1"), MagicMock(toolset_id="ts-2")]
        mock_client = MagicMock()

        with (
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
                return_value=mock_client,
            ),
            patch(
                "litellm.proxy.management_endpoints.mcp_management_endpoints.list_mcp_toolsets",
                new=AsyncMock(return_value=fake_toolsets),
            ) as mock_list,
        ):
            result = await fetch_mcp_toolsets(user_api_key_dict=auth)

        assert len(result) == 2
        mock_list.assert_called_once_with(mock_client, toolset_ids=["ts-1", "ts-2"])


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

    @pytest.mark.asyncio
    async def test_client_header_is_stripped_in_scope(self):
        """handle_streamable_http_mcp strips x-mcp-toolset-id from scope before passing to session manager."""
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
        )

        scope = {
            "type": "http",
            "path": "/mcp",
            "method": "GET",
            "query_string": b"",
            "headers": [
                (b"authorization", b"Bearer sk-test"),
                (b"x-mcp-toolset-id", b"evil-toolset"),
                (b"content-type", b"application/json"),
            ],
        }
        mock_auth = UserAPIKeyAuth(api_key="sk-test")

        async def fake_receive():
            return {"type": "http.disconnect"}

        async def fake_send(msg):
            pass

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new=AsyncMock(
                    return_value=(mock_auth, None, [], {}, {}, scope["headers"])
                ),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.IPAddressUtils",
                MagicMock(get_mcp_client_ip=MagicMock(return_value="127.0.0.1")),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
                MagicMock(get_mcp_server_by_name=MagicMock(return_value=None)),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.MCPDebug",
                MagicMock(
                    maybe_build_debug_headers=MagicMock(return_value=None),
                ),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.set_auth_context",
                MagicMock(),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._handle_stale_mcp_session",
                new=AsyncMock(return_value=True),
            ),
        ):
            await handle_streamable_http_mcp(scope, fake_receive, fake_send)

        header_keys = [k for k, _ in scope["headers"]]
        assert b"x-mcp-toolset-id" not in header_keys
        assert b"authorization" in header_keys
        assert b"content-type" in header_keys
