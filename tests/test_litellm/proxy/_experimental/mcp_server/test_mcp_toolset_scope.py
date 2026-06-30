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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("user_role", [None, LitellmUserRoles.PROXY_ADMIN.value])
    async def test_no_mcp_servers_sentinel_denies_toolset_access(self, user_role):
        """A key scoped to the no-mcp-servers sentinel cannot reach a toolset it
        would otherwise be granted (even as admin); the opt-out covers the
        toolset path, which replaces mcp_servers and would drop the sentinel."""
        from starlette.exceptions import HTTPException

        from litellm.proxy._experimental.mcp_server.server import _apply_toolset_scope

        op = LiteLLM_ObjectPermissionTable(
            object_permission_id="test",
            mcp_servers=["no-mcp-servers"],
            mcp_toolsets=["toolset-123"],
        )
        auth = UserAPIKeyAuth(
            api_key="sk-test", object_permission=op, user_role=user_role
        )

        resolve = AsyncMock(return_value={"server-a": ["tool1"]})
        with patch(
            "litellm.proxy._experimental.mcp_server.server."
            "global_mcp_server_manager.resolve_toolset_tool_permissions",
            new=resolve,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _apply_toolset_scope(auth, "toolset-123")

        assert exc_info.value.status_code == 403
        resolve.assert_not_awaited()


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


class TestToolsetPrefixResolution:
    """Regression for LIT-3419.

    Toolsets store bare tool names; the live tools come back prefixed with the
    server's own prefix. Reconciling them must strip exactly that prefix, not
    chop at the first separator, otherwise tools on a server whose prefix
    contains the separator (a hyphenated alias, or the UUID server_id used when
    a server has no alias) are silently dropped from the toolset.
    """

    # alias, server_name, server_id; the clean-alias row worked before the fix,
    # the hyphenated-alias and no-alias (UUID prefix) rows did not.
    PREFIX_CASES = [
        ("deepwiki", None, "srv-clean"),
        ("deep-wiki", None, "srv-hyphen"),
        (None, None, "117c814c-1a2b-3c4d-9e8f"),
    ]

    @staticmethod
    def _server(alias, server_name, server_id):
        from types import SimpleNamespace

        return SimpleNamespace(
            alias=alias,
            server_name=server_name,
            server_id=server_id,
            short_prefix=None,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias, server_name, server_id", PREFIX_CASES)
    async def test_filter_keeps_tools_when_prefix_contains_separator(
        self, alias, server_name, server_id
    ):
        from mcp.types import Tool as MCPTool

        from litellm.proxy._experimental.mcp_server.server import (
            filter_tools_by_key_team_permissions,
        )
        from litellm.proxy._experimental.mcp_server.utils import (
            add_server_prefix_to_name,
            get_server_prefix,
        )

        server = self._server(alias, server_name, server_id)
        prefix = get_server_prefix(server)
        live_tools = [
            MCPTool(
                name=add_server_prefix_to_name(name, prefix),
                inputSchema={"type": "object"},
            )
            for name in ("read_wiki_contents", "read_wiki_structure", "not_granted")
        ]
        # Bare names as stored in the toolset / resolved into the permission dict.
        allowed = ["read_wiki_contents", "read_wiki_structure"]

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server."
                "MCPRequestHandler.get_allowed_tools_for_server",
                new=AsyncMock(return_value=allowed),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server."
                "global_mcp_server_manager.get_mcp_server_by_id",
                return_value=server,
            ),
        ):
            result = await filter_tools_by_key_team_permissions(
                tools=live_tools,
                server_id=server_id,
                user_api_key_auth=_make_auth(),
            )

        assert sorted(t.name for t in result) == sorted(
            add_server_prefix_to_name(name, prefix)
            for name in ("read_wiki_contents", "read_wiki_structure")
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias, server_name, server_id", PREFIX_CASES)
    async def test_resolve_unprefixes_stored_names_with_separator_prefix(
        self, alias, server_name, server_id
    ):
        from types import SimpleNamespace

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._experimental.mcp_server.utils import (
            add_server_prefix_to_name,
            get_server_prefix,
        )

        server = self._server(alias, server_name, server_id)
        # A caller (e.g. the management API) may persist already-prefixed names;
        # resolution must reduce them to the true bare name regardless of prefix.
        stored = add_server_prefix_to_name(
            "read_wiki_contents", get_server_prefix(server)
        )
        toolset = SimpleNamespace(tools=[{"server_id": server_id, "tool_name": stored}])
        cache = MagicMock(
            async_get_cache=AsyncMock(return_value=None),
            async_set_cache=AsyncMock(),
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager."
                "global_mcp_server_manager.get_mcp_server_by_id",
                return_value=server,
            ),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
            patch(
                "litellm.proxy._experimental.mcp_server.toolset_db.list_mcp_toolsets",
                new=AsyncMock(return_value=[toolset]),
            ),
        ):
            result = await global_mcp_server_manager.resolve_toolset_tool_permissions(
                toolset_ids=["ts-1"]
            )

        assert result == {server_id: ["read_wiki_contents"]}


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
