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
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.resolve_toolset_tool_permissions",
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
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.resolve_toolset_tool_permissions",
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
        auth = UserAPIKeyAuth(api_key="sk-test", object_permission=op, user_role=user_role)

        resolve = AsyncMock(return_value={"server-a": ["tool1"]})
        with patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.resolve_toolset_tool_permissions",
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

    A toolset row names a tool on the server given by its ``server_id``, so the
    stored name is the tool's own name. The live tools come back carrying the
    server's wire prefix, so reconciling them strips that prefix from the LIVE
    name only; the stored name is matched as written. Reducing the stored name
    too renames the tool whenever a native name begins with its own server's
    prefix, which resolves the row to a different tool on the same server.
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
    async def test_filter_keeps_tools_when_prefix_contains_separator(self, alias, server_name, server_id):
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
                "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.get_allowed_tools_for_server",
                new=AsyncMock(return_value=allowed),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_id",
                return_value=server,
            ),
        ):
            result = await filter_tools_by_key_team_permissions(
                tools=live_tools,
                server_id=server_id,
                user_api_key_auth=_make_auth(),
            )

        assert sorted(t.name for t in result) == sorted(
            add_server_prefix_to_name(name, prefix) for name in ("read_wiki_contents", "read_wiki_structure")
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias, server_name, server_id", PREFIX_CASES)
    async def test_resolve_uses_the_stored_name_as_written(self, alias, server_name, server_id):
        """The row names a tool; resolution must not rewrite that name.

        A name that merely looks prefixed is still the tool's own name, and the
        server is already identified by ``server_id``, so there is nothing for a
        prefix to disambiguate.
        """
        server = self._server(alias, server_name, server_id)
        stored = "read_wiki_contents"

        assert await self._resolve(server, server_id, stored) == {server_id: [stored]}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias, server_name, server_id", PREFIX_CASES)
    async def test_resolve_keeps_a_name_that_looks_like_its_own_server_prefix(self, alias, server_name, server_id):
        from litellm.proxy._experimental.mcp_server.utils import (
            add_server_prefix_to_name,
            get_server_prefix,
        )

        server = self._server(alias, server_name, server_id)
        # A native tool whose own name begins with what the gateway would use as
        # this server's wire prefix.
        stored = add_server_prefix_to_name("read_wiki_contents", get_server_prefix(server))

        assert await self._resolve(server, server_id, stored) == {server_id: [stored]}

    @staticmethod
    async def _resolve(server, server_id, stored):
        from types import SimpleNamespace

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
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
            return await global_mcp_server_manager.resolve_toolset_tool_permissions(toolset_ids=["ts-1"])

    @pytest.mark.asyncio
    async def test_bare_stored_name_starting_with_server_prefix_stays_granted(self):
        """A native tool whose own name starts with ``{prefix}{separator}``.

        The dashboard persists the bare native name, so resolution must not read
        that leading segment as the server prefix and strip it. Doing so resolves
        the row to a different tool on the same server: the granted tool vanishes
        from the toolset and an ungranted sibling is served under its wire name.
        """
        from mcp.types import Tool as MCPTool

        from litellm.proxy._experimental.mcp_server.server import (
            filter_tools_by_key_team_permissions,
        )
        from litellm.proxy._experimental.mcp_server.utils import (
            add_server_prefix_to_name,
            get_server_prefix,
            strip_known_server_prefix,
        )

        server = self._server("deepwiki", None, "srv-collide")
        prefix = get_server_prefix(server)
        granted = add_server_prefix_to_name("contents", prefix)
        assert strip_known_server_prefix(granted, server) != granted, (
            "fixture must exercise the collision: the bare native name has to "
            "start with the server's own prefix plus the separator"
        )

        resolved = await self._resolve(server, "srv-collide", granted)

        sibling = "contents"
        live_tools = [
            MCPTool(
                name=add_server_prefix_to_name(name, prefix),
                inputSchema={"type": "object"},
            )
            for name in (granted, sibling)
        ]

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.get_allowed_tools_for_server",
                new=AsyncMock(return_value=resolved["srv-collide"]),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_id",
                return_value=server,
            ),
        ):
            kept = await filter_tools_by_key_team_permissions(
                tools=live_tools,
                server_id="srv-collide",
                user_api_key_auth=_make_auth(),
            )

        # Exactly the granted tool. ``sibling`` is a different tool on the same
        # server and was never selected, so it must not be reachable through
        # this row even though the stored name is its wire name.
        assert [t.name for t in kept] == [add_server_prefix_to_name(granted, prefix)]

    @pytest.mark.asyncio
    async def test_collision_row_grants_only_the_named_tool_when_no_sibling_exists(
        self,
    ):
        """Without the stripped sibling in the catalog there is nothing to widen.

        Resolution emits both readings of an ambiguous row, but a reading only
        grants a tool that actually exists on the server. A server exposing only
        the self-named tool therefore yields exactly that tool, which is the case
        that used to resolve to nothing at all.
        """
        from mcp.types import Tool as MCPTool

        from litellm.proxy._experimental.mcp_server.server import (
            filter_tools_by_key_team_permissions,
        )
        from litellm.proxy._experimental.mcp_server.utils import (
            add_server_prefix_to_name,
            get_server_prefix,
        )

        server = self._server("deepwiki", None, "srv-lonely")
        prefix = get_server_prefix(server)
        granted = add_server_prefix_to_name("contents", prefix)

        resolved = await self._resolve(server, "srv-lonely", granted)

        live_tools = [
            MCPTool(
                name=add_server_prefix_to_name(granted, prefix),
                inputSchema={"type": "object"},
            )
        ]

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler.get_allowed_tools_for_server",
                new=AsyncMock(return_value=resolved["srv-lonely"]),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_id",
                return_value=server,
            ),
        ):
            kept = await filter_tools_by_key_team_permissions(
                tools=live_tools,
                server_id="srv-lonely",
                user_api_key_auth=_make_auth(),
            )

        assert [t.name for t in kept] == [add_server_prefix_to_name(granted, prefix)]

    @pytest.mark.asyncio
    async def test_bare_stored_name_without_collision_grants_only_that_tool(self):
        """The ordinary row must stay exact; accepting both readings of an
        ambiguous row must not widen an unambiguous one."""
        from litellm.proxy._experimental.mcp_server.utils import get_server_prefix

        server = self._server("deepwiki", None, "srv-clean")
        assert get_server_prefix(server) == "deepwiki"

        resolved = await self._resolve(server, "srv-clean", "read_wiki_contents")

        assert resolved == {"srv-clean": ["read_wiki_contents"]}


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
                new=AsyncMock(return_value=(mock_auth, None, [], {}, {}, scope["headers"])),
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
