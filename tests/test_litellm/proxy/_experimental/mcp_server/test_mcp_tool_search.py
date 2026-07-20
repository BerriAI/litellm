"""
Tests for MCP tool search feature.

Covers:
- search_tools() pure function
- get_virtual_tool_definitions() shape
- list_tool_rest_api returns only virtual tools when mcp_tool_search_enabled=True
- call_tool_rest_api intercepts mcp_tool_search calls
- call_tool_rest_api intercepts mcp_tool_call calls
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._experimental.mcp_server.tool_search import (
    MCP_TOOL_CALL_TOOL_NAME,
    MCP_TOOL_SEARCH_TOOL_NAME,
    coerce_top_k,
    get_virtual_tool_definitions,
    search_tools,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _make_tools(specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": desc,
            "inputSchema": {"type": "object", "properties": {}},
        }
        for name, desc in specs
    ]


def _make_perm(**kwargs: Any) -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(object_permission_id="test", **kwargs)


SAMPLE_TOOLS = _make_tools(
    [
        ("github-create_issue", "Create a new issue in a GitHub repository"),
        ("github-list_repos", "List all repositories for a GitHub user"),
        ("slack-send_message", "Send a message to a Slack channel"),
        ("slack-list_channels", "List all Slack channels in a workspace"),
        ("notion-create_page", "Create a new page in Notion"),
    ]
)


class TestCoerceTopK:
    def test_int_passthrough(self) -> None:
        assert coerce_top_k(3) == 3

    def test_numeric_string_coerced(self) -> None:
        assert coerce_top_k("7") == 7

    def test_float_truncated(self) -> None:
        assert coerce_top_k(3.9) == 3

    def test_non_numeric_string_returns_default(self) -> None:
        assert coerce_top_k("abc") == 5

    def test_none_returns_default(self) -> None:
        assert coerce_top_k(None) == 5

    def test_custom_default(self) -> None:
        assert coerce_top_k("nope", default=10) == 10


class TestSearchTools:
    def test_returns_matching_tools(self) -> None:
        results = search_tools("github issue", SAMPLE_TOOLS)
        names = [t["name"] for t in results]
        assert "github-create_issue" in names

    def test_ranks_by_relevance(self) -> None:
        results = search_tools("github", SAMPLE_TOOLS)
        names = [t["name"] for t in results]
        github_positions = [i for i, n in enumerate(names) if n.startswith("github")]
        other_positions = [i for i, n in enumerate(names) if not n.startswith("github")]
        assert all(g < o for g in github_positions for o in other_positions)

    def test_top_k_limits_results(self) -> None:
        results = search_tools("a", SAMPLE_TOOLS, top_k=2)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self) -> None:
        assert search_tools("", SAMPLE_TOOLS) == []

    def test_no_match_returns_empty(self) -> None:
        assert search_tools("xyzzy_nonexistent_zzz", SAMPLE_TOOLS) == []

    def test_matches_description_not_just_name(self) -> None:
        results = search_tools("channel", SAMPLE_TOOLS)
        names = [t["name"] for t in results]
        assert "slack-list_channels" in names

    def test_case_insensitive(self) -> None:
        lower = [t["name"] for t in search_tools("github", SAMPLE_TOOLS)]
        upper = [t["name"] for t in search_tools("GITHUB", SAMPLE_TOOLS)]
        assert lower == upper

    def test_result_tools_have_full_schema(self) -> None:
        for tool in search_tools("github", SAMPLE_TOOLS):
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_exact_word_outranks_prefix_false_positive(self) -> None:
        tools = _make_tools(
            [
                ("address-lookup", "Look up a contact in the address book"),
                ("math-add", "Add two numbers together"),
            ]
        )
        results = search_tools("add", tools)
        assert results[0]["name"] == "math-add"

    def test_name_match_outranks_description_match(self) -> None:
        tools = _make_tools(
            [
                ("files-list", "Search the file index"),
                ("code-search", "Find matches in the repository"),
            ]
        )
        results = search_tools("search", tools)
        assert results[0]["name"] == "code-search"

    def test_rare_token_outranks_common_token(self) -> None:
        tools = _make_tools(
            [
                ("files-list", "List all files"),
                ("repos-list", "List repositories"),
                ("channels-list", "List channels"),
                ("email-send", "Send an email message"),
            ]
        )
        results = search_tools("list email", tools)
        assert results[0]["name"] == "email-send"

    def test_tie_break_is_deterministic_by_name(self) -> None:
        specs = [
            ("b-tool", "Fetch the weather"),
            ("a-tool", "Fetch the weather"),
        ]
        forward = [t["name"] for t in search_tools("weather", _make_tools(specs))]
        reverse = [t["name"] for t in search_tools("weather", _make_tools(specs[::-1]))]
        assert forward == reverse == ["a-tool", "b-tool"]

    def test_hyphen_and_underscore_are_token_boundaries(self) -> None:
        results = search_tools("create_issue", SAMPLE_TOOLS)
        assert results[0]["name"] == "github-create_issue"

    def test_non_positive_top_k_returns_empty(self) -> None:
        assert search_tools("github", SAMPLE_TOOLS, top_k=0) == []
        assert search_tools("github", SAMPLE_TOOLS, top_k=-1) == []

    def test_repeated_query_token_not_double_counted(self) -> None:
        single = [t["name"] for t in search_tools("github issue", SAMPLE_TOOLS)]
        repeated = [t["name"] for t in search_tools("github github issue", SAMPLE_TOOLS)]
        assert single == repeated

    def test_query_tokens_capped_at_32(self) -> None:
        filler = " ".join(f"junk{i}" for i in range(32))
        assert search_tools(f"{filler} github", SAMPLE_TOOLS) == []
        capped = [t["name"] for t in search_tools(f"github {filler}", SAMPLE_TOOLS)]
        assert "github-create_issue" in capped


class TestGetVirtualToolDefinitions:
    def test_returns_two_tools(self) -> None:
        assert len(get_virtual_tool_definitions()) == 2

    def test_has_mcp_tool_search(self) -> None:
        names = [t["name"] for t in get_virtual_tool_definitions()]
        assert MCP_TOOL_SEARCH_TOOL_NAME in names

    def test_has_mcp_tool_call(self) -> None:
        names = [t["name"] for t in get_virtual_tool_definitions()]
        assert MCP_TOOL_CALL_TOOL_NAME in names

    def test_mcp_tool_search_schema_has_query(self) -> None:
        tools = get_virtual_tool_definitions()
        search_tool = next(t for t in tools if t["name"] == MCP_TOOL_SEARCH_TOOL_NAME)
        props = search_tool["inputSchema"]["properties"]
        assert "query" in props
        assert search_tool["inputSchema"]["required"] == ["query"]

    def test_mcp_tool_call_schema_has_tool_name_and_arguments(self) -> None:
        tools = get_virtual_tool_definitions()
        call_tool = next(t for t in tools if t["name"] == MCP_TOOL_CALL_TOOL_NAME)
        props = call_tool["inputSchema"]["properties"]
        assert "tool_name" in props
        assert "arguments" in props
        assert "tool_name" in call_tool["inputSchema"]["required"]

    def test_all_tools_have_description(self) -> None:
        for tool in get_virtual_tool_definitions():
            assert tool.get("description"), f"{tool['name']} missing description"

    def test_definitions_construct_mcp_protocol_tool(self) -> None:
        """The MCP protocol list_tools handler builds mcp.types.Tool(**d) from
        each definition, so the dict keys must stay valid Tool fields."""
        from mcp.types import Tool

        built = [Tool(**d) for d in get_virtual_tool_definitions()]
        assert {t.name for t in built} == {
            MCP_TOOL_SEARCH_TOOL_NAME,
            MCP_TOOL_CALL_TOOL_NAME,
        }


class TestListToolRestApiWithToolSearch:
    @pytest.mark.asyncio
    async def test_returns_only_virtual_tools_when_flag_enabled(self) -> None:
        from litellm.proxy._experimental.mcp_server.rest_endpoints import router

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(
                mcp_tool_search_enabled=True,
                mcp_servers=["github", "slack"],
            ),
        )

        mock_request = MagicMock()
        mock_request.headers = {}

        list_fn = next(
            r.endpoint
            for r in router.routes
            if hasattr(r, "path") and r.path.endswith("/tools/list") and hasattr(r, "methods") and "GET" in r.methods
        )

        result = await list_fn(
            request=mock_request,
            server_id=None,
            include_disabled_tools=False,
            user_api_key_dict=user_api_key_dict,
        )

        assert result["error"] is None
        tool_names = [t["name"] for t in result["tools"]]
        assert set(tool_names) == {MCP_TOOL_SEARCH_TOOL_NAME, MCP_TOOL_CALL_TOOL_NAME}

    @pytest.mark.asyncio
    async def test_returns_full_catalog_when_flag_disabled(self) -> None:
        from litellm.proxy._experimental.mcp_server.rest_endpoints import router

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(
                mcp_tool_search_enabled=False,
                mcp_servers=["github"],
            ),
        )

        mock_request = MagicMock()
        mock_request.headers = {}

        fake_tools = [
            {
                "name": "github-create_issue",
                "description": "Create issue",
                "inputSchema": {"type": "object"},
            }
        ]

        list_fn = next(
            r.endpoint
            for r in router.routes
            if hasattr(r, "path") and r.path.endswith("/tools/list") and hasattr(r, "methods") and "GET" in r.methods
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.build_effective_auth_contexts",
                new_callable=AsyncMock,
                return_value=[user_api_key_dict],
            ),
            patch("litellm.proxy._experimental.mcp_server.rest_endpoints.global_mcp_server_manager") as mock_manager,
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_tools_for_single_server",
                new_callable=AsyncMock,
                return_value=fake_tools,
            ),
            patch("litellm.proxy._experimental.mcp_server.rest_endpoints.IPAddressUtils"),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._prefetch_user_oauth_creds",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_oauth2_server_ids",
                return_value=[],
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_server_auth_header",
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_user_oauth_extra_headers",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
            mock_manager.filter_server_ids_by_ip_with_info = MagicMock(return_value=(["github"], 0))
            mock_manager.get_mcp_server_by_id = MagicMock(return_value=MagicMock(name="github", server_id="github"))
            result = await list_fn(
                request=mock_request,
                server_id=None,
                include_disabled_tools=False,
                user_api_key_dict=user_api_key_dict,
            )

        tool_names = [t["name"] for t in result["tools"]]
        assert MCP_TOOL_SEARCH_TOOL_NAME not in tool_names
        assert "github-create_issue" in tool_names

    @pytest.mark.asyncio
    async def test_admin_include_disabled_tools_bypasses_virtual_catalog(self) -> None:
        """Regression: an admin listing with include_disabled_tools must see the
        real catalog (to configure allowlists) even when mcp_tool_search_enabled is
        set, instead of the two virtual tools."""
        from litellm.proxy._experimental.mcp_server.rest_endpoints import router

        user_api_key_dict = UserAPIKeyAuth(
            api_key="admin_key",
            user_role=LitellmUserRoles.PROXY_ADMIN,
            object_permission=_make_perm(
                mcp_tool_search_enabled=True,
                mcp_servers=["github"],
            ),
        )

        mock_request = MagicMock()
        mock_request.headers = {}

        fake_tools = [
            {
                "name": "github-create_issue",
                "description": "Create issue",
                "inputSchema": {"type": "object"},
            }
        ]

        list_fn = next(
            r.endpoint
            for r in router.routes
            if hasattr(r, "path") and r.path.endswith("/tools/list") and hasattr(r, "methods") and "GET" in r.methods
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.build_effective_auth_contexts",
                new_callable=AsyncMock,
                return_value=[user_api_key_dict],
            ),
            patch("litellm.proxy._experimental.mcp_server.rest_endpoints.global_mcp_server_manager") as mock_manager,
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_tools_for_single_server",
                new_callable=AsyncMock,
                return_value=fake_tools,
            ),
            patch("litellm.proxy._experimental.mcp_server.rest_endpoints.IPAddressUtils"),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._prefetch_user_oauth_creds",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_oauth2_server_ids",
                return_value=[],
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_server_auth_header",
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_user_oauth_extra_headers",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
            mock_manager.filter_server_ids_by_ip_with_info = MagicMock(return_value=(["github"], 0))
            mock_manager.get_mcp_server_by_id = MagicMock(return_value=MagicMock(name="github", server_id="github"))
            result = await list_fn(
                request=mock_request,
                server_id=None,
                include_disabled_tools=True,
                user_api_key_dict=user_api_key_dict,
            )

        tool_names = [t["name"] for t in result["tools"]]
        assert MCP_TOOL_SEARCH_TOOL_NAME not in tool_names
        assert "github-create_issue" in tool_names


class TestCallToolRestApiVirtualTools:
    def _make_request(self, body: dict[str, Any]) -> MagicMock:
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value=body)
        mock_request.headers = {}
        mock_request.url = MagicMock()
        mock_request.url.path = "/mcp-rest/tools/call"
        return mock_request

    def _get_call_fn(self) -> Any:
        from litellm.proxy._experimental.mcp_server.rest_endpoints import router

        return next(
            r.endpoint
            for r in router.routes
            if hasattr(r, "path") and r.path.endswith("/tools/call") and hasattr(r, "methods") and "POST" in r.methods
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_search_call_returns_tool_defs(self) -> None:
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(
                mcp_tool_search_enabled=True,
                mcp_servers=["github"],
            ),
        )

        request = self._make_request({"name": MCP_TOOL_SEARCH_TOOL_NAME, "arguments": {"query": "create issue"}})

        mock_tool = MagicMock()
        mock_tool.name = "github-create_issue"
        mock_tool.description = "Create a GitHub issue"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        with patch(
            "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
            new_callable=AsyncMock,
            return_value=[mock_tool],
        ):
            result = await self._get_call_fn()(
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        assert result.content
        assert result.content[0].type == "text"
        returned_tools = json.loads(result.content[0].text)
        assert isinstance(returned_tools, list)
        assert any(t["name"] == "github-create_issue" for t in returned_tools)

    @pytest.mark.asyncio
    async def test_mcp_tool_call_executes_discovered_tool(self) -> None:
        from mcp.types import CallToolResult, TextContent

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(
                mcp_tool_search_enabled=True,
                mcp_servers=["github"],
            ),
        )

        request = self._make_request(
            {
                "name": MCP_TOOL_CALL_TOOL_NAME,
                "arguments": {
                    "tool_name": "github-create_issue",
                    "arguments": {"title": "bug", "repo": "myrepo"},
                },
            }
        )

        fake_result = CallToolResult(
            content=[TextContent(type="text", text="Issue created")],
            isError=False,
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=[MagicMock()],
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
                new_callable=AsyncMock,
                return_value=fake_result,
            ) as mock_execute,
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._fire_mcp_tool_call_logging",
                new_callable=AsyncMock,
                side_effect=RuntimeError("logging failed"),
            ) as mock_fire_logging,
        ):
            result = await self._get_call_fn()(
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        mock_execute.assert_awaited_once()
        mock_fire_logging.assert_awaited_once()
        assert mock_execute.await_args.kwargs["name"] == "github-create_issue"

        assert result.isError is False
        assert result.content[0].text == "Issue created"

    @pytest.mark.asyncio
    async def test_mcp_tool_call_forwards_client_ip_for_ip_filtering(self) -> None:
        """Regression: the virtual call path must resolve allowed servers with the
        request's client IP so IP-restricted servers (available_on_public_internet:
        false) cannot be reached from a public IP."""
        from mcp.types import CallToolResult, TextContent

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(mcp_tool_search_enabled=True, mcp_servers=["github"]),
        )
        request = self._make_request(
            {
                "name": MCP_TOOL_CALL_TOOL_NAME,
                "arguments": {"tool_name": "github-create_issue", "arguments": {}},
            }
        )

        fake_result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.IPAddressUtils.get_mcp_client_ip",
                return_value="203.0.113.7",
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=[MagicMock()],
            ) as mock_allowed,
            patch(
                "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
                new_callable=AsyncMock,
                return_value=fake_result,
            ),
        ):
            await self._get_call_fn()(request=request, user_api_key_dict=user_api_key_dict)

        mock_allowed.assert_awaited_once()
        assert mock_allowed.await_args.kwargs["client_ip"] == "203.0.113.7"

    @pytest.mark.asyncio
    async def test_mcp_tool_search_forwards_client_ip_for_ip_filtering(self) -> None:
        """Search must list tools through the IP-filtered catalog, not the raw one."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(mcp_tool_search_enabled=True, mcp_servers=["github"]),
        )
        request = self._make_request({"name": MCP_TOOL_SEARCH_TOOL_NAME, "arguments": {"query": "issue"}})

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.IPAddressUtils.get_mcp_client_ip",
                return_value="203.0.113.7",
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_list,
        ):
            await self._get_call_fn()(request=request, user_api_key_dict=user_api_key_dict)

        mock_list.assert_awaited_once()
        assert mock_list.await_args.kwargs["client_ip"] == "203.0.113.7"

    @pytest.mark.asyncio
    async def test_mcp_tool_search_requires_flag_enabled(self) -> None:
        from fastapi import HTTPException

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(mcp_tool_search_enabled=False),
        )

        request = self._make_request({"name": MCP_TOOL_SEARCH_TOOL_NAME, "arguments": {"query": "create issue"}})

        with pytest.raises(HTTPException) as exc_info:
            await self._get_call_fn()(
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        assert exc_info.value.status_code in (400, 403, 404)


class TestDispatchVirtualMcpTool:
    """Covers the SSE/protocol-path interception helper in server.py."""

    @pytest.mark.asyncio
    async def test_returns_none_for_non_virtual_tool(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _dispatch_virtual_mcp_tool,
        )

        result = await _dispatch_virtual_mcp_tool(
            name="github-create_issue",
            arguments={},
            user_api_key_auth=UserAPIKeyAuth(api_key="k"),
            client_ip=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_when_flag_disabled(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _dispatch_virtual_mcp_tool,
        )

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=False))
        result = await _dispatch_virtual_mcp_tool(
            name=MCP_TOOL_SEARCH_TOOL_NAME,
            arguments={"query": "x"},
            user_api_key_auth=uak,
            client_ip=None,
        )
        assert result is not None
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_routes_search_with_client_ip(self) -> None:
        from litellm.proxy._experimental.mcp_server import server as srv

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        with patch(
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_search",
            new_callable=AsyncMock,
            return_value="SEARCH_RESULT",
        ) as mock_search:
            result = await srv._dispatch_virtual_mcp_tool(
                name=MCP_TOOL_SEARCH_TOOL_NAME,
                arguments={"query": "q", "top_k": 3},
                user_api_key_auth=uak,
                client_ip="203.0.113.9",
            )

        assert result == "SEARCH_RESULT"
        assert mock_search.await_args.kwargs["client_ip"] == "203.0.113.9"
        assert mock_search.await_args.kwargs["query"] == "q"
        assert mock_search.await_args.kwargs["top_k"] == 3

    @pytest.mark.asyncio
    async def test_routes_call_with_client_ip(self) -> None:
        from litellm.proxy._experimental.mcp_server import server as srv

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        with patch(
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call",
            new_callable=AsyncMock,
            return_value="CALL_RESULT",
        ) as mock_call:
            result = await srv._dispatch_virtual_mcp_tool(
                name=MCP_TOOL_CALL_TOOL_NAME,
                arguments={"tool_name": "math-add", "arguments": {"a": 1, "b": 2}},
                user_api_key_auth=uak,
                client_ip="203.0.113.9",
                mcp_auth_header="bearer-xyz",
                mcp_server_auth_headers={"github": {"Authorization": "Bearer gh"}},
                oauth2_headers={"Authorization": "Bearer oauth"},
                raw_headers={"x-mcp-auth": "tok"},
            )

        assert result == "CALL_RESULT"
        kw = mock_call.await_args.kwargs
        assert kw["tool_name"] == "math-add"
        assert kw["client_ip"] == "203.0.113.9"
        assert kw["mcp_auth_header"] == "bearer-xyz"
        assert kw["mcp_server_auth_headers"] == {"github": {"Authorization": "Bearer gh"}}
        assert kw["oauth2_headers"] == {"Authorization": "Bearer oauth"}
        assert kw["raw_headers"] == {"x-mcp-auth": "tok"}

    @pytest.mark.asyncio
    async def test_call_builds_and_forwards_logging_obj(self) -> None:
        """Regression: the SSE dispatch must run the pre-call pipeline and forward
        the resulting logging object to handle_mcp_tool_call, otherwise mcp_tool_call
        over /mcp/ skips spend logging and guardrails (unlike the REST path)."""
        from litellm.proxy._experimental.mcp_server import server as srv

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        sentinel_logging_obj = object()
        with (
            patch.object(
                srv,
                "_build_virtual_call_logging_obj",
                new_callable=AsyncMock,
                return_value=sentinel_logging_obj,
            ) as mock_build,
            patch(
                "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call",
                new_callable=AsyncMock,
                return_value="CALL_RESULT",
            ) as mock_call,
        ):
            await srv._dispatch_virtual_mcp_tool(
                name=MCP_TOOL_CALL_TOOL_NAME,
                arguments={"tool_name": "math-add", "arguments": {"a": 1}},
                user_api_key_auth=uak,
                client_ip=None,
            )

        assert mock_build.await_count == 1
        assert mock_call.await_args.kwargs["litellm_logging_obj"] is sentinel_logging_obj

    @pytest.mark.asyncio
    async def test_search_coerces_non_int_top_k(self) -> None:
        """Regression: a non-integer top_k from an MCP client must not raise; it
        falls back to the default instead of ValueError propagating out."""
        from litellm.proxy._experimental.mcp_server import server as srv

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        with patch(
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_search",
            new_callable=AsyncMock,
            return_value="SEARCH_RESULT",
        ) as mock_search:
            await srv._dispatch_virtual_mcp_tool(
                name=MCP_TOOL_SEARCH_TOOL_NAME,
                arguments={"query": "issue", "top_k": "not-a-number"},
                user_api_key_auth=uak,
                client_ip=None,
            )

        assert mock_search.await_args.kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_call_handler_forwards_auth_headers_to_execute(self) -> None:
        """Regression: per-request auth headers must reach execute_mcp_tool so
        upstream MCP servers needing pass-through auth can be called."""
        from mcp.types import CallToolResult, TextContent

        from litellm.proxy._experimental.mcp_server.tool_search import (
            handle_mcp_tool_call,
        )

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        fake = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=[MagicMock()],
            ) as mock_allowed,
            patch(
                "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
                new_callable=AsyncMock,
                return_value=fake,
            ) as mock_exec,
        ):
            sentinel_logging_obj = object()
            await handle_mcp_tool_call(
                tool_name="github-create_issue",
                arguments={},
                user_api_key_dict=uak,
                mcp_servers=["github"],
                mcp_auth_header="bearer-xyz",
                mcp_server_auth_headers={"github": {"Authorization": "Bearer gh"}},
                oauth2_headers={"Authorization": "Bearer oauth"},
                raw_headers={"x-mcp-auth": "tok"},
                litellm_logging_obj=sentinel_logging_obj,
            )

        kw = mock_exec.await_args.kwargs
        assert kw["mcp_auth_header"] == "bearer-xyz"
        assert kw["mcp_server_auth_headers"] == {"github": {"Authorization": "Bearer gh"}}
        assert kw["oauth2_headers"] == {"Authorization": "Bearer oauth"}
        assert kw["raw_headers"] == {"x-mcp-auth": "tok"}
        # Spend logging: the logging object must reach execute_mcp_tool
        assert kw["litellm_logging_obj"] is sentinel_logging_obj
        # Scoped session: the requested mcp_servers scope must reach server resolution
        assert mock_allowed.await_args.kwargs["mcp_servers"] == ["github"]

    @pytest.mark.asyncio
    async def test_call_rejected_when_no_accessible_servers(self) -> None:
        """Regression: a key with no accessible MCP servers must not reach
        execute_mcp_tool, where an unprefixed local tool name would otherwise
        run via the local registry without a server permission check."""
        from fastapi import HTTPException

        from litellm.proxy._experimental.mcp_server.tool_search import (
            handle_mcp_tool_call,
        )

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_mcp_tool_call(
                    tool_name="local_secret_tool",
                    arguments={},
                    user_api_key_dict=uak,
                )

        assert exc_info.value.status_code == 403
        mock_exec.assert_not_awaited()


class TestCaptureHostProgressCallback:
    """Covers the host progress-forwarding helper extracted from the tool call path."""

    def test_returns_none_when_request_context_unavailable(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _capture_host_progress_callback,
        )

        class _NoCtx:
            @property
            def request_context(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("no context")

        assert _capture_host_progress_callback(_NoCtx()) is None

    def test_returns_none_when_no_progress_token(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _capture_host_progress_callback,
        )

        host = MagicMock()
        host.request_context.meta.progressToken = None
        assert _capture_host_progress_callback(host) is None

    def test_returns_callable_when_token_present(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _capture_host_progress_callback,
        )

        host = MagicMock()
        host.request_context.meta.progressToken = "tok12345"
        host.request_context.session = MagicMock()
        assert callable(_capture_host_progress_callback(host))

    def test_returns_callable_when_token_is_integer(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _capture_host_progress_callback,
        )

        host = MagicMock()
        host.request_context.meta.progressToken = 12345
        host.request_context.session = MagicMock()
        assert callable(_capture_host_progress_callback(host))

    def test_returns_callable_when_token_is_zero(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _capture_host_progress_callback,
        )

        host = MagicMock()
        host.request_context.meta.progressToken = 0
        host.request_context.session = MagicMock()
        assert callable(_capture_host_progress_callback(host))

    @pytest.mark.asyncio
    async def test_forwarded_progress_token_preserves_integer_value(self) -> None:
        from litellm.proxy._experimental.mcp_server.server import (
            _capture_host_progress_callback,
        )

        host = MagicMock()
        host.request_context.meta.progressToken = 12345
        session = AsyncMock()
        host.request_context.session = session

        callback = _capture_host_progress_callback(host)
        assert callback is not None
        await callback(0.5, 1.0)

        session.send_progress_notification.assert_awaited_once_with(
            progress_token=12345,
            progress=0.5,
            total=1.0,
        )


class TestHandleListToolsVirtual:
    """Covers the protocol list_tools early-return when the flag is enabled."""

    @pytest.mark.asyncio
    async def test_returns_virtual_tools_when_flag_enabled(self) -> None:
        from litellm.proxy._experimental.mcp_server import server as srv

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        with patch(
            "litellm.proxy._experimental.mcp_server.server.get_or_extract_auth_context",
            new_callable=AsyncMock,
            return_value=(uak, None, None, None, None, None, None),
        ):
            tools = await srv.handle_list_tools()

        assert {t.name for t in tools} == {
            MCP_TOOL_SEARCH_TOOL_NAME,
            MCP_TOOL_CALL_TOOL_NAME,
        }


class TestMcpServerToolCallErrorHandling:
    """The protocol tool-call handler must convert virtual-tool errors to an
    isError CallToolResult instead of letting them raise out of the handler."""

    @pytest.mark.asyncio
    async def test_virtual_tool_error_returns_iserror_not_raised(self) -> None:
        from fastapi import HTTPException

        from litellm.proxy._experimental.mcp_server import server as srv

        uak = UserAPIKeyAuth(api_key="k", object_permission=_make_perm(mcp_tool_search_enabled=True))
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.get_or_extract_auth_context",
                new_callable=AsyncMock,
                return_value=(uak, None, None, None, None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._dispatch_virtual_mcp_tool",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=403, detail="User not allowed to call this tool"),
            ),
        ):
            result = await srv.mcp_server_tool_call(
                name=MCP_TOOL_CALL_TOOL_NAME,
                arguments={"tool_name": "other-server-tool", "arguments": {}},
            )

        assert result.isError is True
        assert "User not allowed to call this tool" in result.content[0].text
