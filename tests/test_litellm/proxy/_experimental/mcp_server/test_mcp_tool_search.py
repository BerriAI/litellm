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
    get_virtual_tool_definitions,
    search_tools,
)
from litellm.proxy._types import UserAPIKeyAuth


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
            if hasattr(r, "path")
            and r.path.endswith("/tools/list")
            and hasattr(r, "methods")
            and "GET" in r.methods
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
            if hasattr(r, "path")
            and r.path.endswith("/tools/list")
            and hasattr(r, "methods")
            and "GET" in r.methods
        )

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.build_effective_auth_contexts",
                new_callable=AsyncMock,
                return_value=[user_api_key_dict],
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.global_mcp_server_manager"
            ) as mock_manager,
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints._get_tools_for_single_server",
                new_callable=AsyncMock,
                return_value=fake_tools,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.IPAddressUtils"
            ),
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
            mock_manager.filter_server_ids_by_ip_with_info = MagicMock(
                return_value=(["github"], 0)
            )
            mock_manager.get_mcp_server_by_id = MagicMock(
                return_value=MagicMock(name="github", server_id="github")
            )
            result = await list_fn(
                request=mock_request,
                server_id=None,
                include_disabled_tools=False,
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
            if hasattr(r, "path")
            and r.path.endswith("/tools/call")
            and hasattr(r, "methods")
            and "POST" in r.methods
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

        request = self._make_request(
            {"name": MCP_TOOL_SEARCH_TOOL_NAME, "arguments": {"query": "create issue"}}
        )

        mock_tool = MagicMock()
        mock_tool.name = "github-create_issue"
        mock_tool.description = "Create a GitHub issue"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        with patch(
            "litellm.proxy._experimental.mcp_server.tool_search.global_mcp_server_manager"
        ) as mock_manager:
            mock_manager.list_tools = AsyncMock(return_value=[mock_tool])

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
                "litellm.proxy._experimental.mcp_server.tool_search.global_mcp_server_manager"
            ) as mock_manager,
            patch(
                "litellm.proxy._experimental.mcp_server.tool_search.build_effective_auth_contexts",
                new_callable=AsyncMock,
                return_value=[user_api_key_dict],
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
                new_callable=AsyncMock,
                return_value=fake_result,
            ),
        ):
            mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["github"])
            mock_manager.get_registry = MagicMock(return_value={})

            result = await self._get_call_fn()(
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        assert result.isError is False
        assert result.content[0].text == "Issue created"

    @pytest.mark.asyncio
    async def test_mcp_tool_search_requires_flag_enabled(self) -> None:
        from fastapi import HTTPException

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            object_permission=_make_perm(mcp_tool_search_enabled=False),
        )

        request = self._make_request(
            {"name": MCP_TOOL_SEARCH_TOOL_NAME, "arguments": {"query": "create issue"}}
        )

        with pytest.raises(HTTPException) as exc_info:
            await self._get_call_fn()(
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        assert exc_info.value.status_code in (400, 403, 404)
