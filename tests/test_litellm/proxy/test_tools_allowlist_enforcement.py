"""
Tests for tool allowlist enforcement (key/team metadata.allowed_tools).

Covers:
- check_tools_allowlist: allowed, disallowed, no allowlist, non-tool routes
- extract_request_tool_names: OpenAI chat, responses, Anthropic, generate_content, MCP
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import check_tools_allowlist
from litellm.proxy.guardrails.tool_name_extraction import (
    TOOL_CAPABLE_CALL_TYPES,
    extract_request_tool_names,
)


def _token(metadata=None, team_metadata=None):
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="user",
        team_id="team",
        org_id=None,
        models=["*"],
        metadata=metadata or {},
        team_metadata=team_metadata or {},
    )


class TestExtractRequestToolNames:
    """Test tool name extraction per API format."""

    def test_openai_chat_tools(self):
        data = {
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "function", "function": {"name": "run_sql"}},
            ]
        }
        assert extract_request_tool_names("/v1/chat/completions", data) == [
            "get_weather",
            "run_sql",
        ]

    def test_openai_chat_functions_legacy(self):
        data = {"functions": [{"name": "get_weather"}, {"name": "run_sql"}]}
        assert extract_request_tool_names("/v1/chat/completions", data) == [
            "get_weather",
            "run_sql",
        ]

    def test_openai_responses_function_tools(self):
        data = {
            "tools": [
                {"type": "function", "name": "get_current_weather", "description": "x"},
            ]
        }
        assert extract_request_tool_names("/v1/responses", data) == ["get_current_weather"]

    def test_openai_responses_mcp_tools(self):
        data = {
            "tools": [
                {"type": "mcp", "server_label": "dmcp", "server_url": "http://x"},
            ]
        }
        assert extract_request_tool_names("/v1/responses", data) == ["dmcp"]

    def test_openai_responses_custom_tools(self):
        """Custom tools become callable function tools on the Chat Completions
        bridge, so their names must be extracted for allowlist enforcement;
        otherwise a restricted key could invoke a disallowed tool by declaring
        it with type "custom" (VERIA finding on PR #32258)."""
        data = {
            "tools": [
                {"type": "custom", "name": "apply_patch", "description": "x"},
                {"type": "function", "name": "get_current_weather"},
            ]
        }
        assert extract_request_tool_names("/v1/responses", data) == [
            "apply_patch",
            "get_current_weather",
        ]

    def test_openai_responses_hoisted_tools(self):
        """Tool search results and additional input tools are merged into the
        provider request, so they must participate in auth allowlist checks."""
        data = {
            "tools": [
                {"type": "function", "name": "allowed", "description": "x"},
                {"type": "mcp", "server_label": "allowed_mcp", "server_url": "http://x"},
            ],
            "input": [
                {
                    "type": "tool_search_output",
                    "call_id": "call_tool_search",
                    "tools": [
                        {"type": "custom", "name": "forbidden_custom"},
                        {"type": "mcp", "server_label": "forbidden_mcp"},
                    ],
                },
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "function", "name": "forbidden_function"}],
                },
            ],
        }

        assert extract_request_tool_names("/v1/responses", data) == [
            "allowed",
            "allowed_mcp",
            "forbidden_custom",
            "forbidden_mcp",
            "forbidden_function",
        ]

    def test_openai_responses_ignores_tools_in_ordinary_input_items(self):
        data = {"input": [{"type": "message", "role": "user", "tools": [{"type": "function", "name": "x"}]}]}

        assert extract_request_tool_names("/v1/responses", data) == []

    def test_openai_responses_ignores_invalid_hoisted_tool_entries(self):
        data = {
            "input": [
                {
                    "type": "tool_search_output",
                    "tools": ["not-a-dict", {"type": "function", "name": "valid"}],
                }
            ]
        }

        assert extract_request_tool_names("/v1/responses", data) == ["valid"]

    def test_anthropic_tools(self):
        data = {"tools": [{"name": "get_weather"}, {"name": "run_sql"}]}
        assert extract_request_tool_names("/v1/messages", data) == [
            "get_weather",
            "run_sql",
        ]

    def test_anthropic_openai_format_tools_forwarded_by_bridge(self):
        data = {
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
                {"name": "run_sql"},
                {"googleSearch": {}},
            ]
        }
        assert extract_request_tool_names("/v1/messages", data) == [
            "get_weather",
            "run_sql",
        ]

    def test_anthropic_hybrid_tool_yields_every_name(self):
        data = {
            "tools": [
                {"type": "function", "name": "decoy", "function": {"name": "blocked_fn"}},
                {"type": "function", "name": "", "function": {"name": "hidden_fn"}},
            ]
        }
        assert extract_request_tool_names("/v1/messages", data) == [
            "decoy",
            "blocked_fn",
            "hidden_fn",
        ]

    def test_generate_content_tools(self):
        data = {
            "tools": [
                {
                    "functionDeclarations": [
                        {"name": "schedule_meeting", "description": "x"},
                    ]
                },
            ]
        }
        assert extract_request_tool_names("/generate_content", data) == ["schedule_meeting"]

    def test_mcp_call_tool_name(self):
        data = {"name": "my_tool", "arguments": {}}
        assert extract_request_tool_names("/mcp/call_tool", data) == ["my_tool"]

    def test_mcp_call_tool_mcp_tool_name(self):
        data = {"mcp_tool_name": "other_tool"}
        assert extract_request_tool_names("/mcp/call_tool", data) == ["other_tool"]

    def test_non_tool_route_returns_empty(self):
        data = {"tools": [{"type": "function", "function": {"name": "x"}}]}
        assert extract_request_tool_names("/v1/embeddings", data) == []


class TestCheckToolsAllowlist:
    """Test allowlist enforcement in auth (no DB in hot path)."""

    @pytest.mark.asyncio
    async def test_no_allowlist_passes(self):
        token = _token(metadata={}, team_metadata={})
        body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
        await check_tools_allowlist(
            request_body=body,
            valid_token=token,
            team_object=None,
            route="/v1/chat/completions",
        )

    @pytest.mark.asyncio
    async def test_allowed_tool_passes(self):
        token = _token(metadata={"allowed_tools": ["get_weather"]})
        body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
        await check_tools_allowlist(
            request_body=body,
            valid_token=token,
            team_object=None,
            route="/v1/chat/completions",
        )

    @pytest.mark.asyncio
    async def test_disallowed_tool_raises(self):
        token = _token(metadata={"allowed_tools": ["other_tool"]})
        body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
        with pytest.raises(ProxyException) as exc_info:
            await check_tools_allowlist(
                request_body=body,
                valid_token=token,
                team_object=None,
                route="/v1/chat/completions",
            )
        assert exc_info.value.type == ProxyErrorTypes.tool_access_denied
        assert "get_weather" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_disallowed_openai_format_tool_raises_on_messages_route(self):
        token = _token(metadata={"allowed_tools": ["other_tool"]})
        body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
        with pytest.raises(ProxyException) as exc_info:
            await check_tools_allowlist(
                request_body=body,
                valid_token=token,
                team_object=None,
                route="/v1/messages",
            )
        assert exc_info.value.type == ProxyErrorTypes.tool_access_denied
        assert "get_weather" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_hybrid_tool_with_decoy_name_raises_on_messages_route(self):
        token = _token(metadata={"allowed_tools": ["decoy"]})
        body = {"tools": [{"type": "function", "name": "decoy", "function": {"name": "run_sql"}}]}
        with pytest.raises(ProxyException) as exc_info:
            await check_tools_allowlist(
                request_body=body,
                valid_token=token,
                team_object=None,
                route="/v1/messages",
            )
        assert exc_info.value.type == ProxyErrorTypes.tool_access_denied
        assert "run_sql" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_disallowed_custom_tool_raises_on_responses_route(self):
        token = _token(metadata={"allowed_tools": ["other_tool"]})
        body = {"tools": [{"type": "custom", "name": "restricted_tool"}]}
        with pytest.raises(ProxyException) as exc_info:
            await check_tools_allowlist(
                request_body=body,
                valid_token=token,
                team_object=None,
                route="/v1/responses",
            )
        assert exc_info.value.type == ProxyErrorTypes.tool_access_denied
        assert "restricted_tool" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_disallowed_hoisted_tool_raises_on_responses_route(self):
        token = _token(metadata={"allowed_tools": ["allowed"]})
        body = {
            "tools": [{"type": "function", "name": "allowed"}],
            "input": [
                {
                    "type": "tool_search_output",
                    "call_id": "call_tool_search",
                    "tools": [{"type": "function", "name": "forbidden_function"}],
                }
            ],
        }

        with pytest.raises(ProxyException) as exc_info:
            await check_tools_allowlist(
                request_body=body,
                valid_token=token,
                team_object=None,
                route="/v1/responses",
            )
        assert exc_info.value.type == ProxyErrorTypes.tool_access_denied
        assert "forbidden_function" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_team_allowlist_used_when_key_empty(self):
        token = _token(
            metadata={},
            team_metadata={"allowed_tools": ["get_weather"]},
        )
        body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
        await check_tools_allowlist(
            request_body=body,
            valid_token=token,
            team_object=None,
            route="/v1/chat/completions",
        )

    @pytest.mark.asyncio
    async def test_key_allowlist_overrides_team(self):
        token = _token(
            metadata={"allowed_tools": ["get_weather"]},
            team_metadata={"allowed_tools": ["other_tool"]},
        )
        body = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
        await check_tools_allowlist(
            request_body=body,
            valid_token=token,
            team_object=None,
            route="/v1/chat/completions",
        )

    @pytest.mark.asyncio
    async def test_valid_token_none_skips(self):
        await check_tools_allowlist(
            request_body={"tools": [{"type": "function", "function": {"name": "x"}}]},
            valid_token=None,
            team_object=None,
            route="/v1/chat/completions",
        )

    @pytest.mark.asyncio
    async def test_no_tools_in_body_passes(self):
        token = _token(metadata={"allowed_tools": ["get_weather"]})
        await check_tools_allowlist(
            request_body={"messages": []},
            valid_token=token,
            team_object=None,
            route="/v1/chat/completions",
        )
