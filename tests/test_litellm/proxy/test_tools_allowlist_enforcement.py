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
        assert extract_request_tool_names("/v1/responses", data) == [
            "get_current_weather"
        ]

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

    def test_openai_responses_builtin_tools(self):
        """Built-in server-side tools carry no name of their own, so they act under their
        type; without that a restricted key could still reach the internet and bill for
        web_search while its allowlist named nothing of the sort (VERIA finding on PR #37995)."""
        data = {
            "tools": [
                {"type": "web_search"},
                {"type": "code_interpreter", "container": {"type": "auto"}},
                {"type": "function", "name": "get_current_weather"},
                {"type": "mcp", "server_label": "dmcp", "server_url": "http://x"},
            ]
        }
        assert extract_request_tool_names("/v1/responses", data) == [
            "web_search",
            "code_interpreter",
            "get_current_weather",
            "dmcp",
        ]

    def test_openai_responses_tools_nested_in_additional_tools_input_item(self):
        """Codex's responses-lite wire mode declares tools inside an `additional_tools` input
        item, and providers hoist them into top-level `tools` before dispatch. Reading only
        `tools` would let a restricted key smuggle any tool through `input`
        (VERIA finding on PR #37995)."""
        data = {
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "web_search"}, {"type": "function", "name": "run_sql"}],
                },
            ],
            "tools": [{"type": "function", "name": "declared_up_front"}],
        }
        assert extract_request_tool_names("/v1/responses", data) == [
            "declared_up_front",
            "web_search",
            "run_sql",
        ]

    def test_openai_responses_malformed_additional_tools_yields_no_name(self):
        """An `additional_tools` item with a missing or non-list `tools` slot, and a plain string
        input, must not raise on the auth hot path."""
        data = {
            "input": [
                {"type": "additional_tools"},
                {"type": "additional_tools", "tools": "not-a-list"},
                "junk",
            ]
        }
        assert extract_request_tool_names("/v1/responses", data) == []
        assert extract_request_tool_names("/v1/responses", {"input": "plain string"}) == []

    def test_openai_responses_unnamed_tool_yields_no_name(self):
        """A function or custom tool missing its name must not fall back to the bare type:
        that would let "function" satisfy an allowlist that never granted the real tool."""
        data = {"tools": [{"type": "function"}, {"type": "custom", "name": ""}, {"type": "mcp"}, "junk"]}
        assert extract_request_tool_names("/v1/responses", data) == []

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
        assert extract_request_tool_names("/generate_content", data) == [
            "schedule_meeting"
        ]

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
    async def test_disallowed_builtin_web_search_raises_on_responses_route(self):
        token = _token(metadata={"allowed_tools": ["run_sql"]})
        body = {"tools": [{"type": "web_search"}]}
        with pytest.raises(ProxyException) as exc_info:
            await check_tools_allowlist(
                request_body=body,
                valid_token=token,
                team_object=None,
                route="/v1/responses",
            )
        assert exc_info.value.type == ProxyErrorTypes.tool_access_denied
        assert "web_search" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_allowlisted_builtin_web_search_passes_on_responses_route(self):
        token = _token(metadata={"allowed_tools": ["web_search", "run_sql"]})
        tools = [{"type": "web_search"}, {"type": "function", "name": "run_sql"}]
        body = {"tools": tools}
        assert extract_request_tool_names("/v1/responses", body) == ["web_search", "run_sql"]
        await check_tools_allowlist(
            request_body=body,
            valid_token=token,
            team_object=None,
            route="/v1/responses",
        )
        assert body["tools"] == tools

    @pytest.mark.asyncio
    async def test_disallowed_tool_nested_in_input_raises_on_responses_route(self):
        token = _token(metadata={"allowed_tools": ["run_sql"]})
        body = {
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [{"type": "web_search"}]},
            ]
        }
        with pytest.raises(ProxyException) as exc_info:
            await check_tools_allowlist(
                request_body=body,
                valid_token=token,
                team_object=None,
                route="/v1/responses",
            )
        assert exc_info.value.type == ProxyErrorTypes.tool_access_denied
        assert "web_search" in str(exc_info.value.message)

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
