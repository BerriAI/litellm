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

    def test_openai_responses_additional_tools_input_items(self):
        """Codex CLI nests its tool definitions in an ``additional_tools`` input item
        and leaves top-level ``tools`` empty. The Chat Completions bridge lifts those
        into the effective tool list, so extraction must see them; otherwise a
        restricted key walks past the allowlist by nesting a disallowed tool -- an
        MCP reference with require_approval "never" included (VERIA finding on
        PR #38388)."""
        data = {
            "tools": [{"type": "function", "name": "get_current_weather"}],
            "input": [
                {"role": "user", "content": "hi"},
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {"type": "custom", "name": "exec", "description": "x"},
                        {"type": "mcp", "server_label": "dmcp", "require_approval": "never"},
                    ],
                },
            ],
        }
        assert extract_request_tool_names("/v1/responses", data) == [
            "get_current_weather",
            "exec",
            "dmcp",
        ]

    def test_openai_responses_codex_namespaced_tools_are_extracted(self):
        """The real Codex 0.149 wire shape: nine tools grouped under two ``namespace``
        containers inside an ``additional_tools`` item, with top-level ``tools`` empty.
        A namespace entry carries only the group name, so without descending into it the
        allowlist extracts nothing enforceable at all."""
        data = {
            "tools": [],
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "tools": [
                                {"type": "custom", "name": "exec"},
                                {"type": "function", "name": "wait"},
                                {"type": "function", "name": "request_user_input"},
                            ],
                        },
                        {
                            "type": "namespace",
                            "name": "collaboration",
                            "tools": [
                                {"type": "function", "name": "followup_task"},
                                {"type": "function", "name": "interrupt_agent"},
                                {"type": "function", "name": "list_agents"},
                                {"type": "function", "name": "send_message"},
                                {"type": "function", "name": "spawn_agent"},
                                {"type": "function", "name": "wait_agent"},
                            ],
                        },
                    ],
                },
                {"role": "user", "content": "hi"},
            ],
        }
        assert extract_request_tool_names("/v1/responses", data) == [
            "exec",
            "wait",
            "request_user_input",
            "followup_task",
            "interrupt_agent",
            "list_agents",
            "send_message",
            "spawn_agent",
            "wait_agent",
        ]

    def test_openai_responses_top_level_namespace_tools_are_extracted(self):
        data = {
            "tools": [{"type": "namespace", "name": "functions", "tools": [{"type": "function", "name": "read_file"}]}]
        }
        assert extract_request_tool_names("/v1/responses", data) == ["read_file"]

    def test_openai_responses_string_input_is_ignored(self):
        data = {"tools": [{"type": "function", "name": "get_current_weather"}], "input": "hi"}
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

    @pytest.mark.asyncio
    async def test_disallowed_tool_nested_in_additional_tools_raises_on_responses_route(self):
        """The bridge lifts nested tools into the request, so nesting must not be a
        way around the allowlist (VERIA finding on PR #38388)."""
        token = _token(metadata={"allowed_tools": ["other_tool"]})
        body = {
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "custom", "name": "restricted_tool"}],
                },
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
        assert "restricted_tool" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_disallowed_tool_inside_a_namespace_raises_on_responses_route(self):
        """The shape Codex actually sends: disallowed tool inside a namespace, inside an
        additional_tools input item (VERIA finding on PR #38388)."""
        token = _token(metadata={"allowed_tools": ["other_tool"]})
        body = {
            "tools": [],
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "collaboration",
                            "tools": [{"type": "function", "name": "spawn_agent"}],
                        }
                    ],
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
        assert "spawn_agent" in str(exc_info.value.message)
