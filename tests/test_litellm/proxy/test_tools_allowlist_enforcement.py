"""
Tests for tool allowlist enforcement by team/key (metadata.allowed_tools).

No implementation yet; these tests define expected behavior. When check_tools_allowlist
is implemented in common_checks, disallowed-tool tests should raise; allowed and
no-allowlist tests should pass.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import common_checks


class MockRequest:
    """Mock request with method attribute."""

    def __init__(self, method: str = "POST"):
        self.method = method


def get_mock_user_token(metadata=None, team_metadata=None) -> UserAPIKeyAuth:
    """Build UserAPIKeyAuth with optional metadata and team_metadata for allowlist."""
    kwargs = {
        "api_key": "test-key",
        "user_id": "test-user",
        "team_id": "test-team",
        "org_id": "test-org",
        "models": ["*"],
        "metadata": metadata or {},
    }
    if team_metadata is not None:
        kwargs["team_metadata"] = team_metadata
    return UserAPIKeyAuth(**kwargs)


def _tools_allowlist_patches():
    """Patches so only tool-allowlist behavior is under test; heavy/DB parts no-op."""
    p1 = patch(
        "litellm.proxy.auth.auth_checks._is_api_route_allowed",
        new_callable=AsyncMock,
        return_value=True,
    )
    p2 = patch(
        "litellm.proxy.auth.auth_checks.vector_store_access_check",
        new_callable=AsyncMock,
        return_value=None,
    )
    p3 = patch(
        "litellm.proxy.auth.auth_checks._run_project_checks",
        new_callable=AsyncMock,
        return_value=None,
    )
    return p1, p2, p3


class TestOpenAIChatCompletionsToolsAllowlist:
    """Tool allowlist enforcement for /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_chat_completions_allowed_tool_passes(self):
        """Request with tools in allowed_tools passes."""
        route = "/v1/chat/completions"
        request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["get_weather"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_chat_completions_disallowed_tool_raises(self):
        """Request with tool not in allowed_tools raises."""
        route = "/v1/chat/completions"
        request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["other_tool"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            with pytest.raises((Exception, ProxyException)) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route=route,
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=token,
                    request=request,
                )
        msg = str(exc_info.value).lower()
        assert "tool" in msg or "allowed" in msg

    @pytest.mark.asyncio
    async def test_chat_completions_legacy_functions_allowed(self):
        """Legacy 'functions' (no tools) with allowed name passes."""
        route = "/v1/chat/completions"
        request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "functions": [{"name": "get_weather"}],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["get_weather"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_chat_completions_no_allowlist_passes(self):
        """Request with tools but no metadata.allowed_tools / team_metadata passes."""
        route = "/v1/chat/completions"
        request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
        token = get_mock_user_token(metadata={}, team_metadata={})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
            )
        assert result is True


class TestOpenAIResponsesAPIToolsAllowlist:
    """Tool allowlist enforcement for /v1/responses."""

    @pytest.mark.asyncio
    async def test_responses_function_tool_allowed_passes(self):
        """Responses request with function tool in allowed_tools passes."""
        route = "/v1/responses"
        request_body = {
            "model": "gpt-4",
            "input": "What is the weather?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_current_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object"},
                }
            ],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["get_current_weather"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_responses_function_tool_disallowed_raises(self):
        """Responses request with function tool not in allowed_tools raises."""
        route = "/v1/responses"
        request_body = {
            "model": "gpt-4",
            "input": "What is the weather?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_current_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object"},
                }
            ],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["other"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            with pytest.raises((Exception, ProxyException)) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route=route,
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=token,
                    request=request,
                )
        msg = str(exc_info.value).lower()
        assert "tool" in msg or "allowed" in msg

    @pytest.mark.asyncio
    async def test_responses_mcp_server_allowed_passes(self):
        """Responses request with MCP server in allowed_tools passes."""
        route = "/v1/responses"
        request_body = {
            "model": "gpt-4",
            "input": "Hi",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_description": "Example MCP server",
                    "server_url": "https://example.com",
                    "require_approval": "never",
                }
            ],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["dmcp"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_responses_mcp_server_disallowed_raises(self):
        """Responses request with MCP server not in allowed_tools raises."""
        route = "/v1/responses"
        request_body = {
            "model": "gpt-4",
            "input": "Hi",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_description": "Example MCP server",
                    "server_url": "https://example.com",
                    "require_approval": "never",
                }
            ],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["other"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            with pytest.raises((Exception, ProxyException)):
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route=route,
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=token,
                    request=request,
                )


class TestAnthropicMessagesToolsAllowlist:
    """Tool allowlist enforcement for Anthropic /v1/messages."""

    @pytest.mark.asyncio
    async def test_anthropic_allowed_tool_passes(self):
        """Request with Anthropic-style tools in allowed_tools passes."""
        route = "/v1/messages"
        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "get_weather", "description": "Get weather"}],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["get_weather"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_anthropic_disallowed_tool_raises(self):
        """Request with Anthropic-style tool not in allowed_tools raises."""
        route = "/v1/messages"
        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "get_weather", "description": "Get weather"}],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["other_tool"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            with pytest.raises((Exception, ProxyException)) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route=route,
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=token,
                    request=request,
                )
        msg = str(exc_info.value).lower()
        assert "tool" in msg or "allowed" in msg


class TestGoogleGenerateContentToolsAllowlist:
    """Tool allowlist enforcement for Google generateContent."""

    @pytest.mark.asyncio
    async def test_google_allowed_tool_passes(self):
        """Request with tools[].functionDeclarations[].name in allowed_tools passes."""
        route = "/v1beta/models/gemini-3-flash-preview:generateContent"
        request_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Schedule a meeting"}]}
            ],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "schedule_meeting",
                            "description": "Schedules a meeting",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                }
            ],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["schedule_meeting"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route=route,
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=request,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_google_disallowed_tool_raises(self):
        """Request with tools[].functionDeclarations[].name not in allowed_tools raises."""
        route = "/v1beta/models/gemini-3-flash-preview:generateContent"
        request_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Schedule a meeting"}]}
            ],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "schedule_meeting",
                            "description": "Schedules a meeting",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                }
            ],
        }
        token = get_mock_user_token(metadata={"allowed_tools": ["other_tool"]})
        request = MockRequest("POST")

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            with pytest.raises((Exception, ProxyException)) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route=route,
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=token,
                    request=request,
                )
        msg = str(exc_info.value).lower()
        assert "tool" in msg or "allowed" in msg


# MCP REST tools/call body shape: server_id, name (tool name), arguments.
# See litellm/proxy/_experimental/mcp_server/rest_endpoints.py call_tool_rest_api.
# The exact field for tool name in the request body should match the implementation.
MCP_TOOL_CALL_BODY_ALLOWED = {
    "server_id": "srv",
    "name": "roll_dice",
    "arguments": {},
}


class TestMCPToolCallToolsAllowlist:
    """Test that MCP tool call routes (/mcp/tools/call, /mcp-rest/tools/call) enforce token allowed_tools via common_checks."""

    @pytest.mark.asyncio
    async def test_mcp_tool_call_allowed_passes(self):
        """Route /mcp-rest/tools/call with tool in token allowed_tools passes common_checks."""
        request = MockRequest("POST")
        request_body = dict(MCP_TOOL_CALL_BODY_ALLOWED)
        valid_token = get_mock_user_token(metadata={"allowed_tools": ["roll_dice"]})

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/mcp-rest/tools/call",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=valid_token,
                request=request,
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_mcp_tool_call_disallowed_raises(self):
        """Route /mcp-rest/tools/call with tool not in token allowed_tools raises."""
        request = MockRequest("POST")
        request_body = dict(MCP_TOOL_CALL_BODY_ALLOWED)
        valid_token = get_mock_user_token(metadata={"allowed_tools": ["other"]})

        p1, p2, p3 = _tools_allowlist_patches()
        with p1, p2, p3:
            with pytest.raises((Exception, ProxyException)) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route="/mcp-rest/tools/call",
                    llm_router=None,
                    proxy_logging_obj=MagicMock(),
                    valid_token=valid_token,
                    request=request,
                )
        exc_str = (
            getattr(exc_info.value, "message", None) or str(exc_info.value) or ""
        ).lower()
        assert "tool" in exc_str or "allowed" in exc_str
