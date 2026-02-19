"""
Tests for MCP Security Guardrail.

Validates that the guardrail blocks requests referencing unregistered MCP servers
and allows requests with only registered servers. Covers both /chat/completions
and /responses API paths (same pre_call_hook logic, different call_type).
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.mcp_security.mcp_security_guardrail import (
    MCPSecurityGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks


@pytest.fixture
def guardrail():
    return MCPSecurityGuardrail(
        guardrail_name="test-mcp-security",
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
        on_violation="block",
    )


class TestExtractMCPServerNames:
    def test_extracts_litellm_proxy_mcp_servers(self):
        tools = [
            {"type": "mcp", "server_url": "litellm_proxy/mcp/zapier"},
            {"type": "mcp", "server_url": "litellm_proxy/mcp/github"},
            {"type": "function", "function": {"name": "get_weather"}},
        ]
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools(tools)
        assert names == {"zapier", "github"}

    def test_ignores_non_mcp_tools(self):
        tools = [
            {"type": "function", "function": {"name": "get_weather"}},
        ]
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools(tools)
        assert names == set()

    def test_ignores_external_mcp_servers(self):
        tools = [
            {"type": "mcp", "server_url": "https://external-server.com/mcp"},
        ]
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools(tools)
        assert names == set()

    def test_empty_tools(self):
        names = MCPSecurityGuardrail._extract_mcp_server_names_from_tools([])
        assert names == set()


class TestMCPSecurityGuardrailPreCall:
    @pytest.mark.asyncio
    @patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager"
    )
    async def test_blocks_unregistered_server_chat_completions(
        self, mock_manager, guardrail
    ):
        """Simulates /chat/completions with an unregistered MCP server."""
        mock_manager.get_registry.return_value = {"zapier": MagicMock()}

        data = {
            "tools": [
                {"type": "mcp", "server_url": "litellm_proxy/mcp/zapier"},
                {"type": "mcp", "server_url": "litellm_proxy/mcp/evil_server"},
            ],
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": ["test-mcp-security"],
        }

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(),
                data=data,
                call_type="acompletion",
            )
        assert exc_info.value.status_code == 400
        assert "evil_server" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager"
    )
    async def test_blocks_unregistered_server_responses_api(
        self, mock_manager, guardrail
    ):
        """Simulates /responses with an unregistered MCP server."""
        mock_manager.get_registry.return_value = {"github": MagicMock()}

        data = {
            "tools": [
                {"type": "mcp", "server_url": "litellm_proxy/mcp/unknown_server"},
            ],
            "model": "gpt-4o",
            "input": "What can you do?",
            "guardrails": ["test-mcp-security"],
        }

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(),
                data=data,
                call_type="aresponses",
            )
        assert exc_info.value.status_code == 400
        assert "unknown_server" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager"
    )
    async def test_allows_registered_servers(self, mock_manager, guardrail):
        """All MCP servers are registered - request passes through."""
        mock_manager.get_registry.return_value = {
            "zapier": MagicMock(),
            "github": MagicMock(),
        }

        data = {
            "tools": [
                {"type": "mcp", "server_url": "litellm_proxy/mcp/zapier"},
                {"type": "mcp", "server_url": "litellm_proxy/mcp/github"},
            ],
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": ["test-mcp-security"],
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(),
            data=data,
            call_type="acompletion",
        )
        assert result == data

    @pytest.mark.asyncio
    async def test_passthrough_no_mcp_tools(self, guardrail):
        """Request with no MCP tools passes through without checking registry."""
        data = {
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
            ],
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": ["test-mcp-security"],
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(),
            data=data,
            call_type="acompletion",
        )
        assert result == data

    @pytest.mark.asyncio
    async def test_passthrough_no_tools(self, guardrail):
        """Request with no tools at all passes through."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": ["test-mcp-security"],
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=MagicMock(),
            data=data,
            call_type="acompletion",
        )
        assert result == data
