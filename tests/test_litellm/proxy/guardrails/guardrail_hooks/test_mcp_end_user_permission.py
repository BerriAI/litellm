"""
Tests for MCP End User Permission Guardrail Hook
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.mcp_end_user_permission import (
    MCPEndUserPermissionGuardrail,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)


class TestMCPEndUserPermissionGuardrail:
    """Test the MCP End User Permission Guardrail"""

    def test_extract_mcp_server_name(self):
        """Test extracting MCP server name from tool name"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Test valid MCP tool names
        assert guardrail._extract_mcp_server_name("github-create_issue") == "github"
        assert guardrail._extract_mcp_server_name("slack-send_message") == "slack"
        assert guardrail._extract_mcp_server_name("jira-create-ticket") == "jira"

        # Test invalid/non-MCP tool names
        assert guardrail._extract_mcp_server_name("search") is None
        assert guardrail._extract_mcp_server_name("") is None
        assert guardrail._extract_mcp_server_name(None) is None

    @pytest.mark.asyncio
    async def test_apply_guardrail_no_end_user(self):
        """Test guardrail when no end_user_id is present"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs with MCP tools
        inputs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "github-create_issue",
                        "description": "Create an issue",
                    },
                }
            ]
        }

        request_data = {}

        # Should pass through all tools when no end_user_id
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

        assert len(result.get("tools", [])) == 1
        assert result["tools"][0]["function"]["name"] == "github-create_issue"

    @pytest.mark.asyncio
    async def test_apply_guardrail_with_authorized_tools(self):
        """Test guardrail when end user has access to MCP servers"""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs with multiple tools
        inputs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "github-create_issue",
                        "description": "Create an issue",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "slack-send_message",
                        "description": "Send a message",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Regular search tool",
                    },
                },
            ]
        }

        request_data = {"user_api_key_end_user_id": "end-user-123"}

        # Mock fetching end user object with permissions
        with patch.object(
            MCPEndUserPermissionGuardrail,
            "_fetch_end_user_object",
            return_value=MagicMock(
                object_permission=LiteLLM_ObjectPermissionTable(
                    object_permission_id="perm-1",
                    mcp_servers=["github", "slack"],
                )
            ),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

            # Should keep all authorized MCP tools + non-MCP tools
            assert len(result.get("tools", [])) == 3

    @pytest.mark.asyncio
    async def test_apply_guardrail_with_unauthorized_tools(self):
        """Test guardrail filters out unauthorized MCP tools"""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs with tools where end user only has access to some
        inputs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "github-create_issue",
                        "description": "Create an issue",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "slack-send_message",
                        "description": "Send a message",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "jira-create_ticket",
                        "description": "Create a ticket",
                    },
                },
            ]
        }

        request_data = {"user_api_key_end_user_id": "end-user-123"}

        # Mock fetching end user object with limited permissions (only slack and jira)
        with patch.object(
            MCPEndUserPermissionGuardrail,
            "_fetch_end_user_object",
            return_value=MagicMock(
                object_permission=LiteLLM_ObjectPermissionTable(
                    object_permission_id="perm-1",
                    mcp_servers=["slack", "jira"],
                )
            ),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

            # Should filter out github tool
            assert len(result.get("tools", [])) == 2
            tool_names = [t["function"]["name"] for t in result["tools"]]
            assert "slack-send_message" in tool_names
            assert "jira-create_ticket" in tool_names
            assert "github-create_issue" not in tool_names

    @pytest.mark.asyncio
    async def test_apply_guardrail_no_permissions_configured(self):
        """Test guardrail when end user has no MCP permissions configured"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs with MCP tools
        inputs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "github-create_issue",
                        "description": "Create an issue",
                    },
                }
            ]
        }

        request_data = {"user_api_key_end_user_id": "end-user-123"}

        # Mock fetching end user object with no object_permission
        with patch.object(
            MCPEndUserPermissionGuardrail,
            "_fetch_end_user_object",
            return_value=MagicMock(object_permission=None),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

            # Should pass through all tools when no permissions configured
            assert len(result.get("tools", [])) == 1
            assert result["tools"][0]["function"]["name"] == "github-create_issue"

    @pytest.mark.asyncio
    async def test_apply_guardrail_with_non_mcp_tools(self):
        """Test guardrail passes through non-MCP tools"""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs with non-MCP tools
        inputs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search tool",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Calculate something",
                    },
                },
            ]
        }

        request_data = {"user_api_key_end_user_id": "end-user-123"}

        # Mock fetching end user object with MCP restrictions
        with patch.object(
            MCPEndUserPermissionGuardrail,
            "_fetch_end_user_object",
            return_value=MagicMock(
                object_permission=LiteLLM_ObjectPermissionTable(
                    object_permission_id="perm-1",
                    mcp_servers=["github"],
                )
            ),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

            # Should keep all non-MCP tools even with MCP restrictions
            assert len(result.get("tools", [])) == 2

    @pytest.mark.asyncio
    async def test_apply_guardrail_filters_unauthorized_mcp_tools(self):
        """Test guardrail filters out unauthorized MCP tools"""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs with MCP tools where user only has access to some
        inputs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "github-create_issue",
                        "description": "Create an issue",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "slack-send_message",
                        "description": "Send a message",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "jira-create_ticket",
                        "description": "Create a ticket",
                    },
                },
            ]
        }

        request_data = {"user_api_key_end_user_id": "end-user-123"}

        # Mock fetching end user object - only has access to slack and jira, not github
        with patch.object(
            MCPEndUserPermissionGuardrail,
            "_fetch_end_user_object",
            return_value=MagicMock(
                object_permission=LiteLLM_ObjectPermissionTable(
                    object_permission_id="perm-1",
                    mcp_servers=["slack", "jira"],
                )
            ),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

            # Should filter out github tool
            assert len(result.get("tools", [])) == 2
            tool_names = [t["function"]["name"] for t in result["tools"]]
            assert "slack-send_message" in tool_names
            assert "jira-create_ticket" in tool_names
            assert "github-create_issue" not in tool_names

    @pytest.mark.asyncio
    async def test_apply_guardrail_with_mixed_tools(self):
        """Test guardrail with both MCP and non-MCP tools"""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs with both MCP and non-MCP tools
        inputs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "github-create_issue",
                        "description": "Create an issue",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search tool",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "slack-send_message",
                        "description": "Send a message",
                    },
                },
            ]
        }

        request_data = {"user_api_key_end_user_id": "end-user-123"}

        # Mock fetching end user object - only has access to slack
        with patch.object(
            MCPEndUserPermissionGuardrail,
            "_fetch_end_user_object",
            return_value=MagicMock(
                object_permission=LiteLLM_ObjectPermissionTable(
                    object_permission_id="perm-1",
                    mcp_servers=["slack"],
                )
            ),
        ):
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )

            # Should keep slack MCP tool and non-MCP search tool, filter out github
            assert len(result.get("tools", [])) == 2
            tool_names = [t["function"]["name"] for t in result["tools"]]
            assert "search" in tool_names
            assert "slack-send_message" in tool_names
            assert "github-create_issue" not in tool_names

    @pytest.mark.asyncio
    async def test_apply_guardrail_no_tools_in_request(self):
        """Test guardrail when request has no tools"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create inputs without tools
        inputs = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

        request_data = {"user_api_key_end_user_id": "end-user-123"}

        # Should pass through unchanged
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

        assert result == inputs
        assert "tools" not in result
