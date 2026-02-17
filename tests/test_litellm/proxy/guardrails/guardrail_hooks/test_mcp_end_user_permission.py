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


@pytest.mark.asyncio
class TestMCPEndUserPermissionGuardrail:
    """Test the MCP End User Permission Guardrail"""

    def test_extract_mcp_server_name_from_tool(self):
        """Test extracting MCP server name from tool name"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Test valid MCP tool names
        assert guardrail._extract_mcp_server_name_from_tool("github-create_issue") == "github"
        assert guardrail._extract_mcp_server_name_from_tool("slack-send_message") == "slack"
        assert guardrail._extract_mcp_server_name_from_tool("jira-create-ticket") == "jira"

        # Test invalid/non-MCP tool names
        assert guardrail._extract_mcp_server_name_from_tool("search") is None
        assert guardrail._extract_mcp_server_name_from_tool("") is None
        assert guardrail._extract_mcp_server_name_from_tool(None) is None

    @pytest.mark.asyncio
    async def test_check_end_user_has_mcp_permission_no_end_user(self):
        """Test permission check when no end_user_id is present"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth without end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
        )

        # Should allow access when no end_user_id
        has_permission = await guardrail._check_end_user_has_mcp_permission(
            server_name="github",
            user_api_key_auth=user_auth,
        )

        assert has_permission is True

    @pytest.mark.asyncio
    async def test_check_end_user_has_mcp_permission_with_access(self):
        """Test permission check when end user has access"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth with end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            end_user_id="end-user-123",
        )

        # Mock the _get_allowed_mcp_servers_for_end_user to return allowed servers
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.mcp_end_user_permission.MCPRequestHandler._get_allowed_mcp_servers_for_end_user"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = ["github", "slack"]

            # Should allow access when server is in allowed list
            has_permission = await guardrail._check_end_user_has_mcp_permission(
                server_name="github",
                user_api_key_auth=user_auth,
            )

            assert has_permission is True
            mock_get_allowed.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_end_user_has_mcp_permission_without_access(self):
        """Test permission check when end user does not have access"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth with end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            end_user_id="end-user-123",
        )

        # Mock the _get_allowed_mcp_servers_for_end_user to return different servers
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.mcp_end_user_permission.MCPRequestHandler._get_allowed_mcp_servers_for_end_user"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = ["slack", "jira"]

            # Should deny access when server is not in allowed list
            has_permission = await guardrail._check_end_user_has_mcp_permission(
                server_name="github",
                user_api_key_auth=user_auth,
            )

            assert has_permission is False

    @pytest.mark.asyncio
    async def test_check_end_user_has_mcp_permission_no_permissions(self):
        """Test permission check when end user has no MCP permissions"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth with end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            end_user_id="end-user-123",
        )

        # Mock the _get_allowed_mcp_servers_for_end_user to return empty list
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.mcp_end_user_permission.MCPRequestHandler._get_allowed_mcp_servers_for_end_user"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = []

            # Should deny access when end user has no permissions
            has_permission = await guardrail._check_end_user_has_mcp_permission(
                server_name="github",
                user_api_key_auth=user_auth,
            )

            assert has_permission is False

    @pytest.mark.asyncio
    async def test_post_call_hook_with_authorized_tools(self):
        """Test post call hook with authorized MCP tools"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth with end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            end_user_id="end-user-123",
        )

        # Create mock response with tool calls
        response = ModelResponse(
            id="test-response",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call-1",
                                type="function",
                                function=Function(
                                    name="github-create_issue",
                                    arguments='{"title": "test"}',
                                ),
                            ),
                        ],
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
        )

        # Mock the permission check to return True
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.mcp_end_user_permission.MCPRequestHandler._get_allowed_mcp_servers_for_end_user"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = ["github", "slack"]

            # Should not raise an exception
            result = await guardrail.async_post_call_success_hook(
                data={"model": "gpt-4"},
                user_api_key_dict=user_auth,
                response=response,
            )

            assert result == response

    @pytest.mark.asyncio
    async def test_post_call_hook_with_unauthorized_tools(self):
        """Test post call hook with unauthorized MCP tools"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth with end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            end_user_id="end-user-123",
        )

        # Create mock response with tool calls
        response = ModelResponse(
            id="test-response",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call-1",
                                type="function",
                                function=Function(
                                    name="github-create_issue",
                                    arguments='{"title": "test"}',
                                ),
                            ),
                        ],
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
        )

        # Mock the permission check to return servers that don't include github
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.mcp_end_user_permission.MCPRequestHandler._get_allowed_mcp_servers_for_end_user"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = ["slack", "jira"]

            # Should raise GuardrailRaisedException
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await guardrail.async_post_call_success_hook(
                    data={"model": "gpt-4"},
                    user_api_key_dict=user_auth,
                    response=response,
                )

            assert "does not have permission" in str(exc_info.value.message)
            assert "github" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_post_call_hook_with_non_mcp_tools(self):
        """Test post call hook with non-MCP tools (no prefix)"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth with end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            end_user_id="end-user-123",
        )

        # Create mock response with non-MCP tool calls
        response = ModelResponse(
            id="test-response",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call-1",
                                type="function",
                                function=Function(
                                    name="search",
                                    arguments='{"query": "test"}',
                                ),
                            ),
                        ],
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
        )

        # Should not raise an exception for non-MCP tools
        result = await guardrail.async_post_call_success_hook(
            data={"model": "gpt-4"},
            user_api_key_dict=user_auth,
            response=response,
        )

        assert result == response

    @pytest.mark.asyncio
    async def test_post_call_hook_no_end_user(self):
        """Test post call hook when no end_user_id is present"""
        guardrail = MCPEndUserPermissionGuardrail()

        # Create user auth without end_user_id
        user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
        )

        # Create mock response with MCP tool calls
        response = ModelResponse(
            id="test-response",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call-1",
                                type="function",
                                function=Function(
                                    name="github-create_issue",
                                    arguments='{"title": "test"}',
                                ),
                            ),
                        ],
                    ),
                )
            ],
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
        )

        # Should not raise an exception when no end_user_id
        result = await guardrail.async_post_call_success_hook(
            data={"model": "gpt-4"},
            user_api_key_dict=user_auth,
            response=response,
        )

        assert result == response
