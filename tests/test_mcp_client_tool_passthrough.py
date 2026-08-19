from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.mcp_handler import (
    anthropic_messages_with_mcp,
)


@pytest.mark.asyncio
async def test_mcp_auto_execute_bypasses_client_side_tools():
    """
    Ensure that if a response contains client-native tools (not present in tool_server_map),
    the auto-execution loop breaks early and passes the response back to the client.
    """
    mock_mcp_references = [{"type": "mcp", "server_url": "http://localhost/mcp", "require_approval": "never"}]
    client_tools = [{"name": "Read", "description": "Client Read tool"}]

    mock_mcp_tools = [SimpleNamespace(name="mcp_tool_1", description="MCP Tool", inputSchema={"type": "object"})]
    mock_tool_server_map = {"mcp_tool_1": "http://localhost/mcp"}

    mock_anthropic_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "call_mcp", "name": "mcp_tool_1", "input": {}},
            {
                "type": "tool_use",
                "id": "call_client",
                "name": "Read",
                "input": {"file_path": "test.txt"},
            },
        ],
        "stop_reason": "tool_use",
    }

    mock_context = MagicMock()
    mock_context.user_api_key_auth = None
    mock_context.litellm_trace_id = "trace_123"
    mock_context.mcp_auth_header = None
    mock_context.mcp_server_auth_headers = None
    mock_context.request_tags = None
    mock_context.oauth2_headers = None
    mock_context.raw_headers = None
    mock_context.litellm_call_id = "call_123"

    path_resolve = "litellm.responses.mcp.request_context.MCPRequestContext.resolve"
    path_parse = "litellm.responses.mcp.litellm_proxy_mcp_handler.LiteLLM_Proxy_MCP_Handler._parse_mcp_tools"
    path_process = (
        "litellm.responses.mcp.litellm_proxy_mcp_handler."
        "LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform"
    )
    path_auto = "litellm.responses.mcp.litellm_proxy_mcp_handler.LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools"
    path_exec = "litellm.responses.mcp.litellm_proxy_mcp_handler.LiteLLM_Proxy_MCP_Handler._execute_tool_calls"
    path_call = "litellm.llms.anthropic.experimental_pass_through.messages.mcp_handler._AnthropicMessagesCall"

    with (
        patch(path_resolve, return_value=mock_context),
        patch(path_parse, return_value=(mock_mcp_references, client_tools)),
        patch(path_process, new_callable=AsyncMock) as mock_process,
        patch(path_auto, return_value=True),
        patch(path_exec, new_callable=AsyncMock) as mock_execute,
        patch(path_call) as mock_call,
    ):
        mock_process.return_value = (mock_mcp_tools, mock_tool_server_map)
        mock_fn = AsyncMock(return_value=mock_anthropic_response)
        mock_call.return_value.fn = mock_fn

        response = await anthropic_messages_with_mcp(
            max_tokens=100,
            messages=[{"role": "user", "content": "Read test.txt and run image_understand"}],
            model="claude-3-5-sonnet-20241022",
            tools=[*mock_mcp_references, *client_tools],
        )

        mock_execute.assert_not_called()
        assert response == mock_anthropic_response
