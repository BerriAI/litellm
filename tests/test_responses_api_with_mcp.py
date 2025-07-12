THIS SHOULD BE A LINTER ERRORimport pytest
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.responses.main import aresponses_api_with_mcp, responses_api_with_mcp
from litellm.types.responses.main import ResponsesAPIResponse


@pytest.mark.asyncio
async def test_aresponses_api_with_mcp_basic():
    """Test basic functionality of aresponses_api_with_mcp without actual MCP server"""
    
    # Mock the helper class methods
    with patch('litellm.responses.main.MCPResponsesAPIHelper._get_mcp_tools_from_manager') as mock_get_tools:
        mock_get_tools.return_value = []
        
        # Mock the regular aresponses function
        with patch('litellm.responses.main.aresponses') as mock_aresponses:
            mock_response = ResponsesAPIResponse(
                id="resp_test_123",
                object="response",
                status="completed",
                output=[{
                    "type": "message",
                    "role": "assistant",
                    "content": "Hello, this is a test response!"
                }],
                usage={
                    "completion_tokens": 10,
                    "prompt_tokens": 5,
                    "total_tokens": 15,
                },
                created_at=1234567890,
                model="test-model",
                parallel_tool_calls=False,
                temperature=1.0,
                tool_choice="auto",
                tools=[],
                top_p=1.0,
            )
            mock_aresponses.return_value = mock_response
            
            # Test with MCP tools that have server_url="litellm_proxy"
            tools = [
                {
                    "type": "mcp",
                    "server_label": "test_server",
                    "server_url": "litellm_proxy",
                    "require_approval": "never"
                }
            ]
            
            response = await aresponses_api_with_mcp(
                input="Test message",
                model="test-model",
                tools=tools
            )
            
            # Verify the response
            assert response.id == "resp_test_123"
            assert response.model == "test-model"
            
            # Verify that get_mcp_tools_from_manager was called
            mock_get_tools.assert_called_once()
            
            # Verify that aresponses was called
            mock_aresponses.assert_called_once()


@pytest.mark.asyncio 
async def test_aresponses_api_with_mcp_tool_execution():
    """Test automatic tool execution when require_approval=never"""
    
    # Mock MCP tool
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "A test tool"
    mock_mcp_tool.inputSchema = {"type": "object", "properties": {}}
    
    # Mock the helper class methods
    with patch('litellm.responses.main.MCPResponsesAPIHelper._get_mcp_tools_from_manager') as mock_get_tools:
        mock_get_tools.return_value = [mock_mcp_tool]
        
        # Mock tool call result
        mock_tool_result = MagicMock()
        mock_tool_result.content = [MagicMock()]
        mock_tool_result.content[0].text = "Tool executed successfully"
        
            with patch('litellm.responses.main.aresponses') as mock_aresponses:
                # First response with tool call
                first_response = ResponsesAPIResponse(
                    id="resp_test_123",
                    object="response", 
                    status="completed",
                    output=[{
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "test_tool",
                        "arguments": json.dumps({"param": "value"})
                    }],
                    usage={
                        "completion_tokens": 10,
                        "prompt_tokens": 5,
                        "total_tokens": 15,
                    },
                    created_at=1234567890,
                    model="test-model",
                    parallel_tool_calls=False,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                )
                
                # Second response after tool execution
                second_response = ResponsesAPIResponse(
                    id="resp_test_456",
                    object="response",
                    status="completed", 
                    output=[{
                        "type": "message",
                        "role": "assistant",
                        "content": "Based on the tool result, here's my response."
                    }],
                    usage={
                        "completion_tokens": 15,
                        "prompt_tokens": 10,
                        "total_tokens": 25,
                    },
                    created_at=1234567891,
                    model="test-model",
                    parallel_tool_calls=False,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                )
                
                # Mock aresponses to return first response, then second response
                mock_aresponses.side_effect = [first_response, second_response]
                
                # Test with MCP tools that have require_approval=never
                tools = [
                    {
                        "type": "mcp", 
                        "server_label": "test_server",
                        "server_url": "litellm_proxy",
                        "require_approval": "never"
                    }
                ]
                
                response = await aresponses_api_with_mcp(
                    input="Test message that requires tool use",
                    model="test-model",
                    tools=tools
                )
                
                # Verify that tools were executed
                mock_execute_tools.assert_called_once()
                
                # Verify that we got the second response (after tool execution)
                assert response.id == "resp_test_456"
                assert response.output[0]["content"] == "Based on the tool result, here's my response."
                
                # Verify aresponses was called twice
                assert mock_aresponses.call_count == 2


def test_responses_api_with_mcp_sync():
    """Test synchronous version of responses_api_with_mcp"""
    
    # Mock the global MCP server manager
    with patch('litellm.responses.main.global_mcp_server_manager') as mock_manager:
        mock_manager.list_tools = AsyncMock(return_value=[])
        
        # Mock the regular aresponses function
        with patch('litellm.responses.main.aresponses') as mock_aresponses:
            mock_response = ResponsesAPIResponse(
                id="resp_test_sync_123",
                object="response",
                status="completed",
                output=[{
                    "type": "message",
                    "role": "assistant", 
                    "content": "Hello, this is a sync test response!"
                }],
                usage={
                    "completion_tokens": 10,
                    "prompt_tokens": 5,
                    "total_tokens": 15,
                },
                created_at=1234567890,
                model="test-model",
                parallel_tool_calls=False,
                temperature=1.0,
                tool_choice="auto",
                tools=[],
                top_p=1.0,
            )
            mock_aresponses.return_value = mock_response
            
            # Test synchronous version
            response = responses_api_with_mcp(
                input="Test sync message",
                model="test-model",
                tools=[{
                    "type": "mcp",
                    "server_label": "test_server", 
                    "server_url": "litellm_proxy",
                    "require_approval": "never"
                }]
            )
            
            # Verify the response
            assert response.id == "resp_test_sync_123"
            assert response.model == "test-model"


@pytest.mark.asyncio
async def test_non_mcp_tools_passthrough():
    """Test that non-MCP tools are passed through normally"""
    
    with patch('litellm.responses.main.global_mcp_server_manager') as mock_manager:
        with patch('litellm.responses.main.aresponses') as mock_aresponses:
            mock_response = ResponsesAPIResponse(
                id="resp_test_passthrough_123",
                object="response",
                status="completed",
                output=[{
                    "type": "message",
                    "role": "assistant",
                    "content": "Response with regular tools"
                }],
                usage={
                    "completion_tokens": 10,
                    "prompt_tokens": 5,
                    "total_tokens": 15,
                },
                created_at=1234567890,
                model="test-model",
                parallel_tool_calls=False,
                temperature=1.0,
                tool_choice="auto",
                tools=[],
                top_p=1.0,
            )
            mock_aresponses.return_value = mock_response
            
            # Test with regular function tools (not MCP)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "regular_tool",
                        "description": "A regular function tool",
                        "parameters": {"type": "object"}
                    }
                }
            ]
            
            response = await aresponses_api_with_mcp(
                input="Test message with regular tools",
                model="test-model",
                tools=tools
            )
            
            # Verify MCP manager was not called since no MCP tools
            mock_manager.list_tools.assert_not_called()
            
            # Verify aresponses was called with the regular tools
            mock_aresponses.assert_called_once()
            call_args = mock_aresponses.call_args[1]
            assert call_args['tools'] == tools  # Regular tools should be passed through