import os
import sys
import pytest
import asyncio
from typing import List
from unittest.mock import Mock, patch, AsyncMock

sys.path.insert(0, os.path.abspath("../../.."))

# Import required modules
import litellm
from litellm.responses.main import aresponses_api_with_mcp
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager, global_mcp_server_manager
from litellm.types.llms.openai import ResponsesAPIResponse, OpenAIMcpServerTool


class MockUserAPIKeyAuth:
    """Mock UserAPIKeyAuth for testing"""
    def __init__(self):
        self.api_key = "test_key"
        self.user_id = "test_user"
        self.team_id = "test_team"
        self.user_email = "test@example.com"
        self.max_budget = 100.0
        self.spend = 0.0
        self.models = []
        self.aliases = {}
        self.config = {}
        self.permissions = {}
        self.metadata = {}
        self.object_permission_id = "test_permission_id"


@pytest.mark.asyncio
async def test_mcp_helper_methods():
    """Test the core MCP helper methods in LiteLLM_Proxy_MCP_Handler"""
    
    # Test _should_use_litellm_mcp_gateway
    mcp_tools = [
        {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never"
        }
    ]
    
    other_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info"
            }
        }
    ]
    
    # Should return True for MCP tools with litellm_proxy
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(mcp_tools) == True
    
    # Should return False for other tools
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(other_tools) == False
    
    # Should return False for None
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(None) == False
    
    # Test _parse_mcp_tools
    mixed_tools = mcp_tools + other_tools
    mcp_parsed, other_parsed = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(mixed_tools)
    
    assert len(mcp_parsed) == 1
    assert len(other_parsed) == 1
    assert mcp_parsed[0]["type"] == "mcp"
    assert other_parsed[0]["type"] == "function"
    
    # Test _should_auto_execute_tools
    mcp_tools_never = [{"require_approval": "never"}]
    mcp_tools_always = [{"require_approval": "always"}]
    
    assert LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(mcp_tools_never) == True
    assert LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(mcp_tools_always) == False
    
    print("✓ MCP helper methods test passed!")


@pytest.mark.asyncio
async def test_mcp_output_elements_addition():
    """Test adding MCP output elements to response"""
    
    # Create a mock response
    mock_response = ResponsesAPIResponse(
        **{  # type: ignore
            "id": "test_response_id",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "id": "msg_123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello, world!",
                            "annotations": []
                        }
                    ]
                }
            ],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15
            },
            "user": None,
            "metadata": {}
        }
    )
    
    # Mock MCP tools and tool results
    mock_mcp_tools = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }
    ]
    
    mock_tool_results = [
        {
            "tool_call_id": "call_123",
            "result": "Tool executed successfully"
        }
    ]
    
    # Test adding output elements
    updated_response = LiteLLM_Proxy_MCP_Handler._add_mcp_output_elements_to_response(
        response=mock_response,
        mcp_tools_fetched=mock_mcp_tools,
        tool_results=mock_tool_results
    )
    
    # Verify output elements were added
    assert len(updated_response.output) == 3  # Original + 2 new elements
    
    # Check that MCP tools output was added
    mcp_tools_output = updated_response.output[1]
    assert mcp_tools_output["type"] == "mcp_tools_fetched"
    assert mcp_tools_output["role"] == "system"
    assert mcp_tools_output["status"] == "completed"
    
    # Check that tool results output was added
    tool_results_output = updated_response.output[2]
    assert tool_results_output["type"] == "tool_execution_results"
    assert tool_results_output["role"] == "system"
    assert tool_results_output["status"] == "completed"
    
    print("✓ MCP output elements addition test passed!")


@pytest.mark.asyncio
async def test_aresponses_api_with_mcp_real_integration():
    """
    Test aresponses_api_with_mcp with real OpenAI GPT-4o API call and real MCP server.
    
    This test:
    1. Initializes the MCP server manager with gitmcp server
    2. Makes a real API call to OpenAI GPT-4o
    3. Uses MCP tools with server_url="litellm_proxy" and require_approval="never"
    4. Tests the actual integration without mocking
    """
    # Initialize the global MCP server manager with gitmcp server
    global_mcp_server_manager.load_servers_from_config({
        "gitmcp_server": {
            "url": "https://gitmcp.io/BerriAI/litellm",
            "transport": "http",  # Use HTTP transport for gitmcp
        }
    })
    
    # Initialize the tool name to server mapping
    await global_mcp_server_manager._initialize_tool_name_to_mcp_server_name_mapping()
    
    # Print available tools for debugging
    print("Available tools from MCP server:")
    for tool_name, server_name in global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.items():
        print(f"  {tool_name} -> {server_name}")
    
    # Create mock user API key auth
    mock_user_api_key_auth = MockUserAPIKeyAuth()
    
    # Define MCP tools with litellm_proxy server_url and require_approval="never"
    mcp_tools: List[OpenAIMcpServerTool] = [
        {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "server_label": "gitmcp_server"
        }
    ]
    
    # Define a query that should trigger tool usage
    test_input = "Can you get information about the LiteLLM repository? I want to understand what this project is about."
    
    try:
        # Call aresponses_api_with_mcp with real OpenAI API
        response = await aresponses_api_with_mcp(
            input=test_input,
            model="gpt-4o",  # Using GPT-4o as requested
            tools=mcp_tools,
            temperature=0.7,
            max_output_tokens=1000,
            user_api_key_auth=mock_user_api_key_auth,
            custom_llm_provider="openai"
        )
        
        # Verify response
        assert response is not None
        assert isinstance(response, ResponsesAPIResponse)
        assert response.model == "gpt-4o"
        assert response.status == "completed"
        assert len(response.output) > 0
        
        # Check if usage information is present
        assert response.usage is not None
        assert response.usage.total_tokens > 0
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        
        # Print response for debugging
        print(f"✓ Response received successfully!")
        print(f"Response ID: {response.id}")
        print(f"Response Status: {response.status}")
        print(f"Response Model: {response.model}")
        print(f"Response Usage: {response.usage}")
        print(f"Response Output: {response.output}")
        
        # Check if the response contains meaningful content
        output_text = ""
        for output_item in response.output:
            if isinstance(output_item, dict):
                if output_item.get("type") == "message":
                    content = output_item.get("content", [])
                    for content_item in content:
                        if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                            output_text += content_item.get("text", "")
        
        print(f"Response text: {output_text}")
        assert len(output_text) > 0, "Response should contain meaningful text"
        
        print("✓ MCP integration test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        # Print more details for debugging
        import traceback
        traceback.print_exc()
        raise

