import os
import sys
import pytest
import asyncio
from typing import List

sys.path.insert(0, os.path.abspath("../../.."))

# Import required modules
import litellm
from litellm.responses.main import aresponses_api_with_mcp
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


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_aresponses_api_with_mcp_real_integration()) 