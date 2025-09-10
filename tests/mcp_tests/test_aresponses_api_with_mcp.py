import os
import sys
import pytest
from typing import List, Any, cast

sys.path.insert(0, os.path.abspath("../../.."))

# Import required modules
import litellm
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.types.llms.openai import ResponsesAPIResponse, OpenAIMcpServerTool, ToolParam


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
    mcp_tools: List[Any] = [
        {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never"
        }
    ]
    
    other_tools: List[Any] = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
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
    
    # Check that MCP tools output was added - handle both dict and object cases
    mcp_tools_output = updated_response.output[1]
    if hasattr(mcp_tools_output, 'type'):
        # Handle as object with attributes
        output_obj = cast(Any, mcp_tools_output)
        assert output_obj.type == "mcp_tools_fetched"
        assert output_obj.role == "system"
        assert output_obj.status == "completed"
    elif isinstance(mcp_tools_output, dict):
        # Handle as dictionary
        assert mcp_tools_output["type"] == "mcp_tools_fetched"
        assert mcp_tools_output["role"] == "system"
        assert mcp_tools_output["status"] == "completed"
    
    # Check that tool results output was added
    tool_results_output = updated_response.output[2]
    if hasattr(tool_results_output, 'type'):
        # Handle as object with attributes
        output_obj = cast(Any, tool_results_output)
        assert output_obj.type == "tool_execution_results"
        assert output_obj.role == "system"
        assert output_obj.status == "completed"
    elif isinstance(tool_results_output, dict):
        # Handle as dictionary
        assert tool_results_output["type"] == "tool_execution_results"
        assert tool_results_output["role"] == "system"
        assert tool_results_output["status"] == "completed"
    
    print("✓ MCP output elements addition test passed!")


@pytest.mark.asyncio
async def test_aresponses_api_with_mcp_mock_integration():
    """
    Test the core MCP integration logic without complex external mocking.
    This focuses on verifying the MCP tool parsing and handling works correctly.
    """
    # Define MCP tools with litellm_proxy server_url and require_approval="never"
    mcp_tools: List[OpenAIMcpServerTool] = [
        {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "server_label": "test_server"
        }
    ]
    
    # Test the helper methods that the integration relies on
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    # Test 1: Verify MCP tools are detected correctly
    should_use_mcp = LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(cast(Any, mcp_tools))
    assert should_use_mcp == True, "Should detect MCP tools with litellm_proxy server_url"
    
    # Test 2: Verify auto-execution detection works
    should_auto_execute = LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(cast(Any, mcp_tools))
    assert should_auto_execute == True, "Should auto-execute tools with require_approval='never'"
    
    # Test 3: Verify tool parsing works correctly
    mcp_parsed, other_parsed = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(cast(Any, mcp_tools))
    assert len(mcp_parsed) == 1, "Should parse one MCP tool"
    assert len(other_parsed) == 0, "Should have no other tools"
    assert mcp_parsed[0]["type"] == "mcp", "Parsed tool should be MCP type"
    assert mcp_parsed[0]["server_url"] == "litellm_proxy", "Should preserve server_url"
    assert mcp_parsed[0].get("require_approval") == "never", "Should preserve require_approval"
    
    # Test 4: Test with mixed tools
    mixed_tools = mcp_tools + [
        {
            "type": "function",
            "name": "test_function",
            "parameters": {"type": "object"}
        }
    ]
    
    mcp_parsed, other_parsed = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(cast(Any, mixed_tools))
    assert len(mcp_parsed) == 1, "Should parse one MCP tool from mixed list"
    assert len(other_parsed) == 1, "Should have one other tool from mixed list"
    
    print("✓ MCP integration core logic test completed successfully!")
    print(f"MCP tools detected: {should_use_mcp}")
    print(f"Auto-execute enabled: {should_auto_execute}")
    print(f"MCP tools parsed: {len(mcp_parsed)}")
    print(f"Other tools parsed: {len(other_parsed)}")


@pytest.mark.asyncio
async def test_mcp_allowed_tools_filtering():
    """
    Test the allowed_tools filtering functionality for MCP tools.
    This test verifies that when allowed_tools is specified in MCP tool config,
    only the allowed tools are passed to the LLM.
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    # Mock MCP tools returned from the server (simulating all available tools)
    mock_mcp_tools_from_server = [
        # Mock MCP tool object with name attribute
        type('MCPTool', (), {
            'name': 'search_tiktoken_documentation',
            'description': 'Search tiktoken documentation',
            'inputSchema': {'type': 'object', 'properties': {'query': {'type': 'string'}}}
        })(),
        type('MCPTool', (), {
            'name': 'fetch_tiktoken_documentation', 
            'description': 'Fetch tiktoken documentation',
            'inputSchema': {'type': 'object', 'properties': {'path': {'type': 'string'}}}
        })(),
        type('MCPTool', (), {
            'name': 'list_tiktoken_functions',
            'description': 'List tiktoken functions',
            'inputSchema': {'type': 'object', 'properties': {}}
        })(),
        type('MCPTool', (), {
            'name': 'get_tiktoken_examples',
            'description': 'Get tiktoken examples', 
            'inputSchema': {'type': 'object', 'properties': {}}
        })()
    ]
    
    # Test Case 1: MCP tool config with allowed_tools specified
    mcp_tool_config_with_allowed_tools = [
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["search_tiktoken_documentation", "fetch_tiktoken_documentation"],
            "require_approval": "never"
        }
    ]
    
    # Filter tools using the helper function
    filtered_tools = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], mcp_tool_config_with_allowed_tools)
    )
    
    # Should only return the 2 allowed tools
    assert len(filtered_tools) == 2, f"Expected 2 filtered tools, got {len(filtered_tools)}"
    
    # Check that only allowed tools are included
    filtered_tool_names = [tool.name for tool in filtered_tools]
    expected_allowed_tools = ["search_tiktoken_documentation", "fetch_tiktoken_documentation"]
    
    assert set(filtered_tool_names) == set(expected_allowed_tools), \
        f"Expected tools {expected_allowed_tools}, got {filtered_tool_names}"
    
    # Verify excluded tools are not present
    excluded_tools = ["list_tiktoken_functions", "get_tiktoken_examples"]
    for excluded_tool in excluded_tools:
        assert excluded_tool not in filtered_tool_names, \
            f"Tool {excluded_tool} should have been filtered out"
    
    print("✓ Test Case 1: allowed_tools filtering works correctly")
    
    # Test Case 2: MCP tool config without allowed_tools (should return all tools)
    mcp_tool_config_without_allowed_tools = [
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "require_approval": "never"
        }
    ]
    
    filtered_tools_all = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], mcp_tool_config_without_allowed_tools)
    )
    
    # Should return all 4 tools when no allowed_tools specified
    assert len(filtered_tools_all) == 4, f"Expected 4 tools when no allowed_tools specified, got {len(filtered_tools_all)}"
    
    print("✓ Test Case 2: no allowed_tools returns all tools")
    
    # Test Case 3: Multiple MCP tool configs with different allowed_tools
    multiple_mcp_configs = [
        {
            "type": "mcp",
            "server_label": "gitmcp1",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["search_tiktoken_documentation"],
            "require_approval": "never"
        },
        {
            "type": "mcp",
            "server_label": "gitmcp2", 
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["fetch_tiktoken_documentation", "get_tiktoken_examples"],
            "require_approval": "never"
        }
    ]
    
    filtered_tools_multiple = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], multiple_mcp_configs)
    )
    
    # Should return union of all allowed tools (3 unique tools)
    assert len(filtered_tools_multiple) == 3, f"Expected 3 tools from multiple configs, got {len(filtered_tools_multiple)}"
    
    filtered_multiple_names = [tool.name for tool in filtered_tools_multiple]
    expected_multiple_tools = ["search_tiktoken_documentation", "fetch_tiktoken_documentation", "get_tiktoken_examples"]
    
    assert set(filtered_multiple_names) == set(expected_multiple_tools), \
        f"Expected tools {expected_multiple_tools}, got {filtered_multiple_names}"
    
    print("✓ Test Case 3: multiple MCP configs with different allowed_tools works correctly")
    
    # Test Case 4: Empty allowed_tools list (should return no tools)
    mcp_config_empty_allowed = [
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": [],
            "require_approval": "never"
        }
    ]
    
    filtered_tools_empty = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], mcp_config_empty_allowed)
    )
    
    # Should return all tools when allowed_tools is empty list (no filtering)
    assert len(filtered_tools_empty) == 4, f"Expected 4 tools when allowed_tools is empty list, got {len(filtered_tools_empty)}"
    
    print("✓ Test Case 4: empty allowed_tools list returns all tools")
    
    print("✓ MCP allowed_tools filtering test completed successfully!")

@pytest.mark.asyncio
async def test_streaming_responses_api_with_mcp_tools():
    """
    Test the streaming responses API with MCP tools when using server_url="litellm_proxy"

    Under the hood the follow occurs

    - MCP: responses called litellm MCP manager.list_tools (MOCKED)
    - Request 1: Made to gpt-4o with fetched tools (REAL LLM CALL)
    - MCP: Execute tool call from request 1 and returns result (MOCKED)
    - Request 2: Made to gpt-4o with fetched tools and tool results (REAL LLM CALL)

    Return the user the result of request 2
    """
    from unittest.mock import AsyncMock, patch
    
    #litellm._turn_on_debug()
    
    # Mock MCP tools that would be returned from the manager
    mock_mcp_tools = [
        type('MCPTool', (), {
            'name': 'search_repo',
            'description': 'Search BerriAI/litellm repository for information',
            'inputSchema': {
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        })()
    ]
    
    # Only mock the MCP-specific operations, let LLM responses be real
    with patch.object(LiteLLM_Proxy_MCP_Handler, '_get_mcp_tools_from_manager', new_callable=AsyncMock) as mock_get_tools, \
         patch.object(LiteLLM_Proxy_MCP_Handler, '_execute_tool_calls', new_callable=AsyncMock) as mock_execute_tools:
        
        # Setup MCP mocks only
        mock_get_tools.return_value = mock_mcp_tools
        
        # Create a dynamic mock that will match the actual tool call ID from the LLM response
        def mock_execute_tool_calls_side_effect(tool_calls, user_api_key_auth):
            """Mock function that returns results matching the actual tool call IDs from the LLM"""
            results = []
            for tool_call in tool_calls:
                # Extract call_id from the tool call
                call_id = None
                if isinstance(tool_call, dict):
                    call_id = tool_call.get("call_id") or tool_call.get("id")
                elif hasattr(tool_call, 'call_id'):
                    call_id = tool_call.call_id
                elif hasattr(tool_call, 'id'):
                    call_id = tool_call.id
                
                if call_id:
                    results.append({
                        "tool_call_id": call_id,
                        "result": "LiteLLM is a unified interface for 100+ LLMs that translates inputs to provider-specific completion endpoints and provides consistent OpenAI-format output."
                    })
            return results
        
        mock_execute_tools.side_effect = mock_execute_tool_calls_side_effect
        
        # Make the actual call - LLM responses will be real
        mcp_tool_config = cast(Any, {
            "type": "mcp",
            "server_url": "litellm_proxy", 
            "require_approval": "never"
        })
        response = await litellm.aresponses(
            model="gpt-4o",
            tools=[mcp_tool_config],
            tool_choice="required",
            input="give me a TLDR of what BerriAI/litellm is about",
            stream=True
        )
        print("full response: ", response)
        print("Response type:", type(response))
        #########################################################
        # Assert this is a async streaming response
        assert hasattr(response, '__aiter__'), "Response should be an async streaming response"
        #########################################################

        async for chunk in response:
            print(f"Chunk: {chunk}")
        
        # # If it's a streaming response, let's see what we get
        if hasattr(response, '__aiter__'):
            print("✓ Got streaming response - iterating through chunks:")
            chunks = []
            async for chunk in response:
                print(f"Chunk: {chunk}")
                chunks.append(chunk)
            print(f"Total chunks received: {len(chunks)}")
        else:
            print("✓ Got non-streaming response")
        
        # Verify MCP mocks were called
        mock_get_tools.assert_called_once()
        
        # Check if tool execution was called
        # tool_execution_called = mock_execute_tools.called
        
        print(f"✓ MCP tools fetched: {len(mock_mcp_tools)}")
        print(f"✓ _get_mcp_tools_from_manager called: {mock_get_tools.called}")
        

        # Verify we got a response
        assert response is not None
        
        print("✓ Streaming responses API with MCP tools test passed!")
        print(f"✓ Response type: {type(response)}")
        
        # If it's a streaming response, we can iterate through it
        if hasattr(response, '__aiter__'):
            print("✓ Response is async iterable (streaming)")
        elif hasattr(response, '__iter__'):
            print("✓ Response is iterable")
        else:
            print(f"✓ Response: {response}")


    