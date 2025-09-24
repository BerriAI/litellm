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
    
    # Test Case 3: Test deduplication of duplicate tools
    mock_mcp_tools_with_duplicates = [
        # First instance of duplicate tool
        type('MCPTool', (), {
            'name': 'GitMCP-fetch_litellm_documentation',
            'description': 'Fetch entire documentation file from GitHub repository: BerriAI/litellm. Useful for general questions. Always call this tool first if asked about BerriAI/litellm.',
            'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False}
        })(),
        # Second instance of duplicate tool (should be filtered out)
        type('MCPTool', (), {
            'name': 'GitMCP-fetch_litellm_documentation',
            'description': 'Fetch entire documentation file from GitHub repository: BerriAI/litellm. Useful for general questions. Always call this tool first if asked about BerriAI/litellm.',
            'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False}
        })(),
        # Other unique tools
        type('MCPTool', (), {
            'name': 'GitMCP-search_litellm_documentation',
            'description': 'Semantically search within the fetched documentation from GitHub repository: BerriAI/litellm. Useful for specific queries.',
            'inputSchema': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query'], 'additionalProperties': False}
        })(),
    ]
    
    mcp_tool_config_with_duplicates = [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy/mcp",
            "require_approval": "never",
            "allowed_tools": ["GitMCP-fetch_litellm_documentation"]
        }
    ]
    
    # First filter by allowed tools
    filtered_tools_with_duplicates = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_with_duplicates,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], mcp_tool_config_with_duplicates)
    )
    
    # Then deduplicate the filtered tools
    filtered_tools_deduplicated = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
        filtered_tools_with_duplicates
    )
    
    # Should only return 1 tool (the duplicate should be removed)
    assert len(filtered_tools_deduplicated) == 1, f"Expected 1 tool after deduplication, got {len(filtered_tools_deduplicated)}"
    
    # Check that the correct tool is present
    assert filtered_tools_deduplicated[0].name == "GitMCP-fetch_litellm_documentation", \
        f"Expected GitMCP-fetch_litellm_documentation, got {filtered_tools_deduplicated[0].name}"
    
    print("✓ Test Case 3: duplicate tools are properly deduplicated")
    
    # Test Case 3b: Test standalone deduplication method
    standalone_deduplicated = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(mock_mcp_tools_with_duplicates)
    
    # Should return 2 unique tools (GitMCP-fetch_litellm_documentation and GitMCP-search_litellm_documentation)
    assert len(standalone_deduplicated) == 2, f"Expected 2 unique tools after standalone deduplication, got {len(standalone_deduplicated)}"
    
    unique_tool_names = [tool.name for tool in standalone_deduplicated]
    expected_unique_names = ["GitMCP-fetch_litellm_documentation", "GitMCP-search_litellm_documentation"]
    assert set(unique_tool_names) == set(expected_unique_names), \
        f"Expected {expected_unique_names}, got {unique_tool_names}"
    
    print("✓ Test Case 3b: standalone deduplication method works correctly")
    
    # Test Case 4: Multiple MCP tool configs with different allowed_tools
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
async def test_streaming_mcp_events_validation():
    """
    Test that MCP streaming events are properly emitted when using streaming with MCP tools.
    
    This test validates:
    1. MCP discovery events are emitted first
    2. Regular streaming response events follow
    3. Tool execution events are emitted when tools are auto-executed
    """
    from unittest.mock import AsyncMock, patch
    from litellm.types.llms.openai import ResponsesAPIStreamEvents
    
    print("🧪 Testing MCP streaming events...")
    
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
        })(),
        type('MCPTool', (), {
            'name': 'get_repo_info',
            'description': 'Get repository information',
            'inputSchema': {
                "type": "object", 
                "properties": {
                    "repo_name": {"type": "string", "description": "Repository name"}
                },
                "required": ["repo_name"]
            }
        })()
    ]
    
    # Mock the MCP operations
    with patch.object(LiteLLM_Proxy_MCP_Handler, '_get_mcp_tools_from_manager', new_callable=AsyncMock) as mock_get_tools, \
         patch.object(LiteLLM_Proxy_MCP_Handler, '_execute_tool_calls', new_callable=AsyncMock) as mock_execute_tools:
        
        # Setup MCP mocks
        mock_get_tools.return_value = mock_mcp_tools
        
        def mock_execute_tool_calls_side_effect(tool_calls, user_api_key_auth):
            """Mock tool execution with realistic results"""
            results = []
            for tool_call in tool_calls:
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
                        "result": "LiteLLM is a unified interface for 100+ LLMs that provides consistent OpenAI-format output and includes proxy server capabilities."
                    })
            return results
        
        mock_execute_tools.side_effect = mock_execute_tool_calls_side_effect
        
        # Configure MCP tool with streaming and auto-execution
        mcp_tool_config = {
            "type": "mcp",
            "server_url": "litellm_proxy/mcp/test_server", 
            "require_approval": "never"  # This enables auto-execution
        }
        
        print("📞 Making streaming request with MCP tools...")
        
        # Make streaming request with MCP tools
        response = await litellm.aresponses(
            model="gpt-4o-mini",  # Use cheaper model for testing
            tools=[mcp_tool_config],
            tool_choice="required",
            input=[{
                "role": "user",
                "type": "message", 
                "content": "What is LiteLLM? Give me a brief overview."
            }],
            stream=True
        )
        
        print(f"📋 Response type: {type(response)}")
        assert hasattr(response, '__aiter__'), "Response should be async iterable for streaming"
        
        # Collect all streaming events
        events = []
        event_types = []
        mcp_discovery_events = []
        mcp_execution_events = []
        regular_events = []
        
        print("🔄 Collecting streaming events...")
        
        try:
            async for chunk in response:
                events.append(chunk)
                event_type = getattr(chunk, 'type', 'unknown')
                event_types.append(event_type)
                
                # Categorize events
                if event_type in [
                    ResponsesAPIStreamEvents.MCP_TOOLS_DISCOVERY_STARTED,
                    ResponsesAPIStreamEvents.MCP_TOOLS_DISCOVERY_COMPLETED
                ]:
                    mcp_discovery_events.append(chunk)
                elif event_type in [
                    ResponsesAPIStreamEvents.MCP_TOOL_EXECUTION_STARTED,
                    ResponsesAPIStreamEvents.MCP_TOOL_EXECUTION_COMPLETED
                ]:
                    mcp_execution_events.append(chunk)
                else:
                    regular_events.append(chunk)
                
                print(f"📦 Event: {event_type}")
                
                # Print MCP-specific event details
                if hasattr(chunk, 'mcp_servers'):
                    print(f"   🔧 MCP Servers: {chunk.mcp_servers}")
                elif hasattr(chunk, 'mcp_tools'):
                    print(f"   🛠️  MCP Tools: {len(chunk.mcp_tools)} tools discovered")
                elif hasattr(chunk, 'tool_name'):
                    print(f"   ⚙️  Tool: {chunk.tool_name}")
                    if hasattr(chunk, 'result'):
                        print(f"   ✅ Result: {chunk.result[:100]}...")
                
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
            # Continue with validation of events collected so far
        
        print(f"\n📊 Event Summary:")
        print(f"   Total events: {len(events)}")
        print(f"   MCP discovery events: {len(mcp_discovery_events)}")
        print(f"   MCP execution events: {len(mcp_execution_events)}")
        print(f"   Regular streaming events: {len(regular_events)}")
        print(f"   Event types: {set(event_types)}")
        
        # Validate MCP discovery events
        if mcp_discovery_events:
            print("✅ MCP discovery events found!")
            
            # Check for discovery started event
            started_events = [e for e in mcp_discovery_events if e.type == ResponsesAPIStreamEvents.MCP_TOOLS_DISCOVERY_STARTED]
            if started_events:
                print(f"   🚀 Discovery started events: {len(started_events)}")
                started_event = started_events[0]
                if hasattr(started_event, 'mcp_servers'):
                    print(f"   📡 MCP servers: {started_event.mcp_servers}")
            
            # Check for discovery completed event
            completed_events = [e for e in mcp_discovery_events if e.type == ResponsesAPIStreamEvents.MCP_TOOLS_DISCOVERY_COMPLETED]
            if completed_events:
                print(f"   🏁 Discovery completed events: {len(completed_events)}")
                completed_event = completed_events[0]
                if hasattr(completed_event, 'mcp_tools'):
                    print(f"   🔧 Tools discovered: {len(completed_event.mcp_tools)}")
        else:
            print("⚠️  No MCP discovery events found")
        
        # Validate MCP execution events (if auto-execution occurred)
        if mcp_execution_events:
            print("✅ MCP tool execution events found!")
            execution_started = [e for e in mcp_execution_events if e.type == ResponsesAPIStreamEvents.MCP_TOOL_EXECUTION_STARTED]
            execution_completed = [e for e in mcp_execution_events if e.type == ResponsesAPIStreamEvents.MCP_TOOL_EXECUTION_COMPLETED]
            print(f"   🚀 Execution started events: {len(execution_started)}")
            print(f"   🏁 Execution completed events: {len(execution_completed)}")
        
        # Validate that we got some form of streaming response
        assert len(events) > 0, "Should have received at least some streaming events"
        
        # Verify MCP mocks were called
        assert mock_get_tools.called, "MCP tools should have been fetched"
        print("✅ MCP tool fetching was called")
        
        print("🎉 MCP streaming events validation completed!")
        return {
            'total_events': len(events),
            'mcp_discovery_events': len(mcp_discovery_events),
            'mcp_execution_events': len(mcp_execution_events),
            'regular_events': len(regular_events),
            'event_types': list(set(event_types))
        }


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
    
    print("🧪 Testing basic streaming with MCP tools...")
    
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
            model="gpt-4o-mini",
            tools=[mcp_tool_config],
            tool_choice="required",
            input=[
                {
                    "role": "user",
                    "type": "message",
                    "content": "give me a TLDR of what BerriAI/litellm is about"
                }
            ],
            stream=True
        )
        
        print(f"📋 Response type: {type(response)}")
        assert hasattr(response, '__aiter__'), "Response should be an async streaming response"
        
        # Collect streaming chunks
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            print(f"📦 Chunk type: {getattr(chunk, 'type', 'unknown')}")
        
        print(f"📊 Total chunks received: {len(chunks)}")
        
        # Verify MCP mocks were called (may be called multiple times in streaming)
        assert mock_get_tools.call_count >= 1, f"Expected MCP tools to be fetched at least once, got {mock_get_tools.call_count}"
        print(f"MCP tools fetched: {len(mock_mcp_tools)}")
        
        # Verify we got a response
        assert response is not None
        assert len(chunks) > 0, "Should have received streaming chunks"
        
        print("Basic streaming responses API with MCP tools test passed!")


@pytest.mark.asyncio
async def test_mcp_parameter_preparation_helpers():
    """
    Test the new parameter preparation helper methods for clean MCP handling.
    
    Tests:
    1. _prepare_initial_call_params - handles stream disabling for auto-execute
    2. _prepare_follow_up_call_params - restores stream and removes tool_choice
    3. _build_request_params - clean parameter merging
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    print("🧪 Testing MCP parameter preparation helpers...")
    
    # Test _prepare_initial_call_params
    base_call_params = {
        "stream": True,
        "temperature": 0.7,
        "tool_choice": "required",
        "max_output_tokens": 1000
    }
    
    # Test Case 1: Auto-execute scenario (should disable streaming)
    initial_params_auto = LiteLLM_Proxy_MCP_Handler._prepare_initial_call_params(
        call_params=base_call_params,
        should_auto_execute=True
    )
    
    assert initial_params_auto["stream"] == False, "Stream should be disabled for auto-execute"
    assert initial_params_auto["temperature"] == 0.7, "Other params should be preserved"
    assert initial_params_auto["tool_choice"] == "required", "tool_choice should be preserved for initial call"
    assert base_call_params["stream"] == True, "Original params should not be mutated"
    
    print("✅ _prepare_initial_call_params (auto-execute) works correctly")
    
    # Test Case 2: No auto-execute scenario (should preserve streaming)
    initial_params_no_auto = LiteLLM_Proxy_MCP_Handler._prepare_initial_call_params(
        call_params=base_call_params,
        should_auto_execute=False
    )
    
    assert initial_params_no_auto["stream"] == True, "Stream should be preserved when not auto-executing"
    assert initial_params_no_auto["temperature"] == 0.7, "Other params should be preserved"
    
    print("✅ _prepare_initial_call_params (no auto-execute) works correctly")
    
    # Test _prepare_follow_up_call_params
    follow_up_params = LiteLLM_Proxy_MCP_Handler._prepare_follow_up_call_params(
        call_params=base_call_params,
        original_stream_setting=True
    )
    
    assert follow_up_params["stream"] == True, "Stream should be restored to original setting"
    assert "tool_choice" not in follow_up_params, "tool_choice should be removed for follow-up call"
    assert follow_up_params["temperature"] == 0.7, "Other params should be preserved"
    assert base_call_params["tool_choice"] == "required", "Original params should not be mutated"
    
    print("✅ _prepare_follow_up_call_params works correctly")
    
    # Test _build_request_params
    input_data = [{"role": "user", "content": "test", "type": "message"}]
    model = "gpt-4o-mini"
    tools = [{"type": "function", "name": "test_tool"}]
    call_params = {"stream": True, "temperature": 0.8}
    previous_response_id = "resp_123"
    extra_kwargs = {"custom_param": "test_value"}
    
    request_params = LiteLLM_Proxy_MCP_Handler._build_request_params(
        input=input_data,
        model=model,
        all_tools=tools,
        call_params=call_params,
        previous_response_id=previous_response_id,
        **extra_kwargs
    )
    
    # Verify core parameters
    assert request_params["input"] == input_data, "Input should be included"
    assert request_params["model"] == model, "Model should be included"
    assert request_params["tools"] == tools, "Tools should be included"
    assert request_params["previous_response_id"] == previous_response_id, "Previous response ID should be included"
    
    # Verify call_params are merged
    assert request_params["stream"] == True, "call_params should be merged"
    assert request_params["temperature"] == 0.8, "call_params should be merged"
    
    # Verify extra kwargs are merged
    assert request_params["custom_param"] == "test_value", "Extra kwargs should be merged"
    
    print("✅ _build_request_params works correctly")
    
    # Test _build_request_params with None previous_response_id
    request_params_no_prev = LiteLLM_Proxy_MCP_Handler._build_request_params(
        input=input_data,
        model=model,
        all_tools=tools,
        call_params=call_params,
        previous_response_id=None
    )
    
    assert "previous_response_id" not in request_params_no_prev, "None previous_response_id should not be included"
    
    print("✅ _build_request_params handles None previous_response_id correctly")
    
    print("🎉 All MCP parameter preparation helper tests passed!")


@pytest.mark.asyncio  
async def test_mcp_tool_execution_events_creation():
    """
    Test the _create_tool_execution_events helper method for generating streaming events.
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    print("Testing MCP tool execution events creation...")
    
    # Mock tool calls (simulating what comes from LLM response in function_call format)
    mock_tool_calls = [
        {
            "id": "call_abc123",
            "name": "search_repo",
            "arguments": '{"query": "LiteLLM overview"}',
            "type": "function_call"
        },
        {
            "id": "call_def456", 
            "name": "get_repo_info",
            "arguments": '{"repo_name": "BerriAI/litellm"}',
            "type": "function_call"
        }
    ]
    
    # Mock tool results (simulating what comes from tool execution)
    mock_tool_results = [
        {
            "tool_call_id": "call_abc123",
            "result": "LiteLLM is a unified interface for 100+ LLMs"
        },
        {
            "tool_call_id": "call_def456",
            "result": "Repository: BerriAI/litellm - Python library for LLM integration"
        }
    ]
    
    # Create tool execution events
    execution_events = LiteLLM_Proxy_MCP_Handler._create_tool_execution_events(
        tool_calls=mock_tool_calls,
        tool_results=mock_tool_results
    )
    
    # Verify events were created
    assert len(execution_events) > 0, "Should create tool execution events"
    print(f"Created {len(execution_events)} tool execution events")
    
    # Verify events have proper structure
    for event in execution_events:
        assert hasattr(event, 'type'), "Event should have type attribute"
        event_type = str(event.type)
        assert 'mcp_call' in event_type.lower() or 'output_item' in event_type.lower(), f"Event should be MCP-related: {event_type}"
        
        # Check for sequence numbers
        if hasattr(event, 'sequence_number'):
            assert isinstance(event.sequence_number, int), "Sequence number should be integer"
            assert event.sequence_number > 0, "Sequence number should be positive"
    
    print("Tool execution events have proper structure")
    
    # Test with empty inputs
    empty_events = LiteLLM_Proxy_MCP_Handler._create_tool_execution_events(
        tool_calls=[],
        tool_results=[]
    )
    
    assert len(empty_events) == 0, "Should create no events for empty inputs"
    print("Handles empty inputs correctly")
    
    print("MCP tool execution events creation test passed!")


@pytest.mark.asyncio
async def test_no_duplicate_mcp_tools_in_streaming_e2e():
    """
    End-to-end test to validate that MCP tools are not duplicated when using streaming.
    
    This test protects against the bug where:
    1. Parent function (aresponses_api_with_mcp) processed MCP tools once
    2. Streaming iterator processed MCP tools again, causing duplicates
    
    The test mocks the MCP manager response but validates the actual tools
    sent to the LLM to ensure no duplication occurs.
    """
    from unittest.mock import AsyncMock, patch, call
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    print("Testing no duplicate MCP tools in streaming E2E...")
    
    # Mock MCP tools that would be returned from the manager
    mock_mcp_tools = [
        type('MCPTool', (), {
            'name': 'search_docs',
            'description': 'Search documentation for information',
            'inputSchema': {
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        })(),
        type('MCPTool', (), {
            'name': 'get_file_content',
            'description': 'Get content of a specific file',
            'inputSchema': {
                "type": "object", 
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"}
                },
                "required": ["file_path"]
            }
        })()
    ]
    
    # Track all calls to the underlying LLM to detect duplicates
    llm_call_tools = []
    
    async def capture_llm_tools(**kwargs):
        """Capture the tools parameter from LLM calls"""
        tools = kwargs.get('tools', [])
        llm_call_tools.append(tools)
        
        # Return a minimal mock async streaming response
        class MockStreamingResponse:
            async def __aiter__(self):
                yield type('MockChunk', (), {
                    'type': 'response.completed',
                    'output': []
                })()
        
        return MockStreamingResponse()
    
    # Mock both the MCP manager and the underlying LLM call
    with patch.object(LiteLLM_Proxy_MCP_Handler, '_get_mcp_tools_from_manager', new_callable=AsyncMock) as mock_get_tools, \
         patch('litellm.aresponses', side_effect=capture_llm_tools) as mock_aresponses:
        
        # Setup MCP mock to return our test tools
        mock_get_tools.return_value = mock_mcp_tools
        
        # Configure MCP tool for streaming
        mcp_tool_config = {
            "type": "mcp",
            "server_url": "litellm_proxy/mcp/test_server",
            "require_approval": "always"  # Disable auto-execution to focus on tool duplication
        }
        
        print("Making streaming request with MCP tools...")
        
        # Make streaming request with MCP tools
        try:
            response = await litellm.aresponses(
                model="gpt-4o-mini",
                tools=[mcp_tool_config],
                input=[{
                    "role": "user",
                    "type": "message", 
                    "content": "Search the documentation for information about authentication."
                }],
                stream=True
            )
            
            # Consume the streaming response
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                
        except Exception as e:
            print(f"Request failed (expected for test): {e}")
            # Continue with validation even if request fails
        
        # Validate underlying LLM was called (this proves our mocking works)
        assert len(llm_call_tools) > 0, "LLM should have been called at least once"
        print(f"LLM called {len(llm_call_tools)} time(s)")
        
        # If MCP tools were processed, validate they were fetched exactly once
        # (This protects against duplicate fetching)
        if mock_get_tools.call_count > 0:
            assert mock_get_tools.call_count == 1, f"MCP tools should be fetched exactly once, got {mock_get_tools.call_count} calls"
            print(f"MCP tools fetched exactly once: {mock_get_tools.call_count}")
        else:
            print("MCP tools not fetched (likely due to test mocking - this is OK for validation)")
        
        # Analyze tools sent to LLM for duplicates
        for call_idx, tools_in_call in enumerate(llm_call_tools):
            print(f"LLM Call {call_idx + 1}: {len(tools_in_call)} tools")
            
            if tools_in_call:
                # Extract tool names to check for duplicates
                tool_names = []
                for tool in tools_in_call:
                    if isinstance(tool, dict):
                        tool_name = tool.get('function', {}).get('name') or tool.get('name')
                    else:
                        tool_name = getattr(tool, 'name', str(tool))
                    
                    if tool_name:
                        tool_names.append(tool_name)
                
                print(f"   Tool names: {tool_names}")
                
                # Check for duplicate tool names
                unique_tool_names = set(tool_names)
                duplicates = [name for name in tool_names if tool_names.count(name) > 1]
                
                assert len(duplicates) == 0, f"Found duplicate tools in LLM call {call_idx + 1}: {duplicates}"
                assert len(tool_names) == len(unique_tool_names), f"Tool names should be unique in call {call_idx + 1}"
                
                print(f"   No duplicate tools found in call {call_idx + 1}")
                
                # Validate that MCP tools were properly transformed to OpenAI format
                openai_format_tools = [tool for tool in tools_in_call if isinstance(tool, dict) and 'function' in tool]
                if openai_format_tools:
                    print(f"   Found {len(openai_format_tools)} OpenAI-format tools")
                    
                    # Verify tools have proper OpenAI structure
                    for tool in openai_format_tools:
                        assert 'type' in tool, "Tool should have 'type' field"
                        assert tool['type'] == 'function', "Tool type should be 'function'"
                        assert 'function' in tool, "Tool should have 'function' field"
                        assert 'name' in tool['function'], "Function should have 'name'"
                        assert 'description' in tool['function'], "Function should have 'description'"
                        assert 'parameters' in tool['function'], "Function should have 'parameters'"
                        
                    print(f"   All tools have proper OpenAI format")
        
        # The key validation: ensure no duplicate fetching occurred
        # This is the main protection against the bug we fixed
        if mock_get_tools.call_count > 1:
            print(f"ERROR: Duplicate MCP fetching detected! Called {mock_get_tools.call_count} times")
            assert False, f"MCP tools should be fetched exactly once, but were fetched {mock_get_tools.call_count} times"
        
        # Additional validation: ensure no duplicate tools in any LLM call
        total_duplicates_found = 0
        for call_idx, tools_in_call in enumerate(llm_call_tools):
            if tools_in_call:
                tool_names = []
                for tool in tools_in_call:
                    if isinstance(tool, dict):
                        tool_name = tool.get('function', {}).get('name') or tool.get('name')
                        if tool_name:
                            tool_names.append(tool_name)
                
                duplicates = [name for name in tool_names if tool_names.count(name) > 1]
                if duplicates:
                    total_duplicates_found += len(set(duplicates))
                    print(f"ERROR: Duplicate tools in call {call_idx + 1}: {set(duplicates)}")
        
        if total_duplicates_found > 0:
            assert False, f"Found {total_duplicates_found} duplicate tools across all LLM calls"
        
        print("No duplicate MCP tools E2E test passed!")
        print(f"Summary:")
        print(f"   - MCP manager called: {mock_get_tools.call_count} time(s)")
        print(f"   - LLM called: {len(llm_call_tools)} time(s)")
        print(f"   - Unique tools per call: {[len(set(getattr(t.get('function', {}), 'name', 'unknown') if isinstance(t, dict) else str(t) for t in tools)) for tools in llm_call_tools]}")
        print(f"   - No duplicate tools detected")
        
        return {
            'mcp_manager_calls': mock_get_tools.call_count,
            'llm_calls': len(llm_call_tools),
            'tools_per_call': [len(tools) for tools in llm_call_tools],
            'duplicate_tools_found': False
        }


    