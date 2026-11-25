import asyncio
import gc
import json
import logging
import sys
import warnings
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as MCPTool

# Import required modules
import litellm
from litellm.constants import MAX_GATEWAY_DIRECT_MCP_SERVERS
from litellm.exceptions import GuardrailRaisedException, MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.langfuse.langfuse import langfuse_trace_id, langfuse_trace_metadata
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.xai.common_utils import xai_reported_cost_in_usd
from litellm.proxy.mcp_server import mcp_server_manager
from litellm.proxy.mcp_server.faults.list_outcomes import AggregateToolListing, ServerListOk
from litellm.proxy.utils import _estimate_dispatched_failure_usage, _failure_fields_to_lift
from litellm.responses.dependencies import (
    ResponsesDependencies,
    responses_dependencies,
    using_responses_dependencies,
)
from litellm.responses.litellm_completion_transformation.session_handler import StoredTurns
from litellm.responses.mcp.direct_url_claims import DirectMCPServer
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.responses.mcp.mcp_streaming_iterator import MCPEnhancedStreamingIterator
from litellm.types.llms.openai import (
    OpenAIMcpServerTool,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
    ToolParam,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import ModelResponse, StandardLoggingPayload, Usage


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
        {"type": "mcp", "server_url": "litellm_proxy", "require_approval": "never"}
    ]

    other_tools: List[Any] = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
    ]

    # Should return True for MCP tools with litellm_proxy
    assert LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(mcp_tools) == True

    # Should return False for other tools
    assert (
        LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(other_tools) == False
    )

    # Should return False for None
    assert LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(None) == False

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

    assert LiteLLM_Proxy_MCP_Handler.should_auto_execute_tools(mcp_tools_never) == True
    assert (
        LiteLLM_Proxy_MCP_Handler.should_auto_execute_tools(mcp_tools_always) == False
    )

    # A single approval-required reference must disable auto-execution for the
    # whole request; otherwise a "never" reference alongside an "always" one
    # would let the approval-gated tool run without approval.
    mcp_tools_mixed = [{"require_approval": "never"}, {"require_approval": "always"}]
    assert LiteLLM_Proxy_MCP_Handler.should_auto_execute_tools(mcp_tools_mixed) == False
    mcp_tools_manual = [{"require_approval": "never"}, {"require_approval": "manual"}]
    assert LiteLLM_Proxy_MCP_Handler.should_auto_execute_tools(mcp_tools_manual) == False
    assert LiteLLM_Proxy_MCP_Handler.should_auto_execute_tools([]) == False

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
                            "annotations": [],
                        }
                    ],
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
                "total_tokens": 15,
            },
            "user": None,
            "metadata": {},
        }
    )

    # Mock MCP tools and tool results
    mock_mcp_tools = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    mock_tool_results = [
        {"tool_call_id": "call_123", "result": "Tool executed successfully"}
    ]

    # Test adding output elements
    updated_response = LiteLLM_Proxy_MCP_Handler.add_mcp_output_elements_to_response(
        response=mock_response,
        mcp_tools_fetched=mock_mcp_tools,
        tool_results=mock_tool_results,
    )

    # Verify output elements were added
    assert len(updated_response.output) == 3  # Original + 2 new elements

    # Check that MCP tools output was added - handle both dict and object cases
    mcp_tools_output = updated_response.output[1]
    if hasattr(mcp_tools_output, "type"):
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
    if hasattr(tool_results_output, "type"):
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
            "server_label": "test_server",
        }
    ]

    # Test the helper methods that the integration relies on
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    # Test 1: Verify MCP tools are detected correctly
    should_use_mcp = LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(
        cast(Any, mcp_tools)
    )
    assert (
        should_use_mcp == True
    ), "Should detect MCP tools with litellm_proxy server_url"

    # Test 2: Verify auto-execution detection works
    should_auto_execute = LiteLLM_Proxy_MCP_Handler.should_auto_execute_tools(
        cast(Any, mcp_tools)
    )
    assert (
        should_auto_execute == True
    ), "Should auto-execute tools with require_approval='never'"

    # Test 3: Verify tool parsing works correctly
    mcp_parsed, other_parsed = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(
        cast(Any, mcp_tools)
    )
    assert len(mcp_parsed) == 1, "Should parse one MCP tool"
    assert len(other_parsed) == 0, "Should have no other tools"
    assert mcp_parsed[0]["type"] == "mcp", "Parsed tool should be MCP type"
    assert mcp_parsed[0]["server_url"] == "litellm_proxy", "Should preserve server_url"
    assert (
        mcp_parsed[0].get("require_approval") == "never"
    ), "Should preserve require_approval"

    # Test 4: Test with mixed tools
    mixed_tools = mcp_tools + [
        {"type": "function", "name": "test_function", "parameters": {"type": "object"}}
    ]

    mcp_parsed, other_parsed = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(
        cast(Any, mixed_tools)
    )
    assert len(mcp_parsed) == 1, "Should parse one MCP tool from mixed list"
    assert len(other_parsed) == 1, "Should have one other tool from mixed list"

    print("✓ MCP integration core logic test completed successfully!")
    print(f"MCP tools detected: {should_use_mcp}")
    print(f"Auto-execute enabled: {should_auto_execute}")
    print(f"MCP tools parsed: {len(mcp_parsed)}")
    print(f"Other tools parsed: {len(other_parsed)}")


@pytest.mark.asyncio
async def test_aresponses_api_with_mcp_passes_mcp_server_auth_headers_to_process_tools():
    """
    Test that MCP auth headers from secret_fields (e.g. x-mcp-linear_config-authorization)
    are passed to _process_mcp_tools_without_openai_transform when using the responses API.
    """
    from litellm.responses.main import aresponses_api_with_mcp

    captured_process_kwargs = {}

    async def mock_process(**kwargs):
        captured_process_kwargs.update(kwargs)
        return ([], {})

    mock_response = ResponsesAPIResponse(
        **{
            "id": "resp_test",
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
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [],
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
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "user": None,
            "metadata": {},
        }
    )

    mcp_tools = [{"type": "mcp", "server_url": "litellm_proxy"}]
    secret_fields = {
        "raw_headers": {"x-mcp-linear_config-authorization": "Bearer linear-token"},
    }

    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            mock_process,
        ),
        patch(
            "litellm.responses.main.aresponses",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        await aresponses_api_with_mcp(
            input=[{"role": "user", "type": "message", "content": "hi"}],
            model="gpt-4o",
            tools=mcp_tools,
            secret_fields=secret_fields,
        )

    assert "mcp_server_auth_headers" in captured_process_kwargs
    mcp_server_auth_headers = captured_process_kwargs["mcp_server_auth_headers"]
    assert mcp_server_auth_headers is not None
    assert "linear_config" in mcp_server_auth_headers
    assert (
        mcp_server_auth_headers["linear_config"]["Authorization"]
        == "Bearer linear-token"
    )


@pytest.mark.asyncio
async def test_mcp_allowed_tools_filtering():
    """
    Test the allowed_tools filtering functionality for MCP tools.
    This test verifies that when allowed_tools is specified in MCP tool config,
    only the allowed tools are passed to the LLM.
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    # Mock MCP tools returned from the server (simulating all available tools)
    mock_mcp_tools_from_server = [
        # Mock MCP tool object with name attribute
        MCPTool.model_validate({
                "name": "search_tiktoken_documentation",
                "description": "Search tiktoken documentation",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }, by_name=False),
        MCPTool.model_validate({
                "name": "fetch_tiktoken_documentation",
                "description": "Fetch tiktoken documentation",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }, by_name=False),
        MCPTool.model_validate({
                "name": "list_tiktoken_functions",
                "description": "List tiktoken functions",
                "inputSchema": {"type": "object", "properties": {}},
            }, by_name=False),
        MCPTool.model_validate({
                "name": "get_tiktoken_examples",
                "description": "Get tiktoken examples",
                "inputSchema": {"type": "object", "properties": {}},
            }, by_name=False),
    ]

    allowed_mcp_servers = ["gitmcp"]

    # Test Case 1: MCP tool config with allowed_tools specified
    mcp_tool_config_with_allowed_tools = [
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": [
                "search_tiktoken_documentation",
                "fetch_tiktoken_documentation",
            ],
            "require_approval": "never",
        }
    ]

    # Filter tools using the helper function
    filtered_tools = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(
            List[ToolParam], mcp_tool_config_with_allowed_tools
        ),
    )

    # Should only return the 2 allowed tools
    assert (
        len(filtered_tools) == 2
    ), f"Expected 2 filtered tools, got {len(filtered_tools)}"

    # Check that only allowed tools are included
    filtered_tool_names = [tool.name for tool in filtered_tools]
    expected_allowed_tools = [
        "search_tiktoken_documentation",
        "fetch_tiktoken_documentation",
    ]

    assert set(filtered_tool_names) == set(
        expected_allowed_tools
    ), f"Expected tools {expected_allowed_tools}, got {filtered_tool_names}"

    # Verify excluded tools are not present
    excluded_tools = ["list_tiktoken_functions", "get_tiktoken_examples"]
    for excluded_tool in excluded_tools:
        assert (
            excluded_tool not in filtered_tool_names
        ), f"Tool {excluded_tool} should have been filtered out"

    print("✓ Test Case 1: allowed_tools filtering works correctly")

    # Test Case 2: MCP tool config without allowed_tools (should return all tools)
    mcp_tool_config_without_allowed_tools = [
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "require_approval": "never",
        }
    ]

    filtered_tools_all = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(
            List[ToolParam], mcp_tool_config_without_allowed_tools
        ),
    )

    # Should return all 4 tools when no allowed_tools specified
    assert (
        len(filtered_tools_all) == 4
    ), f"Expected 4 tools when no allowed_tools specified, got {len(filtered_tools_all)}"

    print("✓ Test Case 2: no allowed_tools returns all tools")

    # Test Case 3: Test deduplication of duplicate tools
    mock_mcp_tools_with_duplicates = [
        # First instance of duplicate tool
        MCPTool.model_validate({
                "name": "GitMCP-fetch_litellm_documentation",
                "description": "Fetch entire documentation file from GitHub repository: BerriAI/litellm. Useful for general questions. Always call this tool first if asked about BerriAI/litellm.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }, by_name=False),
        # Second instance of duplicate tool (should be filtered out)
        MCPTool.model_validate({
                "name": "GitMCP-fetch_litellm_documentation",
                "description": "Fetch entire documentation file from GitHub repository: BerriAI/litellm. Useful for general questions. Always call this tool first if asked about BerriAI/litellm.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }, by_name=False),
        # Other unique tools
        MCPTool.model_validate({
                "name": "GitMCP-search_litellm_documentation",
                "description": "Semantically search within the fetched documentation from GitHub repository: BerriAI/litellm. Useful for specific queries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }, by_name=False),
    ]

    mcp_tool_config_with_duplicates = [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy/mcp",
            "require_approval": "never",
            "allowed_tools": ["GitMCP-fetch_litellm_documentation"],
        }
    ]

    # First filter by allowed tools
    filtered_tools_with_duplicates = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_with_duplicates,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], mcp_tool_config_with_duplicates),
    )

    # Then deduplicate the filtered tools
    filtered_tools_deduplicated, _ = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
        filtered_tools_with_duplicates, []
    )

    # Should only return 1 tool (the duplicate should be removed)
    assert (
        len(filtered_tools_deduplicated) == 1
    ), f"Expected 1 tool after deduplication, got {len(filtered_tools_deduplicated)}"

    # Check that the correct tool is present
    assert (
        filtered_tools_deduplicated[0].name == "GitMCP-fetch_litellm_documentation"
    ), f"Expected GitMCP-fetch_litellm_documentation, got {filtered_tools_deduplicated[0].name}"

    print("✓ Test Case 3: duplicate tools are properly deduplicated")

    # Test Case 3b: Test standalone deduplication method
    standalone_deduplicated, _ = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
        mock_mcp_tools_with_duplicates, allowed_mcp_servers
    )

    # Should return 2 unique tools (GitMCP-fetch_litellm_documentation and GitMCP-search_litellm_documentation)
    assert (
        len(standalone_deduplicated) == 2
    ), f"Expected 2 unique tools after standalone deduplication, got {len(standalone_deduplicated)}"

    unique_tool_names = [tool.name for tool in standalone_deduplicated]
    expected_unique_names = [
        "GitMCP-fetch_litellm_documentation",
        "GitMCP-search_litellm_documentation",
    ]
    assert set(unique_tool_names) == set(
        expected_unique_names
    ), f"Expected {expected_unique_names}, got {unique_tool_names}"

    print("✓ Test Case 3b: standalone deduplication method works correctly")

    # Test Case 4: Multiple MCP tool configs with different allowed_tools
    multiple_mcp_configs = [
        {
            "type": "mcp",
            "server_label": "gitmcp1",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["search_tiktoken_documentation"],
            "require_approval": "never",
        },
        {
            "type": "mcp",
            "server_label": "gitmcp2",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": ["fetch_tiktoken_documentation", "get_tiktoken_examples"],
            "require_approval": "never",
        },
    ]

    filtered_tools_multiple = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], multiple_mcp_configs),
    )

    # Should return union of all allowed tools (3 unique tools)
    assert (
        len(filtered_tools_multiple) == 3
    ), f"Expected 3 tools from multiple configs, got {len(filtered_tools_multiple)}"

    filtered_multiple_names = [tool.name for tool in filtered_tools_multiple]
    expected_multiple_tools = [
        "search_tiktoken_documentation",
        "fetch_tiktoken_documentation",
        "get_tiktoken_examples",
    ]

    assert set(filtered_multiple_names) == set(
        expected_multiple_tools
    ), f"Expected tools {expected_multiple_tools}, got {filtered_multiple_names}"

    print(
        "✓ Test Case 3: multiple MCP configs with different allowed_tools works correctly"
    )

    # Test Case 4: Empty allowed_tools list (should return no tools)
    mcp_config_empty_allowed = [
        {
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/tiktoken",
            "allowed_tools": [],
            "require_approval": "never",
        }
    ]

    filtered_tools_empty = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
        mcp_tools=mock_mcp_tools_from_server,
        mcp_tools_with_litellm_proxy=cast(List[ToolParam], mcp_config_empty_allowed),
    )

    # Should return all tools when allowed_tools is empty list (no filtering)
    assert (
        len(filtered_tools_empty) == 4
    ), f"Expected 4 tools when allowed_tools is empty list, got {len(filtered_tools_empty)}"

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
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    # Mock MCP tools that would be returned from the manager
    mock_mcp_tools = [
        MCPTool.model_validate({
                "name": "search_repo",
                "description": "Search BerriAI/litellm repository for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }, by_name=False),
        MCPTool.model_validate({
                "name": "get_repo_info",
                "description": "Get repository information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_name": {
                            "type": "string",
                            "description": "Repository name",
                        }
                    },
                    "required": ["repo_name"],
                },
            }, by_name=False),
    ]

    # Build fake streaming chunks that the inner aresponses() call would yield
    fake_response_obj = MagicMock(spec=ResponsesAPIResponse)
    fake_response_obj.id = "resp_fake_123"
    fake_response_obj.output = []

    fake_created_chunk = MagicMock(spec=ResponsesAPIStreamingResponse)
    fake_created_chunk.type = ResponsesAPIStreamEvents.RESPONSE_CREATED
    fake_created_chunk.response = fake_response_obj

    fake_in_progress_chunk = MagicMock(spec=ResponsesAPIStreamingResponse)
    fake_in_progress_chunk.type = ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
    fake_in_progress_chunk.response = fake_response_obj

    fake_output_item_added_chunk = MagicMock(spec=ResponsesAPIStreamingResponse)
    fake_output_item_added_chunk.type = ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    fake_output_item_added_chunk.response = fake_response_obj

    fake_completed_chunk = MagicMock(spec=ResponsesAPIStreamingResponse)
    fake_completed_chunk.type = ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    fake_completed_chunk.response = fake_response_obj

    # Create a fake async iterator for the inner LLM streaming call
    class FakeAsyncIterator:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._chunks):
                raise StopAsyncIteration
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

    fake_stream = FakeAsyncIterator(
        [
            fake_created_chunk,
            fake_in_progress_chunk,
            fake_output_item_added_chunk,
            fake_completed_chunk,
        ]
    )

    # Mock the MCP operations and the inner aresponses call
    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "_get_mcp_tools_from_manager",
            new_callable=AsyncMock,
        ) as mock_get_tools,
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "execute_tool_calls",
            new_callable=AsyncMock,
        ) as mock_execute_tools,
        patch(
            "litellm.responses.main.aresponses",
            new_callable=AsyncMock,
            return_value=fake_stream,
        ),
    ):
        # Setup MCP mocks
        mock_get_tools.return_value = (mock_mcp_tools, ["test_server"])

        async def mock_execute_tool_calls_side_effect(
            tool_server_map, tool_calls, user_api_key_auth, **kwargs
        ):
            """Mock tool execution with realistic results"""
            results = []
            for tool_call in tool_calls:
                call_id = None
                if isinstance(tool_call, dict):
                    call_id = tool_call.get("call_id") or tool_call.get("id")
                elif hasattr(tool_call, "call_id"):
                    call_id = tool_call.call_id
                elif hasattr(tool_call, "id"):
                    call_id = tool_call.id

                if call_id:
                    results.append(
                        {
                            "tool_call_id": call_id,
                            "result": "LiteLLM is a unified interface for 100+ LLMs.",
                        }
                    )
            return results

        mock_execute_tools.side_effect = mock_execute_tool_calls_side_effect

        # Configure MCP tool with streaming and auto-execution
        mcp_tool_config = {
            "type": "mcp",
            "server_url": "litellm_proxy/mcp/test_server",
            "require_approval": "never",  # This enables auto-execution
        }

        # Make streaming request with MCP tools
        response = await litellm.aresponses(
            model="gpt-4o-mini",
            tools=[mcp_tool_config],
            tool_choice="required",
            input=[
                {
                    "role": "user",
                    "type": "message",
                    "content": "What is LiteLLM? Give me a brief overview.",
                }
            ],
            stream=True,
        )

        assert hasattr(
            response, "__aiter__"
        ), "Response should be async iterable for streaming"

        # Collect all streaming events
        events = []
        event_types = []
        mcp_discovery_events = []
        regular_events = []

        async for chunk in response:
            events.append(chunk)
            event_type = getattr(chunk, "type", "unknown")
            event_types.append(event_type)

            # Categorize events
            if event_type in [
                ResponsesAPIStreamEvents.MCP_LIST_TOOLS_IN_PROGRESS,
                ResponsesAPIStreamEvents.MCP_LIST_TOOLS_COMPLETED,
            ]:
                mcp_discovery_events.append(chunk)
            else:
                regular_events.append(chunk)

        # Validate that we got streaming events
        assert len(events) > 0, "Should have received at least some streaming events"

        # Validate MCP discovery events were emitted
        assert (
            len(mcp_discovery_events) > 0
        ), "Should have received MCP discovery events"

        # Check that discovery events come before regular content events
        first_discovery_idx = next(
            i
            for i, e in enumerate(events)
            if getattr(e, "type", None)
            in [
                ResponsesAPIStreamEvents.MCP_LIST_TOOLS_IN_PROGRESS,
                ResponsesAPIStreamEvents.MCP_LIST_TOOLS_COMPLETED,
            ]
        )
        # The output_item.added event triggers the transition to MCP discovery,
        # so discovery events should appear after it in the stream
        assert (
            first_discovery_idx > 0
        ), "MCP discovery events should follow the initial output_item.added event"

        # Verify MCP mocks were called
        assert mock_get_tools.called, "MCP tools should have been fetched"


@pytest.mark.asyncio
async def test_mcp_parameter_preparation_helpers():
    """
    Test the new parameter preparation helper methods for clean MCP handling.

    Tests:
    1. _prepare_initial_call_params - handles stream disabling for auto-execute
    2. _prepare_follow_up_call_params - restores stream and removes tool_choice
    3. _build_request_params - clean parameter merging
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    print("🧪 Testing MCP parameter preparation helpers...")

    # Test _prepare_initial_call_params
    base_call_params = {
        "stream": True,
        "temperature": 0.7,
        "tool_choice": "required",
        "max_output_tokens": 1000,
    }

    # Test Case 1: Auto-execute scenario (should disable streaming)
    initial_params_auto = LiteLLM_Proxy_MCP_Handler.prepare_initial_call_params(
        call_params=base_call_params, should_auto_execute=True
    )

    assert (
        initial_params_auto["stream"] == False
    ), "Stream should be disabled for auto-execute"
    assert initial_params_auto["temperature"] == 0.7, "Other params should be preserved"
    assert (
        initial_params_auto["tool_choice"] == "required"
    ), "tool_choice should be preserved for initial call"
    assert base_call_params["stream"] == True, "Original params should not be mutated"

    print("✅ _prepare_initial_call_params (auto-execute) works correctly")

    # Test Case 2: No auto-execute scenario (should preserve streaming)
    initial_params_no_auto = LiteLLM_Proxy_MCP_Handler.prepare_initial_call_params(
        call_params=base_call_params, should_auto_execute=False
    )

    assert (
        initial_params_no_auto["stream"] == True
    ), "Stream should be preserved when not auto-executing"
    assert (
        initial_params_no_auto["temperature"] == 0.7
    ), "Other params should be preserved"

    print("✅ _prepare_initial_call_params (no auto-execute) works correctly")

    # Test _prepare_follow_up_call_params
    follow_up_params = LiteLLM_Proxy_MCP_Handler.prepare_follow_up_call_params(
        call_params=base_call_params, original_stream_setting=True
    )

    assert (
        follow_up_params["stream"] == True
    ), "Stream should be restored to original setting"
    assert (
        "tool_choice" not in follow_up_params
    ), "a tool_choice that is not allowed_tools should be removed for follow-up call"
    assert follow_up_params["temperature"] == 0.7, "Other params should be preserved"
    assert (
        base_call_params["tool_choice"] == "required"
    ), "Original params should not be mutated"

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
        **extra_kwargs,
    )

    # Verify core parameters
    assert request_params["input"] == input_data, "Input should be included"
    assert request_params["model"] == model, "Model should be included"
    assert request_params["tools"] == tools, "Tools should be included"
    assert (
        request_params["previous_response_id"] == previous_response_id
    ), "Previous response ID should be included"

    # Verify call_params are merged
    assert request_params["stream"] == True, "call_params should be merged"
    assert request_params["temperature"] == 0.8, "call_params should be merged"

    # Verify extra kwargs are merged
    assert (
        request_params["custom_param"] == "test_value"
    ), "Extra kwargs should be merged"

    print("✅ _build_request_params works correctly")

    # Test _build_request_params with None previous_response_id
    request_params_no_prev = LiteLLM_Proxy_MCP_Handler._build_request_params(
        input=input_data,
        model=model,
        all_tools=tools,
        call_params=call_params,
        previous_response_id=None,
    )

    assert (
        "previous_response_id" not in request_params_no_prev
    ), "None previous_response_id should not be included"

    print("✅ _build_request_params handles None previous_response_id correctly")

    print("🎉 All MCP parameter preparation helper tests passed!")


@pytest.mark.asyncio
async def test_mcp_tool_execution_events_creation():
    """
    Test the _create_tool_execution_events helper method for generating streaming events.
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    print("Testing MCP tool execution events creation...")

    # Mock tool calls (simulating what comes from LLM response in function_call format)
    mock_tool_calls = [
        {
            "id": "call_abc123",
            "name": "search_repo",
            "arguments": '{"query": "LiteLLM overview"}',
            "type": "function_call",
        },
        {
            "id": "call_def456",
            "name": "get_repo_info",
            "arguments": '{"repo_name": "BerriAI/litellm"}',
            "type": "function_call",
        },
    ]

    # Mock tool results (simulating what comes from tool execution)
    mock_tool_results = [
        {
            "tool_call_id": "call_abc123",
            "result": "LiteLLM is a unified interface for 100+ LLMs",
        },
        {
            "tool_call_id": "call_def456",
            "result": "Repository: BerriAI/litellm - Python library for LLM integration",
        },
    ]

    # Create tool execution events
    execution_events = LiteLLM_Proxy_MCP_Handler.create_tool_execution_events(
        tool_calls=mock_tool_calls, tool_results=mock_tool_results
    )

    # Verify events were created
    assert len(execution_events) > 0, "Should create tool execution events"
    print(f"Created {len(execution_events)} tool execution events")

    # Verify events have proper structure
    for event in execution_events:
        assert hasattr(event, "type"), "Event should have type attribute"
        event_type = str(event.type)
        assert (
            "mcp_call" in event_type.lower() or "output_item" in event_type.lower()
        ), f"Event should be MCP-related: {event_type}"

        # Check for sequence numbers
        if hasattr(event, "sequence_number"):
            assert isinstance(
                event.sequence_number, int
            ), "Sequence number should be integer"
            assert event.sequence_number > 0, "Sequence number should be positive"

    print("Tool execution events have proper structure")

    # Test with empty inputs
    empty_events = LiteLLM_Proxy_MCP_Handler.create_tool_execution_events(
        tool_calls=[], tool_results=[]
    )

    assert len(empty_events) == 0, "Should create no events for empty inputs"
    print("Handles empty inputs correctly")

    print("MCP tool execution events creation test passed!")


@pytest.mark.asyncio
async def test_aresponses_mcp_call_forwards_listed_admission_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import sys
    import types
    from unittest.mock import AsyncMock

    from mcp.types import CallToolResult, TextContent
    from mcp.types import Tool as MCPTool

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.mcp_server import mcp_server_manager
    from litellm.types.mcp import MCPAuth, MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    responses_main: Final = importlib.import_module("litellm.responses.main")
    manager: Final = mcp_server_manager.global_mcp_server_manager
    server: Final = MCPServer(
        server_id="responses-auth-strip",
        name="echo_none",
        server_name="echo_none",
        url="http://mcp.example",
        transport=MCPTransport.http,
        auth_type=MCPAuth.none,
        extra_headers=["Authorization", "x-tenant"],
        allow_all_keys=True,
    )
    monkeypatch.setattr(manager, "registry", {server.server_id: server})

    mcp_client: Final = AsyncMock()
    mcp_client.discovery_auth_fingerprint.return_value = "test-credential-hash"
    mcp_client.list_tools_result.return_value = ListToolsResult(
        tools=[MCPTool(name="whoami", inputSchema={"type": "object", "properties": {}})]
    )
    mcp_client.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        isError=False,
    )
    monkeypatch.setattr(manager, "create_mcp_client", AsyncMock(return_value=mcp_client))
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        types.SimpleNamespace(
            general_settings={},
            master_key="test-master-key",
            proxy_logging_obj=None,
            prisma_client=None,
            user_api_key_cache=None,
        ),
    )

    raw_headers: Final = {"authorization": "Bearer sk-" + "1234", "x-tenant": "tenant-a"}
    user_api_key_auth: Final = UserAPIKeyAuth(api_key="hashed-key", token="hashed-key")
    tools: Final = [
        {
            "type": "mcp",
            "server_label": "echo_none",
            "server_url": "litellm_proxy/mcp/echo_none",
            "require_approval": "never",
        }
    ]

    monkeypatch.setattr(
        responses_main,
        "aresponses",
        AsyncMock(
            side_effect=[
                ResponsesAPIResponse(
                    id="resp_first",
                    created_at=0,
                    output=[
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call-1",
                            "name": "echo_none-whoami",
                            "arguments": "{}",
                            "status": "completed",
                        }
                    ],
                ),
                ResponsesAPIResponse(id="resp_final", created_at=0, output=[]),
            ]
        ),
    )
    monkeypatch.setattr(
        importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler"),
        "aresponses",
        AsyncMock(return_value=ResponsesAPIResponse(id="resp_final", created_at=0, output=[])),
    )

    await responses_main.aresponses_api_with_mcp(
        input="Call whoami",
        model="openai/gpt-5",
        tools=tools,
        litellm_metadata={"user_api_key_auth": user_api_key_auth},
        secret_fields={"raw_headers": raw_headers},
    )

    assert mcp_client.call_tool.await_args is not None
    assert manager.create_mcp_client.await_args is not None
    assert manager.create_mcp_client.await_args.kwargs["extra_headers"] == {
        "Authorization": "Bearer sk-" + "1234",
        "x-tenant": "tenant-a",
    }


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
    from unittest.mock import AsyncMock, patch

    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    print("Testing no duplicate MCP tools in streaming E2E...")

    # Mock MCP tools that would be returned from the manager
    mock_mcp_tools = [
        MCPTool.model_validate({
                "name": "search_docs",
                "description": "Search documentation for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }, by_name=False),
        MCPTool.model_validate({
                "name": "get_file_content",
                "description": "Get content of a specific file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to file"}
                    },
                    "required": ["file_path"],
                },
            }, by_name=False),
    ]

    # Track all calls to the underlying LLM to detect duplicates
    llm_call_tools = []

    async def capture_llm_tools(**kwargs):
        """Capture the tools parameter from LLM calls"""
        tools = kwargs.get("tools", [])
        llm_call_tools.append(tools)

        # Return a minimal mock async streaming response
        class MockStreamingResponse:
            async def __aiter__(self):
                yield type(
                    "MockChunk", (), {"type": "response.completed", "output": []}
                )()

        return MockStreamingResponse()

    # Mock both the MCP manager and the underlying LLM call
    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "_get_mcp_tools_from_manager",
            new_callable=AsyncMock,
        ) as mock_get_tools,
        patch("litellm.aresponses", side_effect=capture_llm_tools) as mock_aresponses,
    ):

        # Setup MCP mock to return our test tools
        mock_get_tools.return_value = mock_mcp_tools

        # Configure MCP tool for streaming
        mcp_tool_config = {
            "type": "mcp",
            "server_url": "litellm_proxy/mcp/test_server",
            "require_approval": "always",  # Disable auto-execution to focus on tool duplication
        }

        print("Making streaming request with MCP tools...")

        # Make streaming request with MCP tools
        try:
            response = await litellm.aresponses(
                model="gpt-4o-mini",
                tools=[mcp_tool_config],
                input=[
                    {
                        "role": "user",
                        "type": "message",
                        "content": "Search the documentation for information about authentication.",
                    }
                ],
                stream=True,
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
            assert (
                mock_get_tools.call_count == 1
            ), f"MCP tools should be fetched exactly once, got {mock_get_tools.call_count} calls"
            print(f"MCP tools fetched exactly once: {mock_get_tools.call_count}")
        else:
            print(
                "MCP tools not fetched (likely due to test mocking - this is OK for validation)"
            )

        # Analyze tools sent to LLM for duplicates
        for call_idx, tools_in_call in enumerate(llm_call_tools):
            print(f"LLM Call {call_idx + 1}: {len(tools_in_call)} tools")

            if tools_in_call:
                # Extract tool names to check for duplicates
                tool_names = []
                for tool in tools_in_call:
                    if isinstance(tool, dict):
                        tool_name = tool.get("function", {}).get("name") or tool.get(
                            "name"
                        )
                    else:
                        tool_name = getattr(tool, "name", str(tool))

                    if tool_name:
                        tool_names.append(tool_name)

                print(f"   Tool names: {tool_names}")

                # Check for duplicate tool names
                unique_tool_names = set(tool_names)
                duplicates = [name for name in tool_names if tool_names.count(name) > 1]

                assert (
                    len(duplicates) == 0
                ), f"Found duplicate tools in LLM call {call_idx + 1}: {duplicates}"
                assert len(tool_names) == len(
                    unique_tool_names
                ), f"Tool names should be unique in call {call_idx + 1}"

                print(f"   No duplicate tools found in call {call_idx + 1}")

                # Validate that MCP tools were properly transformed to OpenAI format
                openai_format_tools = [
                    tool
                    for tool in tools_in_call
                    if isinstance(tool, dict) and "function" in tool
                ]
                if openai_format_tools:
                    print(f"   Found {len(openai_format_tools)} OpenAI-format tools")

                    # Verify tools have proper OpenAI structure
                    for tool in openai_format_tools:
                        assert "type" in tool, "Tool should have 'type' field"
                        assert (
                            tool["type"] == "function"
                        ), "Tool type should be 'function'"
                        assert "function" in tool, "Tool should have 'function' field"
                        assert "name" in tool["function"], "Function should have 'name'"
                        assert (
                            "description" in tool["function"]
                        ), "Function should have 'description'"
                        assert (
                            "parameters" in tool["function"]
                        ), "Function should have 'parameters'"

                    print(f"   All tools have proper OpenAI format")

        # The key validation: ensure no duplicate fetching occurred
        # This is the main protection against the bug we fixed
        if mock_get_tools.call_count > 1:
            print(
                f"ERROR: Duplicate MCP fetching detected! Called {mock_get_tools.call_count} times"
            )
            pytest.fail(f"MCP tools should be fetched exactly once, but were fetched {mock_get_tools.call_count} times")

        # Additional validation: ensure no duplicate tools in any LLM call
        total_duplicates_found = 0
        for call_idx, tools_in_call in enumerate(llm_call_tools):
            if tools_in_call:
                tool_names = []
                for tool in tools_in_call:
                    if isinstance(tool, dict):
                        tool_name = tool.get("function", {}).get("name") or tool.get(
                            "name"
                        )
                        if tool_name:
                            tool_names.append(tool_name)

                duplicates = [name for name in tool_names if tool_names.count(name) > 1]
                if duplicates:
                    total_duplicates_found += len(set(duplicates))
                    print(
                        f"ERROR: Duplicate tools in call {call_idx + 1}: {set(duplicates)}"
                    )

        if total_duplicates_found > 0:
            pytest.fail(f"Found {total_duplicates_found} duplicate tools across all LLM calls")

        print("No duplicate MCP tools E2E test passed!")
        print(f"Summary:")
        print(f"   - MCP manager called: {mock_get_tools.call_count} time(s)")
        print(f"   - LLM called: {len(llm_call_tools)} time(s)")
        print(
            f"   - Unique tools per call: {[len(set(getattr(t.get('function', {}), 'name', 'unknown') if isinstance(t, dict) else str(t) for t in tools)) for tools in llm_call_tools]}"
        )
        print(f"   - No duplicate tools detected")

        return {
            "mcp_manager_calls": mock_get_tools.call_count,
            "llm_calls": len(llm_call_tools),
            "tools_per_call": [len(tools) for tools in llm_call_tools],
            "duplicate_tools_found": False,
        }


_ROUND_ONE_BODY: Final = MappingProxyType(
    {
        "id": "resp_round_one",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.6",
        "output": (
            MappingProxyType(
                {
                    "type": "function_call",
                    "id": "fc_round_one",
                    "call_id": "call_echo",
                    "name": "echo",
                    "arguments": '{"value": "PING"}',
                    "status": "completed",
                }
            ),
        ),
        "usage": MappingProxyType(
            {
                "input_tokens": 71,
                "input_tokens_details": MappingProxyType({"cached_tokens": 32}),
                "output_tokens": 19,
                "output_tokens_details": MappingProxyType({"reasoning_tokens": 0}),
                "total_tokens": 90,
            }
        ),
    }
)
_ROUND_TWO_BODY: Final = MappingProxyType(
    {
        "id": "resp_round_two",
        "object": "response",
        "created_at": 2,
        "status": "completed",
        "model": "gpt-5.6",
        "output": (
            MappingProxyType(
                {
                    "type": "message",
                    "id": "msg_round_two",
                    "status": "completed",
                    "role": "assistant",
                    "content": (MappingProxyType({"type": "output_text", "text": "PING", "annotations": ()}),),
                }
            ),
        ),
        "usage": MappingProxyType(
            {
                "input_tokens": 149,
                "input_tokens_details": MappingProxyType({"cached_tokens": 64}),
                "output_tokens": 5,
                "output_tokens_details": MappingProxyType({"reasoning_tokens": 0}),
                "total_tokens": 154,
            }
        ),
    }
)


WEATHER_TOOL: Final = MCPTool(
    name="get_weather",
    description="Get weather for a city",
    inputSchema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
WEATHER_ARGUMENTS: Final = {"city": "San Francisco"}
WEATHER_RESULT: Final = "Sunny, 72F"
WEATHER_ANSWER: Final = "It is sunny in San Francisco"
SSE_HEADERS: Final = {"content-type": "text/event-stream"}


def _sse(events: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def _openai_response(response_id: str, status: str, output: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1700000000,
        "status": status,
        "model": "gpt-4o-mini",
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    }


def _openai_round(response_id: str, item: Mapping[str, object], done_item: Mapping[str, object]) -> bytes:
    events: Final = (
        {"type": "response.created", "response": _openai_response(response_id, "in_progress", ())},
        {"type": "response.in_progress", "response": _openai_response(response_id, "in_progress", ())},
        {"type": "response.output_item.added", "output_index": 0, "item": dict(item)},
        {"type": "response.output_item.done", "output_index": 0, "item": dict(done_item)},
        {"type": "response.completed", "response": _openai_response(response_id, "completed", (done_item,))},
    )
    return _sse(tuple({**event, "sequence_number": number} for number, event in enumerate(events)))


def _openai_tool_round() -> bytes:
    call: Final = {"type": "function_call", "id": "fc_1", "call_id": "call_weather", "name": "get_weather"}
    return _openai_round(
        "resp_tool_round",
        {**call, "arguments": "", "status": "in_progress"},
        {**call, "arguments": json.dumps(WEATHER_ARGUMENTS), "status": "completed"},
    )


def _openai_answer_round() -> bytes:
    message: Final = {"type": "message", "id": "msg_1", "role": "assistant"}
    return _openai_round(
        "resp_answer_round",
        {**message, "status": "in_progress", "content": []},
        {**message, "status": "completed", "content": [{"type": "output_text", "text": WEATHER_ANSWER, "annotations": []}]},
    )


def _anthropic_message_start(
    message_id: str, model: str, input_tokens: int, cache_usage: Mapping[str, object] = MappingProxyType({})
) -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 1, **cache_usage},
        },
    }


def _anthropic_round(
    message_id: str,
    block: Mapping[str, object],
    delta: Mapping[str, object],
    stop: str,
    model: str = "claude-haiku-4-5",
    usage: tuple[int, int] = (5, 5),
    cache_usage: Mapping[str, object] = MappingProxyType({}),
) -> bytes:
    return _sse(
        (
            _anthropic_message_start(message_id, model, usage[0], cache_usage),
            {"type": "content_block_start", "index": 0, "content_block": dict(block)},
            {"type": "content_block_delta", "index": 0, "delta": dict(delta)},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": usage[1]}},
            {"type": "message_stop"},
        )
    )


def _anthropic_tool_round() -> bytes:
    return _anthropic_round(
        "msg_tool_round",
        {"type": "tool_use", "id": "call_weather", "name": "get_weather", "input": {}},
        {"type": "input_json_delta", "partial_json": json.dumps(WEATHER_ARGUMENTS)},
        "tool_use",
    )


def _anthropic_answer_round() -> bytes:
    return _anthropic_round(
        "msg_answer_round",
        {"type": "text", "text": ""},
        {"type": "text_delta", "text": WEATHER_ANSWER},
        "end_turn",
    )


SCRIPTED_PROVIDERS: Final = MappingProxyType(
    {
        "openai": ("openai/gpt-4o-mini", "https://api.openai.com/v1/responses", _openai_tool_round, _openai_answer_round),
        "anthropic": (
            "anthropic/claude-haiku-4-5",
            "https://api.anthropic.com/v1/messages",
            _anthropic_tool_round,
            _anthropic_answer_round,
        ),
    }
)


def _round_cost(body: Mapping[str, object]) -> float:
    return litellm.completion_cost(
        completion_response=ResponsesAPIResponse.model_validate(body),
        model="gpt-5.6",
        custom_llm_provider="openai",
        call_type="aresponses",
    )


class _RoundRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: tuple[StandardLoggingPayload, ...] = ()
        self.results: tuple[object, ...] = ()
        self.failed: tuple[StandardLoggingPayload, ...] = ()
        self.hooked: tuple[object, ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged = (*self.logged, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))
        self.results = (*self.results, response_obj)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failed = (*self.failed, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))

    def logging_hook(self, kwargs: dict, result: Any, call_type: str) -> tuple[dict, Any]:
        self.hooked = (*self.hooked, result)
        return kwargs, result


def _no_op_sync_callback(
    kwargs: Mapping[str, object], completion_response: object, start_time: datetime, end_time: datetime
) -> None:
    return None


async def _settle(condition: Callable[[], bool]) -> None:
    for _ in range(200):
        if condition():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)


_ECHO_MCP_TOOL: Final = MappingProxyType(
    {"type": "mcp", "server_url": "litellm_proxy/mcp/demo", "require_approval": "never"}
)


def _serve_responses(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[httpx.Response],
    url_regex: str = r"https://mcp-rounds\.example\.com/.*responses",
) -> respx.Route:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.post(url__regex=url_regex).mock(side_effect=list(responses))


def _serve_rounds(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, bodies: Sequence[Mapping[str, object]]
) -> respx.Route:
    return _serve_responses(
        respx_mock, monkeypatch, tuple(httpx.Response(200, content=json.dumps(body, default=dict)) for body in bodies)
    )


@contextmanager
def _echo_tool_served(tool_result: str) -> Iterator[None]:
    echo_tool: Final = MCPTool.model_validate(
        {"name": "echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
        by_name=False,
    )
    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            AsyncMock(return_value=([echo_tool], {"echo": "demo"})),
        ),
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "execute_tool_calls",
            AsyncMock(return_value=[{"tool_call_id": "call_echo", "result": tool_result, "name": "echo"}]),
        ),
    ):
        yield


async def _auto_execute(
    logging_obj: Logging, mcp_tool: Mapping[str, object], tool_result: str
) -> ResponsesAPIResponse:
    with _echo_tool_served(tool_result):
        response: Final = await litellm.aresponses(
            model="openai/gpt-5.6",
            input="Call echo with PING",
            tools=[dict(mcp_tool)],
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com/v1",
            litellm_logging_obj=logging_obj,
        )
    assert isinstance(response, ResponsesAPIResponse)
    return response


async def _auto_execute_echo(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, logging_obj: Logging
) -> ResponsesAPIResponse:
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY))
    return await _auto_execute(logging_obj, _ECHO_MCP_TOOL, "PING")


_DEMO_MCP_SERVER: Final = MCPServer(
    server_id="demo", name="demo", server_name="demo", transport="http", url="https://demo.example/mcp", auth_type="none"
)


def _demo_server_named(name: str) -> MCPServer | None:
    return _DEMO_MCP_SERVER if name == "demo" else None


def _served_by_a_recording_mcp_server(monkeypatch: pytest.MonkeyPatch, call_tool: AsyncMock) -> None:
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", SimpleNamespace(proxy_logging_obj=None, prisma_client=None)
    )
    monkeypatch.setattr(
        "litellm.proxy.mcp_server.mcp_server_manager.global_mcp_server_manager",
        SimpleNamespace(
            catalog=SimpleNamespace(operation=nullcontext),
            get_registry=MagicMock(return_value={}),
            call_tool=call_tool,
            get_mcp_server_from_tool_name=MagicMock(return_value=None),
            get_mcp_server_by_name=MagicMock(side_effect=_demo_server_named),
            get_allowed_mcp_servers=AsyncMock(return_value=[_DEMO_MCP_SERVER.server_id]),
            get_mcp_servers_from_ids=MagicMock(return_value=[_DEMO_MCP_SERVER]),
        ),
    )


def _echo_tool_listed() -> AbstractContextManager[object]:
    return patch(
        "litellm.proxy.mcp_server.server._get_tools_from_mcp_servers",
        AsyncMock(
            return_value=AggregateToolListing(
                tools=[
                    MCPTool.model_validate(
                        {"name": "echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
                        by_name=False,
                    )
                ],
                outcomes={},
            )
        ),
    )


async def _auto_execute_through_mcp_server(
    monkeypatch: pytest.MonkeyPatch, call_tool: AsyncMock, logging_obj: Logging
) -> None:
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)
    with _echo_tool_listed():
        await litellm.aresponses(
            model="openai/gpt-5.6",
            input="Call echo with PING",
            tools=[dict(_ECHO_MCP_TOOL)],
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com/v1",
            litellm_call_id=logging_obj.litellm_call_id,
            litellm_logging_obj=logging_obj,
        )


_ALLOWED_ONLY_ANOTHER_TOOL: Final = {
    "type": "allowed_tools",
    "mode": "auto",
    "tools": [{"type": "function", "name": "another_tool"}],
}


async def _call_with_a_choice_excluding_echo(stream: bool) -> tuple[object, ...]:
    with _echo_tool_listed():
        response: Final = await litellm.aresponses(
            model="openai/gpt-5.6",
            input="Call echo with PING",
            tools=[dict(_ECHO_MCP_TOOL)],
            tool_choice=_ALLOWED_ONLY_ANOTHER_TOOL,
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com/v1",
            stream=stream,
        )
        return await _drain(response) if stream else (response,)


@pytest.mark.asyncio
async def test_an_mcp_tool_the_choice_excludes_never_runs_when_the_model_ignored_the_choice(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY,))
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    (response,) = await _call_with_a_choice_excluding_echo(stream=False)

    call_tool.assert_not_awaited()
    assert isinstance(response, ResponsesAPIResponse)
    assert response.status == "completed"
    assert _output_types(response.output) == ("function_call",)


def _anthropic_call_to_echo() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_echo",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "tool_use", "id": "toolu_echo", "name": "echo", "input": {"value": "PING"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


async def _call_echo_on_anthropic_under_drop_params(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, rounds: Sequence[httpx.Response], tool_choice: object
) -> tuple[object, respx.Route]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(side_effect=list(rounds))
    with _echo_tool_listed():
        response: Final = await litellm.aresponses(
            model="anthropic/claude-haiku-4-5",
            input="Call echo with PING",
            tools=[dict(_ECHO_MCP_TOOL)],
            tool_choice=tool_choice,
            drop_params=True,
            api_key="sk-ant-fake",
        )
    return response, route


@pytest.mark.asyncio
async def test_a_dropped_choice_hands_back_an_mcp_call_it_excludes(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    response, route = await _call_echo_on_anthropic_under_drop_params(
        respx_mock, monkeypatch, (_anthropic_call_to_echo(),), _ALLOWED_ONLY_ANOTHER_TOOL
    )

    call_tool.assert_not_awaited()
    assert "tool_choice" not in json.loads(route.calls[0].request.content)
    assert isinstance(response, ResponsesAPIResponse)
    assert response.tool_choice == "auto"
    assert _output_types(response.output) == ("function_call",)


def _anthropic_answer() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_answer",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "PING back"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "allowed_tools", "mode": "auto", "tools": []},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "name": ["echo"]}]},
    ],
)
async def test_a_malformed_choice_under_drop_params_runs_an_mcp_call_as_if_never_sent(
    tool_choice: dict[str, object], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    response, route = await _call_echo_on_anthropic_under_drop_params(
        respx_mock, monkeypatch, (_anthropic_call_to_echo(), _anthropic_answer()), tool_choice
    )

    assert call_tool.await_count == 1
    assert all("tool_choice" not in json.loads(call.request.content) for call in route.calls)
    assert isinstance(response, ResponsesAPIResponse)
    assert "message" in _output_types(response.output)


@pytest.mark.asyncio
async def test_a_malformed_choice_without_drop_params_answers_400_before_listing_or_calling_a_model(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, AsyncMock())
    monkeypatch.setattr("litellm.proxy.mcp_server.server._get_tools_from_mcp_servers", listing)

    with pytest.raises(litellm.BadRequestError, match="Invalid tool choice"):
        await litellm.aresponses(
            model="openai/gpt-5.6",
            input="Call echo with PING",
            tools=[dict(_ECHO_MCP_TOOL)],
            tool_choice={"type": "allowed_tools", "mode": "auto", "tools": []},
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com/v1",
        )

    listing.assert_not_awaited()
    assert not respx_mock.calls


_DEMO_LABELED_ECHO_TOOL: Final = MappingProxyType({**_ECHO_MCP_TOOL, "server_label": "demo"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "credentials"),
    [
        ("anthropic/claude-haiku-5-5", {"api_key": "sk-ant-fake"}),
        ("openai/gpt-5.6", {"api_key": "sk-mcp-rounds", "api_base": "https://mcp-rounds.example.com/v1"}),
    ],
)
async def test_an_mcp_choice_forcing_a_gateway_server_under_drop_params_answers_400_instead_of_skipping_the_tool(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, model: str, credentials: dict[str, str]
) -> None:
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(side_effect=[_anthropic_answer()])
    respx_mock.post("https://mcp-rounds.example.com/v1/responses").mock(
        side_effect=[httpx.Response(200, content=json.dumps(_ROUND_TWO_BODY, default=dict))]
    )

    with _echo_tool_listed() as listing, pytest.raises(litellm.BadRequestError) as exc:
        await litellm.aresponses(
            model=model,
            input="Call echo with PING",
            tools=[dict(_DEMO_LABELED_ECHO_TOOL)],
            tool_choice={"type": "mcp", "server_label": "demo"},
            drop_params=True,
            **credentials,
        )

    assert exc.value.status_code == 400
    assert "cannot force `demo`, which runs on the LiteLLM MCP gateway" in exc.value.message
    listing.assert_not_awaited()
    call_tool.assert_not_awaited()
    assert not respx_mock.calls


async def _call_echo_with_the_demo_label(tool_choice: object, stream: bool) -> tuple[object, ...]:
    with _echo_tool_listed():
        response: Final = await litellm.aresponses(
            model="openai/gpt-5.6",
            input="Call echo with PING",
            tools=[dict(_DEMO_LABELED_ECHO_TOOL)],
            tool_choice=tool_choice,
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com/v1",
            stream=stream,
        )
        return await _drain(response) if stream else (response,)


@pytest.mark.asyncio
async def test_the_provider_gets_the_converted_function_in_place_of_a_gateway_mcp_entry(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    route: Final = _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY))
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    await _call_echo_with_the_demo_label(
        {"type": "allowed_tools", "mode": "required", "tools": [{"type": "mcp", "server_label": "demo", "name": "echo"}]},
        stream=False,
    )

    assert json.loads(route.calls[0].request.content)["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "echo"}],
    }
    assert call_tool.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_the_follow_up_after_a_gateway_call_keeps_the_converted_allowlist_without_forcing_a_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    route: Final = _serve_responses(
        respx_mock,
        monkeypatch,
        tuple(
            _streamed(body) if stream else httpx.Response(200, content=json.dumps(body, default=dict))
            for body in (_ROUND_ONE_BODY, _ROUND_TWO_BODY)
        ),
    )
    _served_by_a_recording_mcp_server(
        monkeypatch, AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    )

    await _call_echo_with_the_demo_label(
        {"type": "allowed_tools", "mode": "required", "tools": [{"type": "mcp", "server_label": "demo", "name": "echo"}]},
        stream=stream,
    )

    assert json.loads(route.calls[1].request.content)["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "echo"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_gateway_mcp_entry_that_allows_no_converted_tool_sends_none_and_runs_nothing(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    route: Final = _serve_responses(
        respx_mock,
        monkeypatch,
        (_streamed(_ROUND_ONE_BODY) if stream else httpx.Response(200, content=json.dumps(_ROUND_ONE_BODY, default=dict)),),
    )
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    await _call_echo_with_the_demo_label(
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "mcp", "server_label": "demo", "name": "reverse"}]},
        stream=stream,
    )

    assert json.loads(route.calls[0].request.content)["tool_choice"] == "none"
    call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_mcp_tool_the_choice_lists_still_runs(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY))
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    with _echo_tool_listed():
        await litellm.aresponses(
            model="openai/gpt-5.6",
            input="Call echo with PING",
            tools=[dict(_ECHO_MCP_TOOL)],
            tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "echo"}]},
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com/v1",
        )

    assert call_tool.await_count == 1


@pytest.mark.asyncio
async def test_a_streamed_mcp_tool_the_choice_excludes_never_runs_when_the_model_ignored_the_choice(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY),))
    call_tool: Final = AsyncMock()
    _served_by_a_recording_mcp_server(monkeypatch, call_tool)

    events: Final = await _call_with_a_choice_excluding_echo(stream=True)

    call_tool.assert_not_awaited()
    last: Final = events[-1]
    assert isinstance(last, ResponseCompletedEvent)
    assert _output_types(last.response.output) == ("function_call",)
    assert not any("mcp_call" in str(getattr(event, "type", "")) for event in events)


def _rounds_logging_obj(recorder: _RoundRecorder, call_id: str) -> Logging:
    return Logging(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Call echo with PING"}],
        stream=False,
        call_type="aresponses",
        start_time=datetime(2026, 1, 1),
        litellm_call_id=call_id,
        function_id=call_id,
        dynamic_async_success_callbacks=[recorder],
        dynamic_success_callbacks=[recorder, _no_op_sync_callback],
        dynamic_async_failure_callbacks=[recorder],
    )


def _output_types(output: Sequence[object]) -> tuple[object, ...]:
    return tuple(item.get("type") if isinstance(item, Mapping) else getattr(item, "type", None) for item in output)


def _assert_one_row_bills_both_rounds(logged: tuple[StandardLoggingPayload, ...], client: ResponsesAPIResponse) -> None:
    assert len(logged) == 1
    row: Final = logged[0]
    both_rounds: Final = _round_cost(_ROUND_ONE_BODY) + _round_cost(_ROUND_TWO_BODY)
    assert row["id"] == client.id
    assert _output_types(row["response"]["output"]) == _output_types(client.output)
    assert "message" in _output_types(client.output)
    assert "function_call" not in _output_types(client.output)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (71 + 149, 19 + 5)
    assert row["metadata"]["usage_object"]["prompt_tokens_details"]["cached_tokens"] == 32 + 64
    assert row["response_cost"] == pytest.approx(both_rounds)
    assert row["cost_breakdown"]["total_cost"] == pytest.approx(both_rounds)
    assert client.hidden_params["response_cost"] == pytest.approx(both_rounds)
    assert client.usage is not None
    assert (client.usage.input_tokens, client.usage.output_tokens) == (71 + 149, 19 + 5)


@pytest.mark.asyncio
async def test_aresponses_api_with_mcp_records_each_round_in_the_metadata_the_caller_passed(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY))
    caller_metadata: Final[Mapping[str, object]] = {"run": "nightly"}

    with _echo_tool_served("PING"):
        response: Final = await litellm.aresponses_api_with_mcp(
            input="Call echo with PING",
            model="openai/gpt-5.6",
            tools=[dict(_ECHO_MCP_TOOL)],
            metadata=caller_metadata,
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com/v1",
        )

    assert isinstance(response, ResponsesAPIResponse)
    timing_windows: Final = caller_metadata.get("llm_api_timing_windows")
    assert caller_metadata.get("run") == "nightly"
    assert isinstance(timing_windows, Sequence) and len(timing_windows) == 2


@pytest.mark.asyncio
async def test_non_stream_auto_execute_logs_the_client_response_billed_for_both_rounds(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    recorder: Final = _RoundRecorder()
    response: Final = await _auto_execute_echo(
        respx_mock, monkeypatch, _rounds_logging_obj(recorder, "mcp-rounds-immediate")
    )

    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_row_bills_both_rounds(recorder.logged, response)


@pytest.mark.asyncio
async def test_non_stream_auto_execute_deferred_for_a_post_call_guardrail_logs_both_rounds_once(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-rounds-deferred")
    logging_obj.defer_async_logging = True
    response: Final = await _auto_execute_echo(respx_mock, monkeypatch, logging_obj)

    enqueue_deferred: Final = logging_obj.enqueue_deferred_logging
    assert enqueue_deferred is not None
    enqueue_deferred()
    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_row_bills_both_rounds(recorder.logged, response)


@pytest.mark.asyncio
async def test_non_stream_auto_execute_hands_sync_callbacks_the_client_response(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    recorder: Final = _RoundRecorder()
    response: Final = await _auto_execute_echo(respx_mock, monkeypatch, _rounds_logging_obj(recorder, "mcp-rounds-sync"))

    await _settle(lambda: len(recorder.hooked) > 0)

    assert len(recorder.hooked) == 1
    hooked: Final = recorder.hooked[0]
    assert isinstance(hooked, ResponsesAPIResponse)
    assert hooked.id == response.id
    assert _output_types(hooked.output) == _output_types(response.output)
    assert hooked.usage is not None
    assert (hooked.usage.model_dump().get("prompt_tokens"), hooked.usage.model_dump().get("completion_tokens")) == (
        71 + 149,
        19 + 5,
    )


@pytest.mark.asyncio
async def test_non_stream_auto_execute_bills_the_paid_first_round_when_the_follow_up_fails(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    context_window_error: Final = {
        "error": {
            "message": "Your input exceeds the context window of this model.",
            "type": "invalid_request_error",
            "param": "input",
            "code": "context_length_exceeded",
        }
    }
    _serve_responses(
        respx_mock,
        monkeypatch,
        (
            httpx.Response(200, content=json.dumps(_ROUND_ONE_BODY, default=dict)),
            httpx.Response(400, json=context_window_error),
        ),
    )
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    recorder: Final = _RoundRecorder()

    with pytest.raises(litellm.BadRequestError):
        await _auto_execute_through_mcp_server(
            monkeypatch, call_tool, _rounds_logging_obj(recorder, "mcp-rounds-follow-up-fails")
        )
    await _settle(lambda: len(recorder.failed) > 0)

    assert recorder.logged == ()
    assert len(recorder.failed) == 1
    assert (recorder.failed[0]["prompt_tokens"], recorder.failed[0]["completion_tokens"]) == (71, 19)
    assert recorder.failed[0]["response_cost"] == pytest.approx(_round_cost(_ROUND_ONE_BODY))
    assert recorder.failed[0]["response_cost"] > 0
    assert call_tool.await_count == 1
    assert call_tool.await_args is not None
    tool_logging_obj: Final = call_tool.await_args.kwargs["litellm_logging_obj"]
    assert tool_logging_obj.litellm_call_id == f"{recorder.failed[0]['id']}_call_echo"
    assert tool_logging_obj.litellm_trace_id == recorder.failed[0]["trace_id"]


_RETRIED_ROUND_ONE: Final = MappingProxyType(
    {
        **_ROUND_ONE_BODY,
        "id": "resp_retried_round_one",
        "usage": {
            "input_tokens": 500,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 540,
        },
    }
)
_SERVER_ERROR: Final = MappingProxyType(
    {"error": {"message": "upstream overloaded", "type": "server_error", "code": "server_error"}}
)


def _ok(body: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(body, default=dict))


def _server_error() -> httpx.Response:
    return httpx.Response(500, content=json.dumps(_SERVER_ERROR, default=dict))


def _router_retrying_once() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5.6",
                "litellm_params": {
                    "model": "openai/gpt-5.6",
                    "api_key": "sk-mcp-rounds",
                    "api_base": "https://mcp-rounds.example.com/v1",
                },
            }
        ],
        num_retries=1,
        retry_after=0,
    )


@pytest.mark.asyncio
async def test_non_stream_auto_execute_retried_after_a_failed_follow_up_bills_each_attempt_its_own_rounds(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_ok(_ROUND_ONE_BODY), _server_error(), _ok(_RETRIED_ROUND_ONE), _ok(_ROUND_TWO_BODY)),
    )
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        response: Final = await _router_retrying_once().aresponses(
            model="gpt-5.6",
            input="Call echo with PING",
            tools=[dict(_ECHO_MCP_TOOL)],
            litellm_logging_obj=_rounds_logging_obj(recorder, "mcp-rounds-retried"),
        )
    await _settle(lambda: len(recorder.logged) > 0)

    assert isinstance(response, ResponsesAPIResponse)
    assert response.usage is not None
    assert len(recorder.failed) == 1
    assert (recorder.failed[0]["prompt_tokens"], recorder.failed[0]["completion_tokens"]) == (71, 19)
    assert recorder.failed[0]["response_cost"] == pytest.approx(_round_cost(_ROUND_ONE_BODY))
    assert len(recorder.logged) == 1
    row: Final = recorder.logged[0]
    assert row["id"] == response.id
    assert (row["prompt_tokens"], row["completion_tokens"]) == (500 + 149, 40 + 5)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (response.usage.input_tokens, response.usage.output_tokens)
    assert row["response_cost"] == pytest.approx(_round_cost(_RETRIED_ROUND_ONE) + _round_cost(_ROUND_TWO_BODY))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retried_first_rounds", "billed_tokens"),
    [
        pytest.param((), (71, 19), id="the retry fails its first call"),
        pytest.param((_RETRIED_ROUND_ONE,), (71 + 500, 19 + 40), id="the retry fails its follow-up"),
    ],
)
async def test_non_stream_auto_execute_failing_every_retry_bills_each_paid_first_round_once_on_the_request_failure(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    retried_first_rounds: tuple[Mapping[str, object], ...],
    billed_tokens: tuple[int, int],
):
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_ok(_ROUND_ONE_BODY), _server_error(), *(_ok(body) for body in retried_first_rounds), _server_error()),
    )
    logging_obj: Final = _rounds_logging_obj(_RoundRecorder(), "mcp-rounds-every-retry-fails")

    with _echo_tool_served("PING"), pytest.raises(litellm.InternalServerError):
        await _router_retrying_once().aresponses(
            model="gpt-5.6",
            input="Call echo with PING",
            tools=[dict(_ECHO_MCP_TOOL)],
            litellm_logging_obj=logging_obj,
        )

    failure_row_fields: Final = _failure_fields_to_lift(MappingProxyType({"litellm_logging_obj": logging_obj}))
    billed_usage: Final = failure_row_fields["combined_usage_object"]
    assert isinstance(billed_usage, Usage)
    estimated_failure_usage: Final = (
        _estimate_dispatched_failure_usage(
            model="gpt-5.6",
            request_input=logging_obj.model_call_details.get("messages"),
            system_input=None,
        )
        if not retried_first_rounds
        else None
    )
    expected_prompt_tokens: Final = billed_tokens[0] + (
        estimated_failure_usage.prompt_tokens if estimated_failure_usage is not None else 0
    )
    assert (billed_usage.prompt_tokens, billed_usage.completion_tokens) == (
        expected_prompt_tokens,
        billed_tokens[1],
    )
    assert failure_row_fields["response_cost"] == pytest.approx(
        _round_cost(_ROUND_ONE_BODY) + sum(_round_cost(body) for body in retried_first_rounds)
    )


@pytest.mark.asyncio
async def test_non_stream_auto_execute_tool_log_continues_the_request_trace_without_renaming_it(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY))
    call_tool: Final = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")]))
    recorder: Final = _RoundRecorder()

    await _auto_execute_through_mcp_server(monkeypatch, call_tool, _rounds_logging_obj(recorder, "mcp-rounds-trace"))
    await _settle(lambda: len(recorder.logged) > 0)

    assert call_tool.await_args is not None
    tool_metadata: Final = call_tool.await_args.kwargs["litellm_logging_obj"].litellm_params["metadata"]
    assert tool_metadata["existing_trace_id"] == recorder.logged[0]["trace_id"]


class _ListingRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.listings: tuple[Mapping[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("call_type") == "list_mcp_tools":
            self.listings = (*self.listings, kwargs)


def _serve_the_demo_mcp_listing(monkeypatch: pytest.MonkeyPatch) -> _ListingRecorder:
    echo_tool: Final = MCPTool.model_validate(
        {"name": "echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
        by_name=False,
    )
    demo_server: Final = SimpleNamespace(
        server_id="demo", name="demo", alias="demo", server_name="demo", auth_type=None, extra_headers=None
    )
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.proxy_server", SimpleNamespace(proxy_logging_obj=None, prisma_client=None)
    )
    monkeypatch.setattr(
        "litellm.proxy.mcp_server.mcp_server_manager.global_mcp_server_manager",
        SimpleNamespace(
            catalog=SimpleNamespace(operation=nullcontext),
            get_registry=MagicMock(return_value={}),
            call_tool=AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="PING")])),
            get_mcp_server_from_tool_name=MagicMock(return_value=None),
            get_mcp_server_by_name=MagicMock(return_value=None),
            get_allowed_mcp_servers=AsyncMock(return_value=["demo"]),
            get_mcp_servers_from_ids=MagicMock(return_value=[demo_server]),
        ),
    )
    monkeypatch.setattr(
        "litellm.proxy.mcp_server.operations._get_allowed_mcp_servers",
        AsyncMock(return_value=[demo_server]),
    )
    monkeypatch.setattr(
        "litellm.proxy.mcp_server.catalog.get_filtered_server_tools",
        AsyncMock(return_value=(ListToolsResult(tools=[echo_tool]), ServerListOk(tool_count=1))),
    )
    listing_recorder: Final = _ListingRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [listing_recorder])
    return listing_recorder


async def _assert_the_listing_logs_in_the_request_trace(
    listing_recorder: _ListingRecorder, request_recorder: _RoundRecorder
) -> None:
    await _settle(lambda: len(request_recorder.logged) > 0 and len(listing_recorder.listings) > 0)
    request_trace_id: Final = request_recorder.logged[0]["trace_id"]
    assert len(listing_recorder.listings) == 1
    listing: Final = listing_recorder.listings[0]
    listing_row: Final = cast(StandardLoggingPayload, listing["standard_logging_object"])
    assert listing_row["trace_id"] == request_trace_id
    assert cast(Mapping[str, Mapping[str, object]], listing["litellm_params"])["metadata"][
        "existing_trace_id"
    ] == request_trace_id


@pytest.mark.asyncio
async def test_non_stream_auto_execute_tool_listing_log_continues_the_request_trace(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY))
    listing_recorder: Final = _serve_the_demo_mcp_listing(monkeypatch)
    request_recorder: Final = _RoundRecorder()

    await litellm.aresponses(
        model="openai/gpt-5.6",
        input="Call echo with PING",
        tools=[dict(_ECHO_MCP_TOOL)],
        api_key="sk-mcp-rounds",
        api_base="https://mcp-rounds.example.com/v1",
        litellm_call_id="mcp-listing-trace",
        litellm_trace_id="router-trace",
        litellm_logging_obj=_rounds_logging_obj(request_recorder, "mcp-listing-trace"),
    )

    await _assert_the_listing_logs_in_the_request_trace(listing_recorder, request_recorder)


@pytest.mark.asyncio
async def test_bridged_auto_execute_tool_listing_log_continues_the_request_trace(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post("https://mcp-rounds.example.com/v1/messages").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_anthropic_message(
                    "msg_tool_round",
                    ({"type": "tool_use", "id": "toolu_echo", "name": "echo", "input": {"value": "PING"}},),
                    "tool_use",
                ),
            ),
            httpx.Response(200, json=_anthropic_message("msg_answer", ({"type": "text", "text": "PING"},), "end_turn")),
        ]
    )
    listing_recorder: Final = _serve_the_demo_mcp_listing(monkeypatch)
    request_recorder: Final = _RoundRecorder()

    await litellm.aresponses(
        model="anthropic/claude-haiku-4-5",
        input="Call echo with PING",
        tools=[dict(_ECHO_MCP_TOOL)],
        api_key="sk-mcp-rounds",
        api_base="https://mcp-rounds.example.com",
        litellm_call_id="mcp-bridged-listing-trace",
        litellm_trace_id="router-trace",
        litellm_logging_obj=_rounds_logging_obj(request_recorder, "mcp-bridged-listing-trace"),
    )

    await _assert_the_listing_logs_in_the_request_trace(listing_recorder, request_recorder)


class _CallRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: tuple[Mapping[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged = (*self.logged, kwargs)

    def of(self, call_type: str) -> tuple[Mapping[str, object], ...]:
        return tuple(kwargs for kwargs in self.logged if kwargs.get("call_type") == call_type)


def _langfuse_trace_of(logged: Mapping[str, object]) -> object:
    litellm_params: Final = cast(Mapping[str, object], logged["litellm_params"])
    return langfuse_trace_id(
        metadata=langfuse_trace_metadata(litellm_params["metadata"], litellm_params.get("proxy_server_request")),
        standard_trace_id=cast(StandardLoggingPayload, logged["standard_logging_object"])["trace_id"],
        litellm_trace_id=logged.get("litellm_trace_id"),
        litellm_call_id=logged.get("litellm_call_id"),
        proxy_server_request=litellm_params.get("proxy_server_request"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_metadata",
    [None, {"trace_id": "client-trace", "session_id": "client-session"}],
    ids=["no_body_metadata", "body_metadata_with_its_own_trace_and_session"],
)
async def test_auto_execute_mcp_logs_join_the_langfuse_trace_of_a_session_header_request(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, body_metadata: dict[str, str] | None
):
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY))
    listing_recorder: Final = _serve_the_demo_mcp_listing(monkeypatch)
    call_recorder: Final = _CallRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [listing_recorder, call_recorder])
    session_id: Final = "sess-abc12345"

    await litellm.aresponses(
        model="openai/gpt-5.6",
        input="Call echo with PING",
        tools=[dict(_ECHO_MCP_TOOL)],
        api_key="sk-mcp-rounds",
        api_base="https://mcp-rounds.example.com/v1",
        litellm_call_id="mcp-session-trace",
        litellm_trace_id=session_id,
        litellm_session_id=session_id,
        metadata=body_metadata,
        litellm_metadata={"trace_id": session_id, "session_id": session_id},
        proxy_server_request={"headers": {"x-litellm-session-id": session_id}},
    )
    await _settle(lambda: len(call_recorder.of("aresponses")) > 0 and len(listing_recorder.listings) > 0)

    parent_trace_id: Final = _langfuse_trace_of(call_recorder.of("aresponses")[0])
    call_tool: Final = cast(AsyncMock, mcp_server_manager.global_mcp_server_manager.call_tool)
    assert call_tool.await_args is not None
    tool_logging_obj: Final = cast(Logging, call_tool.await_args.kwargs["litellm_logging_obj"])
    mcp_log_traces: Final = (
        cast(Mapping[str, Mapping[str, object]], listing_recorder.listings[0]["litellm_params"])["metadata"][
            "existing_trace_id"
        ],
        cast(Mapping[str, Mapping[str, object]], tool_logging_obj.model_call_details["litellm_params"])["metadata"][
            "existing_trace_id"
        ],
    )
    assert parent_trace_id == "mcp-session-trace"
    assert mcp_log_traces == (parent_trace_id, parent_trace_id)


@pytest.mark.asyncio
async def test_non_stream_auto_execute_bills_the_paid_first_round_when_the_client_disconnects_during_a_tool_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY,))
    logging_obj: Final = _rounds_logging_obj(_RoundRecorder(), "mcp-rounds-disconnect")

    with pytest.raises(asyncio.CancelledError):
        await _auto_execute_through_mcp_server(
            monkeypatch, AsyncMock(side_effect=asyncio.CancelledError()), logging_obj
        )

    failure_row_fields: Final = _failure_fields_to_lift(MappingProxyType({"litellm_logging_obj": logging_obj}))
    billed_usage: Final = failure_row_fields["combined_usage_object"]
    assert isinstance(billed_usage, Usage)
    assert (billed_usage.prompt_tokens, billed_usage.completion_tokens) == (71, 19)
    assert failure_row_fields["response_cost"] == pytest.approx(_round_cost(_ROUND_ONE_BODY))


@pytest.mark.asyncio
async def test_non_stream_auto_execute_bills_nothing_for_rounds_served_from_the_response_cache(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    """A cached final round hands back the id of the turn that cached it, so that turn's row is logged under a
    unique id prefixed by the client's, and as a cache hit only when no round was paid."""
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    provider: Final = _serve_rounds(respx_mock, monkeypatch, (_ROUND_ONE_BODY, _ROUND_TWO_BODY, _ROUND_TWO_BODY))
    warm: Final = _RoundRecorder()
    await _auto_execute(_rounds_logging_obj(warm, "mcp-cache-warm"), _ECHO_MCP_TOOL, "PING")
    await _settle(lambda: len(warm.logged) > 0)

    first_cached: Final = _RoundRecorder()
    first_cached_response: Final = await _auto_execute(
        _rounds_logging_obj(first_cached, "mcp-cache-first-round"),
        {**_ECHO_MCP_TOOL, "server_description": "echo server"},
        "PONG",
    )
    all_cached: Final = _RoundRecorder()
    all_cached_response: Final = await _auto_execute(
        _rounds_logging_obj(all_cached, "mcp-cache-every-round"),
        {**_ECHO_MCP_TOOL, "server_description": "echo server, same tools"},
        "PONG",
    )
    await _settle(lambda: len(first_cached.logged) > 0 and len(all_cached.logged) > 0)

    assert provider.call_count == 3
    for logged, client in ((first_cached.logged, first_cached_response), (all_cached.logged, all_cached_response)):
        assert len(logged) == 1
        assert logged[0]["id"].startswith(client.id)
        assert _output_types(logged[0]["response"]["output"]) == _output_types(client.output)
        assert "function_call" not in _output_types(client.output)
        assert (logged[0]["prompt_tokens"], logged[0]["completion_tokens"]) == (71 + 149, 19 + 5)
    assert first_cached.logged[0]["response_cost"] == pytest.approx(_round_cost(_ROUND_TWO_BODY))
    assert first_cached.logged[0]["cost_breakdown"]["total_cost"] == pytest.approx(_round_cost(_ROUND_TWO_BODY))
    assert first_cached_response.hidden_params["response_cost"] == pytest.approx(_round_cost(_ROUND_TWO_BODY))
    assert all_cached.logged[0]["response_cost"] == 0.0
    assert all_cached_response.hidden_params["response_cost"] == 0.0
    assert first_cached.logged[0]["id"] == first_cached_response.id
    assert first_cached.logged[0]["cache_hit"] is None
    assert all_cached.logged[0]["id"].startswith(f"{all_cached_response.id}_cache_hit")
    assert all_cached.logged[0]["cache_hit"] is True


_PRIOR_TURN: Final = MappingProxyType(
    {
        "request_id": "chatcmpl-prior-turn",
        "session_id": "prior-session",
        "proxy_server_request": {"input": "My favorite color is teal."},
        "response": {
            "id": "chatcmpl-prior-turn",
            "object": "chat.completion",
            "created": 1767225600,
            "model": "claude-haiku-4-5",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Noted."}}],
        },
    }
)


def _anthropic_message(message_id: str, content: Sequence[Mapping[str, object]], stop: str) -> dict[str, object]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [dict(block) for block in content],
        "stop_reason": stop,
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


class _SpendLogs:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows: Final = tuple(rows)
        self.asked: tuple[frozenset[str], ...] = ()

    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
        self.asked = (*self.asked, frozenset(request_ids))
        return [dict(row) for row in self.rows if row["request_id"] in request_ids]


@pytest.mark.asyncio
async def test_bridged_auto_execute_follow_up_continues_the_client_conversation_in_its_session(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    spend_logs: Final = _SpendLogs((_PRIOR_TURN,))
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(proxy_logging_obj=None, prisma_client=SimpleNamespace(db=spend_logs), disable_spend_logs=False),
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    thinking: Final = {"type": "thinking", "thinking": "Echo PING back.", "signature": "sig-round-one"}
    provider: Final = respx_mock.post("https://mcp-rounds.example.com/v1/messages").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_anthropic_message(
                    "msg_tool_round",
                    (thinking, {"type": "tool_use", "id": "toolu_echo", "name": "echo", "input": {"value": "PING"}}),
                    "tool_use",
                ),
            ),
            httpx.Response(200, json=_anthropic_message("msg_answer", ({"type": "text", "text": "PING"},), "end_turn")),
        ]
    )
    echo_tool: Final = MCPTool.model_validate(
        {"name": "echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
        by_name=False,
    )
    execute: Final = AsyncMock(return_value=[{"tool_call_id": "toolu_echo", "result": "PING", "name": "echo"}])
    recorder: Final = _RoundRecorder()

    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            AsyncMock(return_value=([echo_tool], {"echo": "demo"})),
        ),
        patch.object(LiteLLM_Proxy_MCP_Handler, "execute_tool_calls", execute),
    ):
        await litellm.aresponses(
            model="anthropic/claude-haiku-4-5",
            input="Call echo with PING",
            previous_response_id="chatcmpl-prior-turn",
            tools=[dict(_ECHO_MCP_TOOL)],
            api_key="sk-mcp-rounds",
            api_base="https://mcp-rounds.example.com",
            litellm_logging_obj=_rounds_logging_obj(recorder, "mcp-rounds-bridged"),
        )
    await _settle(lambda: len(recorder.logged) > 0)

    follow_up: Final = json.loads(provider.calls[1].request.content)["messages"]
    assert spend_logs.asked and set(spend_logs.asked) == {frozenset({"chatcmpl-prior-turn"})}
    assert [message["role"] for message in follow_up] == ["user", "assistant", "user", "assistant", "user"]
    assert json.dumps(follow_up).count("My favorite color is teal.") == 1
    assert [block["type"] for block in follow_up[3]["content"]] == ["thinking", "tool_use"]
    assert follow_up[3]["content"][0]["signature"] == "sig-round-one"
    assert execute.await_args is not None
    assert execute.await_args.kwargs["litellm_trace_id"] == "prior-session"
    assert recorder.logged[0]["trace_id"] == "prior-session"


def _event_type(event: object) -> str:
    event_type: Final = getattr(event, "type", "")
    return str(getattr(event_type, "value", event_type))


def _first_index(events: Sequence[object], event_type: str) -> int:
    return next(index for index, event in enumerate(events) if _event_type(event) == event_type)


async def _scripted_mcp_stream(
    model: str,
    upstream_url: str,
    rounds: Sequence[bytes],
    max_agentic_loops: int | None = None,
) -> tuple[tuple[object, ...], respx.Route, AsyncMock, AsyncMock, MCPEnhancedStreamingIterator]:
    upstream: Final = respx.post(upstream_url).mock(
        side_effect=[httpx.Response(200, content=body, headers=SSE_HEADERS) for body in rounds]
    )
    with (
        patch.object(LiteLLM_Proxy_MCP_Handler, "_get_mcp_tools_from_manager", new_callable=AsyncMock) as fetch,
        patch.object(LiteLLM_Proxy_MCP_Handler, "execute_tool_calls", new_callable=AsyncMock) as execute,
    ):
        fetch.return_value = ([WEATHER_TOOL], ["litellm_proxy"])
        execute.return_value = [{"tool_call_id": "call_weather", "result": WEATHER_RESULT}]
        response: Final = await litellm.aresponses(
            model=model,
            api_key="sk-scripted",
            tools=[cast(Any, {"type": "mcp", "server_url": "litellm_proxy", "require_approval": "never"})],
            tool_choice="required",
            input=[{"role": "user", "type": "message", "content": "What's the weather in San Francisco?"}],
            stream=True,
            **({"max_agentic_loops": max_agentic_loops} if max_agentic_loops is not None else {}),
        )
        assert isinstance(response, MCPEnhancedStreamingIterator)
        events: Final = tuple([event async for event in response])
    return events, upstream, fetch, execute, response


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", tuple(SCRIPTED_PROVIDERS))
@respx.mock
async def test_streaming_responses_api_with_mcp_tools(
    provider: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model, upstream_url, tool_round, answer_round = SCRIPTED_PROVIDERS[provider]

    with caplog.at_level(logging.ERROR):
        events, upstream, fetch, execute, _ = await _scripted_mcp_stream(
            model, upstream_url, (tool_round(), answer_round())
        )

    assert fetch.await_count == 1
    executed: Final = execute.await_args.kwargs["tool_calls"]
    assert [(call.name, json.loads(call.arguments)) for call in executed] == [("get_weather", WEATHER_ARGUMENTS)]
    assert upstream.call_count == 2
    first_body: Final = json.loads(upstream.calls[0].request.content)
    follow_up: Final = json.dumps(json.loads(upstream.calls[1].request.content))
    assert "get_weather" in json.dumps(first_body["tools"])
    assert WEATHER_RESULT in follow_up and "call_weather" in follow_up
    event_types: Final = [_event_type(event) for event in events]
    assert event_types.index("response.mcp_list_tools.completed") < event_types.index("response.mcp_call.completed")
    assert event_types[-1] == "response.completed"
    assert WEATHER_ANSWER in json.dumps(events[-1].response.model_dump(mode="json"))
    assert not [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.asyncio
@respx.mock
async def test_streaming_mcp_iterator_uses_max_agentic_loops_from_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model, upstream_url, tool_round, answer_round = SCRIPTED_PROVIDERS["openai"]

    _, _, _, _, iterator = await _scripted_mcp_stream(
        model,
        upstream_url,
        (tool_round(), answer_round()),
        max_agentic_loops=7,
    )

    assert iterator.max_tool_call_rounds == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "one_id"), [("openai", False), ("anthropic", True)], ids=["native", "bridged"])
@respx.mock
async def test_streaming_mcp_event_order_and_response_id_consistency(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, provider: str, one_id: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model, upstream_url, tool_round, answer_round = SCRIPTED_PROVIDERS[provider]

    with caplog.at_level(logging.ERROR):
        events, _, _, _, _ = await _scripted_mcp_stream(model, upstream_url, (tool_round(), answer_round()))

    created: Final = _first_index(events, "response.created")
    in_progress: Final = _first_index(events, "response.in_progress")
    output_item_added: Final = _first_index(events, "response.output_item.added")
    mcp_list_in_progress: Final = _first_index(events, "response.mcp_list_tools.in_progress")
    completed: Final = _first_index(events, "response.completed")
    assert created < in_progress < output_item_added < mcp_list_in_progress < completed
    lifecycle_ids: Final = [
        event.response.id
        for event in events
        if _event_type(event) in ("response.created", "response.in_progress", "response.completed")
    ]
    assert len(lifecycle_ids) == 3
    assert lifecycle_ids[0] == lifecycle_ids[1]
    assert (lifecycle_ids[2] == lifecycle_ids[0]) is one_id
    assert not [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]


_FIXED_MARGIN: Final = 0.001


def _streamed(body: Mapping[str, object], **terminal: object) -> httpx.Response:
    completed: Final = json.loads(json.dumps({**body, **terminal}, default=dict))
    opened: Final = {**completed, "status": "in_progress", "output": [], "usage": None}
    items: Final = tuple(enumerate(completed["output"]))
    events: Final = (
        {"type": "response.created", "response": opened},
        {"type": "response.in_progress", "response": opened},
        *({"type": "response.output_item.added", "output_index": index, "item": item} for index, item in items),
        *({"type": "response.output_item.done", "output_index": index, "item": item} for index, item in items),
        {"type": f"response.{completed['status']}", "response": completed},
    )
    return httpx.Response(
        200,
        content=_sse(tuple({**event, "sequence_number": number} for number, event in enumerate(events))),
        headers=SSE_HEADERS,
    )


class _CutAfterTheFirstEvent(httpx.AsyncByteStream):
    async def __aiter__(self):
        opened: Final = {**json.loads(json.dumps(_ROUND_TWO_BODY, default=dict)), "status": "in_progress"}
        yield _sse(({"type": "response.created", "response": opened, "sequence_number": 0},))
        raise httpx.ReadError("connection reset mid-stream")


async def _open_stream_auto_execute(logging_obj: Logging) -> object:
    return await litellm.aresponses(
        model="openai/gpt-5.6",
        input="Call echo with PING",
        tools=[dict(_ECHO_MCP_TOOL)],
        api_key="sk-mcp-rounds",
        api_base="https://mcp-rounds.example.com/v1",
        stream=True,
        litellm_logging_obj=logging_obj,
    )


async def _drain(stream: object) -> tuple[ResponsesAPIStreamingResponse, ...]:
    return tuple([event async for event in cast(Any, stream)])


async def _stream_auto_execute(logging_obj: Logging) -> tuple[ResponsesAPIStreamingResponse, ...]:
    with _echo_tool_served("PING"):
        return await _drain(await _open_stream_auto_execute(logging_obj))


def _final_response(
    events: Sequence[ResponsesAPIStreamingResponse], terminal: str = "response.completed"
) -> ResponsesAPIResponse:
    assert _event_type(events[-1]) == terminal
    response: Final = getattr(events[-1], "response", None)
    assert isinstance(response, ResponsesAPIResponse)
    return response


def _assert_one_stream_row_bills_both_rounds_once(
    logged: tuple[StandardLoggingPayload, ...], client: ResponsesAPIResponse
) -> None:
    charged_once: Final = _round_cost(_ROUND_ONE_BODY) + _round_cost(_ROUND_TWO_BODY) - _FIXED_MARGIN
    assert len(logged) == 1
    row: Final = logged[0]
    assert _output_types(row["response"]["output"]) == _output_types(client.output)
    assert "message" in _output_types(client.output)
    assert "function_call" not in _output_types(client.output)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (71 + 149, 19 + 5)
    assert row["cache_hit"] is None
    assert row["metadata"]["usage_object"]["prompt_tokens_details"]["cached_tokens"] == 32 + 64
    assert row["hidden_params"]["additional_headers"]["llm_provider-content-type"] == SSE_HEADERS["content-type"]
    assert row["response_cost"] == pytest.approx(charged_once)
    assert row["cost_breakdown"]["total_cost"] == pytest.approx(charged_once)
    assert client.usage is not None
    assert (client.usage.input_tokens, client.usage.output_tokens) == (71 + 149, 19 + 5)
    assert client.usage.cost == pytest.approx(charged_once)


def _assert_breakdown_agrees_with_the_billed_cost(row: StandardLoggingPayload) -> None:
    breakdown: Final = row.get("cost_breakdown")
    assert breakdown is None or breakdown["total_cost"] == pytest.approx(row["response_cost"])


def _request_failure_row(logging_obj: Logging, success_row: StandardLoggingPayload) -> Mapping[str, object]:
    failure_row: Final = _failure_fields_to_lift(MappingProxyType({"litellm_logging_obj": logging_obj}))
    failure_payload: Final = cast(StandardLoggingPayload, failure_row["standard_logging_object"])
    assert failure_payload["status"] == "failure"
    assert failure_payload["id"] != success_row["id"]
    return failure_row


def _assert_billed_tokens(row: Mapping[str, object], tokens: tuple[int, int]) -> None:
    usage: Final = row["combined_usage_object"]
    assert isinstance(usage, Usage)
    assert (usage.prompt_tokens, usage.completion_tokens) == tokens


def _assert_one_row_bills(logged: tuple[StandardLoggingPayload, ...], body: Mapping[str, object]) -> None:
    usage: Final = cast(Mapping[str, int], body["usage"])
    assert len(logged) == 1
    assert (logged[0]["prompt_tokens"], logged[0]["completion_tokens"]) == (
        usage["input_tokens"],
        usage["output_tokens"],
    )
    assert logged[0]["response_cost"] == pytest.approx(_round_cost(body))
    assert logged[0]["response_cost"] > 0


def _assert_one_row_bills_round_one_twice_with_one_fixed_margin(logged: tuple[StandardLoggingPayload, ...]) -> None:
    charged_once: Final = 2 * _round_cost(_ROUND_ONE_BODY) - _FIXED_MARGIN
    assert len(logged) == 1
    assert (logged[0]["prompt_tokens"], logged[0]["completion_tokens"]) == (2 * 71, 2 * 19)
    assert logged[0]["response_cost"] == pytest.approx(charged_once)
    assert logged[0]["cost_breakdown"]["total_cost"] == pytest.approx(charged_once)


@pytest.mark.asyncio
async def test_stream_auto_execute_logs_the_final_response_billed_for_both_rounds_with_one_fixed_margin(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_TWO_BODY)))
    recorder: Final = _RoundRecorder()

    events: Final = await _stream_auto_execute(_rounds_logging_obj(recorder, "mcp-stream-rounds"))
    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_stream_row_bills_both_rounds_once(recorder.logged, _final_response(events))


@pytest.mark.asyncio
async def test_stream_auto_execute_deferred_for_a_post_call_guardrail_logs_both_rounds_once_when_the_proxy_releases(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_TWO_BODY)))
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-stream-rounds-deferred")

    with _echo_tool_served("PING"):
        stream: Final = await _open_stream_auto_execute(logging_obj)
        ProxyBaseLLMRequestProcessing(data={})._arm_deferred_stream_dispatch(
            response=stream, route_type="aresponses", user_api_key_dict=MagicMock(), logging_obj=logging_obj
        )
        events: Final = await _drain(stream)
    await asyncio.sleep(0.2)
    logged_before_release: Final = recorder.logged
    ProxyLogging.fire_deferred_stream_logging({"litellm_logging_obj": logging_obj})
    await _settle(lambda: len(recorder.logged) > 0)

    assert logged_before_release == ()
    _assert_one_stream_row_bills_both_rounds_once(recorder.logged, _final_response(events))


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_the_paid_first_round_on_its_success_row_when_the_follow_up_stream_breaks(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.proxy.utils import ProxyLogging

    _serve_responses(
        respx_mock,
        monkeypatch,
        (_streamed(_ROUND_ONE_BODY), httpx.Response(200, stream=_CutAfterTheFirstEvent(), headers=SSE_HEADERS)),
    )
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-stream-follow-up-breaks")

    with pytest.raises(litellm.ServiceUnavailableError):
        await _stream_auto_execute(logging_obj)
    ProxyLogging.fire_deferred_stream_logging({"litellm_logging_obj": logging_obj})
    await _settle(lambda: len(recorder.logged) > 0 and len(recorder.failed) > 0)

    _assert_one_row_bills(recorder.logged, _ROUND_ONE_BODY)
    assert [row["response_cost"] for row in recorder.failed] == [0.0]
    assert _request_failure_row(logging_obj, recorder.logged[0]).get("response_cost", 0.0) == 0.0


class _CutAfterStreamedText(httpx.AsyncByteStream):
    async def __aiter__(self):
        opened: Final = {**json.loads(json.dumps(_ROUND_TWO_BODY, default=dict)), "status": "in_progress"}
        yield _sse(
            (
                {"type": "response.created", "response": opened, "sequence_number": 0},
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_round_two",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "PING " * 40,
                    "sequence_number": 1,
                },
            )
        )
        raise httpx.ReadError("connection reset mid-stream")


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_a_follow_up_cut_after_streamed_text_with_no_second_fixed_margin(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_streamed(_ROUND_ONE_BODY), httpx.Response(200, stream=_CutAfterStreamedText(), headers=SSE_HEADERS)),
    )
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-stream-follow-up-cut-after-text")

    with pytest.raises(httpx.ReadError):
        await _stream_auto_execute(logging_obj)
    ProxyLogging.fire_deferred_stream_logging({"litellm_logging_obj": logging_obj})
    await _settle(lambda: len(recorder.logged) > 0 and len(recorder.failed) > 0)

    _assert_one_row_bills(recorder.logged, _ROUND_ONE_BODY)
    assert len(recorder.failed) == 1
    failed: Final = recorder.failed[0]
    assert failed["completion_tokens"] > 0
    priced_with_its_own_margin: Final = litellm.completion_cost(
        completion_response=ModelResponse(
            model="gpt-5.6",
            usage=Usage(
                prompt_tokens=failed["prompt_tokens"],
                completion_tokens=failed["completion_tokens"],
                total_tokens=failed["prompt_tokens"] + failed["completion_tokens"],
            ),
        ),
        model="gpt-5.6",
        custom_llm_provider="openai",
    )
    assert failed["response_cost"] == pytest.approx(priced_with_its_own_margin - _FIXED_MARGIN)
    _assert_breakdown_agrees_with_the_billed_cost(failed)


@pytest.mark.asyncio
async def test_stream_auto_execute_blocked_by_a_post_call_guardrail_keeps_both_rounds_on_one_failure_row(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_TWO_BODY)))
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-stream-rounds-blocked")
    blocked: Final = GuardrailRaisedException(guardrail_name="output-policy", message="blocked")

    with _echo_tool_served("PING"):
        stream: Final = await _open_stream_auto_execute(logging_obj)
        ProxyBaseLLMRequestProcessing(data={})._arm_deferred_stream_dispatch(
            response=stream, route_type="aresponses", user_api_key_dict=MagicMock(), logging_obj=logging_obj
        )
        events: Final = await _drain(stream)
    if not ProxyLogging._discard_deferred_stream_logging_for_failure({"litellm_logging_obj": logging_obj}, blocked):
        ProxyLogging.fire_deferred_stream_logging({"litellm_logging_obj": logging_obj})
    await logging_obj.async_failure_handler(blocked, "blocked by output-policy")
    await _settle(lambda: len(recorder.failed) > 0)

    assert recorder.logged == ()
    assert len(recorder.failed) == 1
    row: Final = recorder.failed[0]
    charged_once: Final = _round_cost(_ROUND_ONE_BODY) + _round_cost(_ROUND_TWO_BODY) - _FIXED_MARGIN
    assert row["status"] == "failure"
    assert (row["prompt_tokens"], row["completion_tokens"]) == (71 + 149, 19 + 5)
    assert row["response_cost"] == pytest.approx(charged_once)
    assert row["cost_breakdown"]["total_cost"] == pytest.approx(charged_once)
    client: Final = _final_response(events)
    assert "message" in _output_types(client.output)
    assert "function_call" not in _output_types(client.output)
    assert client.usage is not None
    assert (client.usage.input_tokens, client.usage.output_tokens) == (71 + 149, 19 + 5)
    assert client.usage.cost == pytest.approx(charged_once)


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_the_paid_first_round_once_when_the_follow_up_is_rejected(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    context_window_error: Final = {
        "error": {
            "message": "Your input exceeds the context window of this model.",
            "type": "invalid_request_error",
            "param": "input",
            "code": "context_length_exceeded",
        }
    }
    _serve_responses(
        respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), httpx.Response(400, json=context_window_error))
    )
    recorder: Final = _RoundRecorder()

    events: Final = await _stream_auto_execute(_rounds_logging_obj(recorder, "mcp-stream-follow-up-rejected"))
    await _settle(lambda: len(recorder.logged) > 0 and len(recorder.failed) > 0)

    assert _event_type(events[-1]) == "error"
    _assert_one_row_bills(recorder.logged, _ROUND_ONE_BODY)
    assert [row["response_cost"] for row in recorder.failed] == [0.0]


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_every_paid_round_with_one_fixed_margin_when_the_third_round_is_rejected(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock,
        monkeypatch,
        (
            _streamed(_ROUND_ONE_BODY),
            _streamed(_ROUND_ONE_BODY),
            httpx.Response(400, json={"error": {"message": "Your input exceeds the context window of this model."}}),
        ),
    )
    recorder: Final = _RoundRecorder()

    await _stream_auto_execute(_rounds_logging_obj(recorder, "mcp-stream-third-round-rejected"))
    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_row_bills_round_one_twice_with_one_fixed_margin(recorder.logged)


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_every_finished_round_when_the_client_disconnects_during_a_later_tool_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_ONE_BODY)))
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-stream-disconnect-in-tool-call")
    echo_then_disconnect: Final = AsyncMock(
        side_effect=[[{"tool_call_id": "call_echo", "result": "PING", "name": "echo"}], asyncio.CancelledError()]
    )

    with (
        _echo_tool_served("PING"),
        patch.object(LiteLLM_Proxy_MCP_Handler, "execute_tool_calls", echo_then_disconnect),
        pytest.raises(asyncio.CancelledError),
    ):
        await _drain(await _open_stream_auto_execute(logging_obj))
    ProxyLogging.fire_deferred_stream_logging({"litellm_logging_obj": logging_obj})
    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_row_bills_round_one_twice_with_one_fixed_margin(recorder.logged)


@pytest.mark.asyncio
async def test_stream_auto_execute_answered_without_a_tool_call_logs_its_one_round(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_TWO_BODY),))
    recorder: Final = _RoundRecorder()

    events: Final = await _stream_auto_execute(_rounds_logging_obj(recorder, "mcp-stream-one-round"))
    await _settle(lambda: len(recorder.logged) > 0)

    assert _output_types(_final_response(events).output) == ("message",)
    _assert_one_row_bills(recorder.logged, _ROUND_TWO_BODY)


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_both_rounds_with_one_fixed_margin_when_the_final_round_is_incomplete(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    truncated: Final = _streamed(
        _ROUND_TWO_BODY, status="incomplete", incomplete_details={"reason": "max_output_tokens"}
    )
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), truncated))
    recorder: Final = _RoundRecorder()

    events: Final = await _stream_auto_execute(_rounds_logging_obj(recorder, "mcp-stream-final-round-incomplete"))
    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_stream_row_bills_both_rounds_once(recorder.logged, _final_response(events, "response.incomplete"))


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_a_failed_follow_up_on_its_own_failure_row_with_no_second_fixed_margin(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": _FIXED_MARGIN}})
    server_error: Final = {"code": "server_error", "message": "The server had an error processing your request."}
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_TWO_BODY, status="failed", error=server_error)),
    )
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-stream-follow-up-failed")

    with pytest.raises(litellm.ServiceUnavailableError):
        await _stream_auto_execute(logging_obj)
    await _settle(lambda: len(recorder.logged) > 0 and len(recorder.failed) > 0)

    _assert_one_row_bills(recorder.logged, _ROUND_ONE_BODY)
    assert len(recorder.failed) == 1
    assert (recorder.failed[0]["prompt_tokens"], recorder.failed[0]["completion_tokens"]) == (149, 5)
    assert recorder.failed[0]["response_cost"] == pytest.approx(_round_cost(_ROUND_TWO_BODY) - _FIXED_MARGIN)
    _assert_breakdown_agrees_with_the_billed_cost(recorder.failed[0])
    failure_row: Final = _request_failure_row(logging_obj, recorder.logged[0])
    _assert_billed_tokens(failure_row, (149, 5))
    assert failure_row["response_cost"] == pytest.approx(_round_cost(_ROUND_TWO_BODY) - _FIXED_MARGIN)


_XAI_MODEL: Final = "xai/grok-4.7"
_XAI_RESPONSES_URL: Final = r"https://api\.x\.ai/v1/responses"


def _xai_round(body: Mapping[str, object], cost_in_usd_ticks: int) -> Mapping[str, object]:
    usage: Final = cast(Mapping[str, object], body["usage"])
    stated: Final = {**usage, "cost_in_usd_ticks": cost_in_usd_ticks, "context_details": {"cached_tokens": 3}}
    return MappingProxyType({**body, "model": "grok-4.7", "usage": stated})


def _xai_round_cost(body: Mapping[str, object]) -> float:
    usage: Final = cast(Mapping[str, object], body["usage"])
    stated: Final = {**usage, "cost": xai_reported_cost_in_usd(usage["cost_in_usd_ticks"])}
    return litellm.completion_cost(
        completion_response=ResponsesAPIResponse.model_validate({**body, "usage": stated}),
        model=_XAI_MODEL,
        custom_llm_provider="xai",
        call_type="aresponses",
    )


@pytest.mark.asyncio
async def test_xai_stream_auto_execute_bills_each_round_at_its_own_price_with_one_fixed_margin(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"xai": {"fixed_amount": _FIXED_MARGIN}})
    rounds: Final = (_xai_round(_ROUND_ONE_BODY, 12_400_000), _xai_round(_ROUND_TWO_BODY, 8_800_000))
    _serve_responses(respx_mock, monkeypatch, tuple(_streamed(body) for body in rounds), _XAI_RESPONSES_URL)
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        events: Final = await _drain(
            await litellm.aresponses(
                model=_XAI_MODEL,
                input="Call echo with PING",
                tools=[dict(_ECHO_MCP_TOOL)],
                api_key="xai-mcp-rounds",
                stream=True,
                litellm_logging_obj=_rounds_logging_obj(recorder, "mcp-xai-stream-rounds"),
            )
        )
    await _settle(lambda: len(recorder.logged) > 0)

    charged_once: Final = _xai_round_cost(rounds[0]) + _xai_round_cost(rounds[1]) - _FIXED_MARGIN
    client: Final = _final_response(events)
    assert len(recorder.logged) == 1
    assert recorder.logged[0]["response_cost"] == pytest.approx(charged_once)
    assert recorder.logged[0]["cost_breakdown"]["total_cost"] == pytest.approx(charged_once)
    assert client.usage is not None
    assert client.usage.cost == pytest.approx(charged_once)
    assert json.loads(events[-1].model_dump_json())["response"]["usage"]["context_details"] == {"cached_tokens": 6}


_UNPRICED_MODEL: Final = "litellm-unpriced-mcp-rounds"


@pytest.mark.asyncio
async def test_stream_auto_execute_on_an_unpriced_model_logs_both_rounds_as_a_failed_cost_lookup_not_a_free_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    rounds: Final = tuple(
        MappingProxyType({**body, "model": _UNPRICED_MODEL}) for body in (_ROUND_ONE_BODY, _ROUND_TWO_BODY)
    )
    _serve_rounds(respx_mock, monkeypatch, rounds)
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        events: Final = await _drain(
            await litellm.aresponses(
                model=f"openai/{_UNPRICED_MODEL}",
                input="Call echo with PING",
                tools=[dict(_ECHO_MCP_TOOL)],
                api_key="sk-mcp-rounds",
                api_base="https://mcp-rounds.example.com/v1",
                stream=True,
                litellm_logging_obj=_rounds_logging_obj(recorder, "mcp-stream-unpriced-rounds"),
            )
        )
    await _settle(lambda: len(recorder.logged) > 0)

    client: Final = _final_response(events)
    assert len(recorder.logged) == 1
    row: Final = recorder.logged[0]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (71 + 149, 19 + 5)
    assert row["response_cost_failure_debug_info"] is not None
    assert client.usage is not None
    assert client.usage.cost is None


async def _stop_reading_once_the_follow_up_answers(logging_obj: Logging) -> None:
    stream: Final = await _open_stream_auto_execute(logging_obj)
    async for event in cast(Any, stream):
        if _event_type(event) == "response.output_item.added" and _output_types((event.item,)) == ("message",):
            return


@pytest.mark.asyncio
async def test_stream_auto_execute_bills_the_finished_first_round_when_the_caller_stops_reading_the_follow_up(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_TWO_BODY)))
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        await _stop_reading_once_the_follow_up_answers(_rounds_logging_obj(recorder, "mcp-stream-caller-stops"))
    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_row_bills(recorder.logged, _ROUND_ONE_BODY)


async def _stream_read_until_the_follow_up_answers(logging_obj: Logging) -> object:
    stream: Final = await _open_stream_auto_execute(logging_obj)
    async for event in cast(Any, stream):
        if _event_type(event) == "response.output_item.added" and _output_types((event.item,)) == ("message",):
            return stream
    pytest.fail("the follow-up round never answered")


_STREAM_BILLING_NAMES: Final = ("StreamedRoundBilling", "dispatch_success_handlers", "StreamingIterator")


def _names_stream_billing(text: str) -> bool:
    return any(name in text for name in _STREAM_BILLING_NAMES)


class _UnraisableRecorder:
    def __init__(self) -> None:
        self.raised: tuple[str, ...] = ()

    def __call__(self, unraisable: "sys.UnraisableHookArgs") -> None:
        self.raised = (*self.raised, f"{unraisable.object!r}: {unraisable.exc_value!r}")


def test_stream_auto_execute_dropped_after_its_event_loop_closed_discards_its_parked_log_cleanly(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_TWO_BODY)))
    recorder: Final = _RoundRecorder()
    unraisable: Final = _UnraisableRecorder()
    gc.collect()
    monkeypatch.setattr(sys, "unraisablehook", unraisable)

    with _echo_tool_served("PING"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(_stream_read_until_the_follow_up_answers(_rounds_logging_obj(recorder, "mcp-stream-loop-closed")))
        gc.collect()

    assert recorder.logged == ()
    assert [raised for raised in unraisable.raised if _names_stream_billing(raised)] == []
    assert [str(warning.message) for warning in caught if _names_stream_billing(str(warning.message))] == []


_BRIDGED_MODEL: Final = "anthropic/claude-haiku-5-5"
_ANTHROPIC_MESSAGES_URL: Final = r"https://api\.anthropic\.com/v1/messages"


def _bridged_round(content: bytes) -> httpx.Response:
    return httpx.Response(200, content=content, headers=SSE_HEADERS)


def _bridged_echo_call_round(cache_usage: Mapping[str, object] = MappingProxyType({})) -> httpx.Response:
    return _bridged_round(
        _anthropic_round(
            "msg_echo_call",
            {"type": "tool_use", "id": "call_echo", "name": "echo", "input": {}},
            {"type": "input_json_delta", "partial_json": json.dumps({"value": "PING"})},
            "tool_use",
            model="claude-haiku-5-5",
            usage=(533, 40),
            cache_usage=cache_usage,
        )
    )


def _bridged_echo_answer_round(
    stop: str = "end_turn", cache_usage: Mapping[str, object] = MappingProxyType({})
) -> httpx.Response:
    return _bridged_round(
        _anthropic_round(
            "msg_echo_answer",
            {"type": "text", "text": ""},
            {"type": "text_delta", "text": "PING"},
            stop,
            model="claude-haiku-5-5",
            usage=(470, 4),
            cache_usage=cache_usage,
        )
    )


def _anthropic_prompt_cache_usage(written: int, read: int) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "cache_creation_input_tokens": written,
            "cache_read_input_tokens": read,
            "cache_creation": {"ephemeral_5m_input_tokens": written, "ephemeral_1h_input_tokens": 0},
            "inference_geo": "us",
            "speed": "standard",
        }
    )


class _CutAfterTheMessageStart(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _sse((_anthropic_message_start("msg_echo_answer", "claude-haiku-5-5", 470),))
        raise httpx.ReadError("connection reset mid-stream")


def _bridged_round_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return litellm.completion_cost(
        completion_response=ModelResponse(
            model="claude-haiku-5-5",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        ),
        model=_BRIDGED_MODEL,
    )


async def _open_bridged_stream_auto_execute(logging_obj: Logging) -> object:
    return await litellm.aresponses(
        model=_BRIDGED_MODEL,
        input="Call echo with PING",
        tools=[dict(_ECHO_MCP_TOOL)],
        api_key="sk-ant-mcp-rounds",
        stream=True,
        litellm_logging_obj=logging_obj,
    )


def _assert_logged_as_the_final_chat_answer_with_both_rounds_usage(recorder: _RoundRecorder, answer: str) -> None:
    assert len(recorder.results) == 1
    logged: Final = recorder.results[0]
    assert isinstance(logged, ModelResponse)
    assert logged.choices[0].message.content == answer
    usage: Final[object] = getattr(logged, "usage", None)
    assert isinstance(usage, Usage)
    assert (usage.prompt_tokens, usage.completion_tokens) == (533 + 470, 40 + 4)


def _assert_one_bridged_row_bills_both_rounds_once(recorder: _RoundRecorder, client: ResponsesAPIResponse) -> None:
    charged_once: Final = _bridged_round_cost(533, 40) + _bridged_round_cost(470, 4) - _FIXED_MARGIN
    assert len(recorder.logged) == 1
    row: Final = recorder.logged[0]
    _assert_logged_as_the_final_chat_answer_with_both_rounds_usage(recorder, "PING")
    assert row["id"] == client.id
    assert "message" in _output_types(client.output)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (533 + 470, 40 + 4)
    assert row["cache_hit"] is False
    assert row["response_cost"] == pytest.approx(charged_once)
    assert row["cost_breakdown"]["total_cost"] == pytest.approx(charged_once)
    assert client.usage is not None
    assert (client.usage.input_tokens, client.usage.output_tokens) == (533 + 470, 40 + 4)
    assert client.usage.cost == pytest.approx(charged_once)


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_logs_the_final_response_billed_for_both_rounds_with_one_fixed_margin(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock, monkeypatch, (_bridged_echo_call_round(), _bridged_echo_answer_round()), _ANTHROPIC_MESSAGES_URL
    )
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        events: Final = await _drain(
            await _open_bridged_stream_auto_execute(_rounds_logging_obj(recorder, "mcp-bridged-stream-rounds"))
        )
    await _settle(lambda: len(recorder.logged) > 0)

    assert _output_types(_final_response(events).output) == ("mcp_call", "message")
    _assert_one_bridged_row_bills_both_rounds_once(recorder, _final_response(events))


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_logs_the_provider_usage_of_both_rounds_summed(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock,
        monkeypatch,
        (
            _bridged_echo_call_round(_anthropic_prompt_cache_usage(written=10, read=5)),
            _bridged_echo_answer_round(cache_usage=_anthropic_prompt_cache_usage(written=3, read=7)),
        ),
        _ANTHROPIC_MESSAGES_URL,
    )
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        await _drain(
            await _open_bridged_stream_auto_execute(_rounds_logging_obj(recorder, "mcp-bridged-provider-usage"))
        )
    await _settle(lambda: len(recorder.logged) > 0)

    assert len(recorder.logged) == 1
    row: Final = recorder.logged[0]
    usage: Final = row["metadata"]["usage_object"]
    assert "choices" in row["response"]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (533 + 10 + 5 + 470 + 3 + 7, 40 + 4)
    assert (usage["cache_creation_input_tokens"], usage["cache_read_input_tokens"]) == (10 + 3, 5 + 7)
    assert usage["prompt_tokens_details"]["cache_creation_token_details"]["ephemeral_5m_input_tokens"] == 10 + 3
    assert (usage["inference_geo"], usage["speed"]) == ("us", "standard")
    assert usage["cost"] == pytest.approx(row["response_cost"])


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_bills_both_rounds_with_one_fixed_margin_when_the_final_round_is_incomplete(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_bridged_echo_call_round(), _bridged_echo_answer_round("max_tokens")),
        _ANTHROPIC_MESSAGES_URL,
    )
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        events: Final = await _drain(
            await _open_bridged_stream_auto_execute(_rounds_logging_obj(recorder, "mcp-bridged-stream-incomplete"))
        )
    await _settle(lambda: len(recorder.logged) > 0)

    _assert_one_bridged_row_bills_both_rounds_once(recorder, _final_response(events, "response.incomplete"))


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_deferred_for_a_post_call_guardrail_logs_both_rounds_once_on_release(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock, monkeypatch, (_bridged_echo_call_round(), _bridged_echo_answer_round()), _ANTHROPIC_MESSAGES_URL
    )
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-bridged-stream-rounds-deferred")

    with _echo_tool_served("PING"):
        stream: Final = await _open_bridged_stream_auto_execute(logging_obj)
        ProxyBaseLLMRequestProcessing(data={})._arm_deferred_stream_dispatch(
            response=stream, route_type="aresponses", user_api_key_dict=MagicMock(), logging_obj=logging_obj
        )
        events: Final = await _drain(stream)
    await asyncio.sleep(0.2)
    logged_before_release: Final = recorder.logged
    ProxyLogging.fire_deferred_stream_logging({"litellm_logging_obj": logging_obj})
    await _settle(lambda: len(recorder.logged) > 0)

    assert logged_before_release == ()
    _assert_one_bridged_row_bills_both_rounds_once(recorder, _final_response(events))


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_blocked_by_a_post_call_guardrail_bills_both_rounds_on_the_failure_row(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock, monkeypatch, (_bridged_echo_call_round(), _bridged_echo_answer_round()), _ANTHROPIC_MESSAGES_URL
    )
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-bridged-stream-rounds-blocked")
    blocked: Final = GuardrailRaisedException(guardrail_name="output-policy", message="blocked")

    with _echo_tool_served("PING"):
        stream: Final = await _open_bridged_stream_auto_execute(logging_obj)
        ProxyBaseLLMRequestProcessing(data={})._arm_deferred_stream_dispatch(
            response=stream, route_type="aresponses", user_api_key_dict=MagicMock(), logging_obj=logging_obj
        )
        await _drain(stream)
    ProxyLogging._discard_deferred_stream_logging_for_failure({"litellm_logging_obj": logging_obj}, blocked)
    ProxyLogging.fire_deferred_stream_logging({"litellm_logging_obj": logging_obj})
    await logging_obj.async_failure_handler(blocked, "blocked by output-policy")
    await _settle(lambda: len(recorder.failed) > 0)

    charged_once: Final = _bridged_round_cost(533, 40) + _bridged_round_cost(470, 4) - _FIXED_MARGIN
    assert recorder.logged == ()
    assert len(recorder.failed) == 1
    assert (recorder.failed[0]["prompt_tokens"], recorder.failed[0]["completion_tokens"]) == (533 + 470, 40 + 4)
    assert recorder.failed[0]["response_cost"] == pytest.approx(charged_once)
    assert recorder.failed[0]["cost_breakdown"]["total_cost"] == pytest.approx(charged_once)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ("end_turn", "max_tokens"), ids=("completed", "incomplete"))
async def test_bridged_stream_auto_execute_bills_both_rounds_when_the_caller_stops_reading_after_the_answer_text(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stop: str
):
    _serve_responses(
        respx_mock, monkeypatch, (_bridged_echo_call_round(), _bridged_echo_answer_round(stop)), _ANTHROPIC_MESSAGES_URL
    )
    recorder: Final = _RoundRecorder()

    async def stop_reading_once_the_answer_text_is_done() -> None:
        stream: Final = await _open_bridged_stream_auto_execute(
            _rounds_logging_obj(recorder, "mcp-bridged-stream-caller-stops")
        )
        async for event in cast(Any, stream):
            if _event_type(event) == "response.output_text.done":
                return

    with _echo_tool_served("PING"):
        await stop_reading_once_the_answer_text_is_done()
    await _settle(lambda: len(recorder.logged) > 0)

    assert len(recorder.logged) == 1
    _assert_logged_as_the_final_chat_answer_with_both_rounds_usage(recorder, "PING")
    assert (recorder.logged[0]["prompt_tokens"], recorder.logged[0]["completion_tokens"]) == (533 + 470, 40 + 4)
    assert recorder.logged[0]["response_cost"] == pytest.approx(
        _bridged_round_cost(533, 40) + _bridged_round_cost(470, 4)
    )


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_bills_a_broken_follow_up_on_its_own_failure_row_with_no_second_fixed_margin(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": _FIXED_MARGIN}})
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_bridged_echo_call_round(), httpx.Response(200, stream=_CutAfterTheMessageStart(), headers=SSE_HEADERS)),
        _ANTHROPIC_MESSAGES_URL,
    )
    recorder: Final = _RoundRecorder()
    logging_obj: Final = _rounds_logging_obj(recorder, "mcp-bridged-stream-breaks")

    with _echo_tool_served("PING"), pytest.raises(MidStreamFallbackError):
        await _drain(await _open_bridged_stream_auto_execute(logging_obj))
    await _settle(lambda: len(recorder.logged) > 0 and len(recorder.failed) > 0)

    assert len(recorder.logged) == 1
    assert (recorder.logged[0]["prompt_tokens"], recorder.logged[0]["completion_tokens"]) == (533, 40)
    assert recorder.logged[0]["response_cost"] == pytest.approx(_bridged_round_cost(533, 40))
    assert len(recorder.failed) == 1
    assert (recorder.failed[0]["prompt_tokens"], recorder.failed[0]["completion_tokens"]) == (470, 0)
    assert recorder.failed[0]["response_cost"] == pytest.approx(_bridged_round_cost(470, 0) - _FIXED_MARGIN)
    _assert_breakdown_agrees_with_the_billed_cost(recorder.failed[0])
    failure_row: Final = _request_failure_row(logging_obj, recorder.logged[0])
    _assert_billed_tokens(failure_row, (470, 0))
    assert failure_row["response_cost"] == pytest.approx(_bridged_round_cost(470, 0) - _FIXED_MARGIN)


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_bills_a_broken_follow_up_its_own_fixed_margin_after_a_cached_first_round(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": _FIXED_MARGIN}})
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    _serve_responses(
        respx_mock,
        monkeypatch,
        (
            _bridged_echo_call_round(),
            _bridged_echo_answer_round(),
            httpx.Response(200, stream=_CutAfterTheMessageStart(), headers=SSE_HEADERS),
        ),
        _ANTHROPIC_MESSAGES_URL,
    )
    warm: Final = _RoundRecorder()
    with _echo_tool_served("PING"):
        await _drain(
            await _open_bridged_stream_auto_execute(_rounds_logging_obj(warm, "mcp-bridged-cache-then-break-warm"))
        )
    await _settle(lambda: len(warm.logged) > 0)

    recorder: Final = _RoundRecorder()
    with _echo_tool_served("PONG"), pytest.raises(MidStreamFallbackError):
        await _drain(
            await _open_bridged_stream_auto_execute(_rounds_logging_obj(recorder, "mcp-bridged-cache-then-break"))
        )
    await _settle(lambda: len(recorder.logged) > 0 and len(recorder.failed) > 0)

    assert [row["response_cost"] for row in recorder.logged] == [0.0]
    assert len(recorder.failed) == 1
    assert recorder.failed[0]["response_cost"] == pytest.approx(_bridged_round_cost(470, 0))


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_answered_without_a_tool_call_logs_its_one_round(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_responses(respx_mock, monkeypatch, (_bridged_echo_answer_round(),), _ANTHROPIC_MESSAGES_URL)
    recorder: Final = _RoundRecorder()

    with _echo_tool_served("PING"):
        await _drain(await _open_bridged_stream_auto_execute(_rounds_logging_obj(recorder, "mcp-bridged-one-round")))
    await _settle(lambda: len(recorder.logged) > 0)

    assert len(recorder.logged) == 1
    assert "choices" in recorder.logged[0]["response"]
    assert (recorder.logged[0]["prompt_tokens"], recorder.logged[0]["completion_tokens"]) == (470, 4)
    assert recorder.logged[0]["response_cost"] == pytest.approx(_bridged_round_cost(470, 4))


@pytest.mark.asyncio
async def test_bridged_stream_auto_execute_bills_nothing_for_a_round_served_from_the_response_cache(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    provider: Final = _serve_responses(
        respx_mock,
        monkeypatch,
        (_bridged_echo_call_round(), _bridged_echo_answer_round(), _bridged_echo_answer_round()),
        _ANTHROPIC_MESSAGES_URL,
    )
    warm: Final = _RoundRecorder()
    with _echo_tool_served("PING"):
        await _drain(await _open_bridged_stream_auto_execute(_rounds_logging_obj(warm, "mcp-bridged-cache-warm")))
    await _settle(lambda: len(warm.logged) > 0)

    first_cached: Final = _RoundRecorder()
    with _echo_tool_served("PONG"):
        events: Final = await _drain(
            await _open_bridged_stream_auto_execute(_rounds_logging_obj(first_cached, "mcp-bridged-cache-first-round"))
        )
    await _settle(lambda: len(first_cached.logged) > 0)

    both_cached: Final = _RoundRecorder()
    with _echo_tool_served("PING"):
        await _drain(
            await _open_bridged_stream_auto_execute(_rounds_logging_obj(both_cached, "mcp-bridged-cache-both-rounds"))
        )
    await _settle(lambda: len(both_cached.logged) > 0)

    client: Final = _final_response(events)
    assert provider.call_count == 3
    assert len(first_cached.logged) == 1
    assert (first_cached.logged[0]["prompt_tokens"], first_cached.logged[0]["completion_tokens"]) == (533 + 470, 40 + 4)
    assert first_cached.logged[0]["response_cost"] == pytest.approx(_bridged_round_cost(470, 4))
    assert first_cached.logged[0]["cache_hit"] is False
    assert [(row["response_cost"], row["cache_hit"]) for row in both_cached.logged] == [(0.0, True)]
    assert client.usage is not None
    assert client.usage.cost == pytest.approx(_bridged_round_cost(470, 4))


_DEEPWIKI_URL: Final = "https://mcp.deepwiki.com/mcp"
_DEEPWIKI_TOOL: Final = MappingProxyType(
    {"type": "mcp", "server_label": "deepwiki", "server_url": _DEEPWIKI_URL, "require_approval": "never"}
)
_ASK_TOOL: Final = MCPTool(
    name="ask_question",
    description="Ask a question about a repository",
    inputSchema={"type": "object", "properties": {"question": {"type": "string"}}},
)
_ASK_ARGUMENTS: Final = {"question": "What is LiteLLM?"}
_ASK_OUTPUT: Final = "LiteLLM is an AI gateway"
_ASK_ANSWER: Final = "DeepWiki says LiteLLM is an AI gateway"
_DEEPSEEK_MODEL: Final = "deepseek/deepseek-v4-flash"
_DEEPSEEK_URL: Final = r"https://gateway-mcp\.example\.com/v1/chat/completions"
_BEDROCK_MODEL: Final = "bedrock/converse/us.anthropic.claude-haiku-5-5"
_BEDROCK_URL: Final = r"https://bedrock-runtime\.us-west-2\.amazonaws\.com/model/.*/converse"
_ROUTE_KWARGS: Final = MappingProxyType(
    {
        _DEEPSEEK_MODEL: MappingProxyType(
            {"api_key": "sk-gateway-mcp", "api_base": "https://gateway-mcp.example.com/v1"}
        ),
        _BEDROCK_MODEL: MappingProxyType(
            {"aws_access_key_id": "AKIDGATEWAYMCP", "aws_secret_access_key": "secret", "aws_region_name": "us-west-2"}
        ),
    }
)
_ROUTE_URLS: Final = MappingProxyType({_DEEPSEEK_MODEL: _DEEPSEEK_URL, _BEDROCK_MODEL: _BEDROCK_URL})


class _ScriptedMCPServer:
    def __init__(self, tool_failed: bool = False) -> None:
        self.opened: tuple[tuple[DirectMCPServer, tuple[str, ...]], ...] = ()
        self.calls: tuple[CallToolRequestParams, ...] = ()
        self._tool_failed: Final = tool_failed

    def client_for(self, server: DirectMCPServer, allowed_hosts: Sequence[str]) -> "_ScriptedMCPServer":
        self.opened = (*self.opened, (server, tuple(allowed_hosts)))
        return self

    async def list_tools(self, raise_on_error: bool = False) -> Sequence[MCPTool]:
        return (_ASK_TOOL,)

    async def call_tool(self, call_tool_request_params: CallToolRequestParams) -> CallToolResult:
        self.calls = (*self.calls, call_tool_request_params)
        return CallToolResult(content=[TextContent(type="text", text=_ASK_OUTPUT)], isError=self._tool_failed)


def _chat_round(message: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-gateway-mcp",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "message": dict(message),
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
        },
    )


def _converse_round(content: Mapping[str, object], stop_reason: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "output": {"message": {"role": "assistant", "content": [dict(content)]}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 40, "outputTokens": 8, "totalTokens": 48},
            "metrics": {"latencyMs": 10},
        },
    )


def _tool_call_round(model: str, tool_name: str) -> httpx.Response:
    if model == _BEDROCK_MODEL:
        return _converse_round(
            {"toolUse": {"toolUseId": "call_ask", "name": tool_name, "input": _ASK_ARGUMENTS}}, "tool_use"
        )
    return _chat_round(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_ask",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(_ASK_ARGUMENTS)},
                }
            ],
        }
    )


def _answer_round(model: str) -> httpx.Response:
    if model == _BEDROCK_MODEL:
        return _converse_round({"text": _ASK_ANSWER}, "end_turn")
    return _chat_round({"role": "assistant", "content": _ASK_ANSWER})


def _chat_chunk(delta: Mapping[str, object], finish_reason: str | None) -> Mapping[str, object]:
    return {
        "id": "chatcmpl-gateway-mcp",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }


def _chat_stream(*chunks: Mapping[str, object]) -> httpx.Response:
    body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers=SSE_HEADERS)


def _streamed_tool_call_round(tool_name: str = "deepwiki-ask_question") -> httpx.Response:
    tool_call: Final = {
        "index": 0,
        "id": "call_ask",
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(_ASK_ARGUMENTS)},
    }
    return _chat_stream(
        _chat_chunk({"role": "assistant", "tool_calls": [tool_call]}, None), _chat_chunk({}, "tool_calls")
    )


def _streamed_answer_round() -> httpx.Response:
    return _chat_stream(_chat_chunk({"role": "assistant", "content": _ASK_ANSWER}, None), _chat_chunk({}, "stop"))


def _gateway_runs_hosted_mcp(monkeypatch: pytest.MonkeyPatch, allowed_hosts: Sequence[str]) -> None:
    monkeypatch.setattr(litellm, "gateway_runs_hosted_mcp", True)
    monkeypatch.setattr(litellm, "gateway_hosted_mcp_allowed_hosts", list(allowed_hosts))


async def _gateway_hosted_mcp_request(
    model: str,
    server: _ScriptedMCPServer,
    tool: Mapping[str, object] = _DEEPWIKI_TOOL,
    *,
    beside: Sequence[Mapping[str, object]] = (),
    drop_params: bool | None = None,
    stream: bool = False,
    caller: object = None,
    tool_choice: object = None,
) -> object:
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        return await litellm.aresponses(
            model=model,
            input="Ask DeepWiki what LiteLLM is",
            tools=[dict(tool), *map(dict, beside)],
            drop_params=drop_params,
            stream=stream,
            litellm_metadata={"user_api_key_auth": caller},
            tool_choice=tool_choice,
            **_ROUTE_KWARGS[model],
        )


def _sent_body(route: respx.Route, index: int) -> Mapping[str, object]:
    return json.loads(route.calls[index].request.content)


def _sent_tool_names(body: Mapping[str, object]) -> tuple[str, ...]:
    tools: Final = body.get("tools") or body.get("toolConfig", {}).get("tools", [])
    return tuple((tool.get("function") or tool.get("toolSpec") or {}).get("name", tool.get("type")) for tool in tools)


def _item(item: object) -> Mapping[str, object]:
    return item if isinstance(item, Mapping) else item.model_dump()


def _assert_openai_mcp_output(output: Sequence[object], status: str = "completed") -> None:
    items: Final = [_item(item) for item in output]
    assert [item["type"] for item in items] == ["mcp_list_tools", "mcp_call", "message"]
    assert {key: items[0][key] for key in ("server_label", "tools", "error")} == {
        "server_label": "deepwiki",
        "tools": [
            {
                "name": "ask_question",
                "description": _ASK_TOOL.description,
                "input_schema": _ASK_TOOL.input_schema,
                "annotations": {"read_only": False},
            }
        ],
        "error": None,
    }
    call: Final = items[1]
    assert {key: call[key] for key in ("server_label", "name", "arguments", "status")} == {
        "server_label": "deepwiki",
        "name": "ask_question",
        "arguments": json.dumps(_ASK_ARGUMENTS),
        "status": status,
    }
    assert (call["output"], call["error"]) == ((_ASK_OUTPUT, None) if status == "completed" else (None, _ASK_OUTPUT))
    assert str(call["id"]).startswith("mcp_")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [_BEDROCK_MODEL, _DEEPSEEK_MODEL])
@pytest.mark.parametrize(
    "allowed_hosts", [(), ("mcp.example.com",)], ids=["nothing_allowlisted", "another_host_allowlisted"]
)
@respx.mock
async def test_a_hosted_mcp_url_neither_registered_nor_allowlisted_opens_no_connection_and_keeps_the_400(
    monkeypatch: pytest.MonkeyPatch, model: str, allowed_hosts: tuple[str, ...]
):
    _gateway_runs_hosted_mcp(monkeypatch, allowed_hosts)
    provider: Final = _serve_responses(respx.mock, monkeypatch, (), _ROUTE_URLS[model])
    server: Final = _ScriptedMCPServer()

    with pytest.raises(litellm.BadRequestError) as refused:
        await _gateway_hosted_mcp_request(model, server)

    assert "cannot run the hosted MCP tool `deepwiki`" in str(refused.value)
    assert (server.opened, provider.call_count) == ((), 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [_BEDROCK_MODEL, _DEEPSEEK_MODEL])
@respx.mock
async def test_a_hosted_mcp_url_neither_registered_nor_allowlisted_is_dropped_under_drop_params_without_a_connection(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.example.com",))
    provider: Final = _serve_responses(respx.mock, monkeypatch, (_answer_round(model),), _ROUTE_URLS[model])
    server: Final = _ScriptedMCPServer()

    response: Final = await _gateway_hosted_mcp_request(model, server, drop_params=True)

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_types(response.output) == ("message",)
    assert _sent_tool_names(_sent_body(provider, 0)) == ()
    assert server.opened == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [_BEDROCK_MODEL, _DEEPSEEK_MODEL])
@pytest.mark.parametrize("allowed_hosts", [(), ("mcp.deepwiki.com",)], ids=["not_allowlisted", "allowlisted"])
@respx.mock
async def test_a_hosted_mcp_tool_on_a_deployment_that_drops_tools_answers_without_tools_or_a_connection(
    monkeypatch: pytest.MonkeyPatch, model: str, allowed_hosts: tuple[str, ...]
):
    _gateway_runs_hosted_mcp(monkeypatch, allowed_hosts)
    provider: Final = _serve_responses(respx.mock, monkeypatch, (_answer_round(model),), _ROUTE_URLS[model])
    server: Final = _ScriptedMCPServer()

    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        response: Final = await litellm.aresponses(
            model=model,
            input="Ask DeepWiki what LiteLLM is",
            tools=[dict(_DEEPWIKI_TOOL)],
            additional_drop_params=["tools"],
            **_ROUTE_KWARGS[model],
        )

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_types(response.output) == ("message",)
    assert _sent_tool_names(_sent_body(provider, 0)) == ()
    assert server.opened == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [_BEDROCK_MODEL, _DEEPSEEK_MODEL])
@pytest.mark.parametrize("tool_failed", [False, True], ids=["tool_succeeds", "tool_fails"])
@respx.mock
async def test_an_allowlisted_hosted_mcp_url_runs_on_the_gateway_and_answers_in_openai_mcp_items(
    monkeypatch: pytest.MonkeyPatch, model: str, tool_failed: bool
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        (_tool_call_round(model, "deepwiki-ask_question"), _answer_round(model)),
        _ROUTE_URLS[model],
    )
    server: Final = _ScriptedMCPServer(tool_failed=tool_failed)
    tool: Final = {**_DEEPWIKI_TOOL, "headers": {"X-Caller": "a"}, "authorization": "caller-token"}

    response: Final = await _gateway_hosted_mcp_request(model, server, tool)

    assert isinstance(response, ResponsesAPIResponse)
    _assert_openai_mcp_output(response.output, "failed" if tool_failed else "completed")
    assert _item(response.output[2])["content"][0]["text"] == _ASK_ANSWER
    assert [(opened.headers, hosts) for opened, hosts in server.opened] == [
        ({"X-Caller": "a", "Authorization": "Bearer caller-token"}, ("mcp.deepwiki.com",))
    ]
    assert [(call.name, call.arguments) for call in server.calls] == [("ask_question", _ASK_ARGUMENTS)]
    assert _sent_tool_names(_sent_body(provider, 0)) == ("deepwiki-ask_question",)
    assert _ASK_OUTPUT in json.dumps(_sent_body(provider, 1))


_LOOKUP_FUNCTION: Final = MappingProxyType(
    {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [_BEDROCK_MODEL, _DEEPSEEK_MODEL])
@pytest.mark.parametrize(
    ("allowed", "runs"),
    [({"type": "mcp", "server_label": "deepwiki"}, True), ({"type": "function", "name": "lookup"}, False)],
    ids=["choice_allows_the_server", "choice_excludes_the_server"],
)
@respx.mock
async def test_an_allowed_tools_choice_decides_whether_the_gateway_runs_an_allowlisted_url_call(
    monkeypatch: pytest.MonkeyPatch, model: str, allowed: Mapping[str, object], runs: bool
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        (_tool_call_round(model, "deepwiki-ask_question"), _answer_round(model)),
        _ROUTE_URLS[model],
    )
    server: Final = _ScriptedMCPServer()

    response: Final = await _gateway_hosted_mcp_request(
        model,
        server,
        beside=(_LOOKUP_FUNCTION,),
        drop_params=True,
        tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [allowed]},
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert [call.name for call in server.calls] == (["ask_question"] if runs else [])
    assert provider.call_count == (2 if runs else 1)
    assert _output_types(response.output)[0] == "mcp_list_tools"
    assert _output_types(response.output)[-1] == ("message" if runs else "function_call")
    assert ("mcp_call" in _output_types(response.output)) == runs


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True], ids=["allowlisted_url", "registered_url"])
@pytest.mark.parametrize("drop_params", [None, True], ids=["strict", "drop_params"])
@respx.mock
async def test_an_mcp_tool_choice_forcing_a_server_the_gateway_runs_gets_the_gateway_400_before_any_listing_or_call(
    monkeypatch: pytest.MonkeyPatch, registered: bool, drop_params: bool | None
):
    _gateway_runs_hosted_mcp(monkeypatch, () if registered else ("mcp.deepwiki.com",))
    mcp_client: Final = _deepwiki_registered(monkeypatch) if registered else None
    provider: Final = _serve_responses(respx.mock, monkeypatch, (), _BEDROCK_URL)
    server: Final = _ScriptedMCPServer()

    with pytest.raises(litellm.BadRequestError) as refused:
        await _gateway_hosted_mcp_request(
            _BEDROCK_MODEL,
            server,
            drop_params=drop_params,
            caller=_key_holding("srv-deepwiki"),
            tool_choice={"type": "mcp", "server_label": "deepwiki"},
        )

    assert "tool_choice type 'mcp' cannot force `deepwiki`, which runs on the LiteLLM MCP gateway" in str(refused.value)
    assert (server.opened, provider.call_count) == ((), 0)
    assert mcp_client is None or mcp_client.list_tools_result.await_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_a_streamed_call_an_allowed_tools_choice_excludes_comes_back_as_a_function_call(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    provider: Final = _serve_responses(
        respx.mock, monkeypatch, (_streamed_tool_call_round(), _streamed_answer_round()), _DEEPSEEK_URL
    )
    server: Final = _ScriptedMCPServer()

    stream: Final = await _gateway_hosted_mcp_request(
        _DEEPSEEK_MODEL,
        server,
        beside=(_LOOKUP_FUNCTION,),
        drop_params=True,
        stream=True,
        tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "lookup"}]},
    )
    events: Final = await _drain(stream)

    assert server.calls == ()
    assert provider.call_count == 1
    deltas: Final = tuple(event for event in events if _event_type(event) == "response.function_call_arguments.delta")
    assert tuple(item for item in _stream_items(events) if item[0] != "response.function_call_arguments.delta") == (
        ("response.created", None, None),
        ("response.in_progress", None, None),
        ("response.output_item.added", 0, "mcp_list_tools"),
        ("response.mcp_list_tools.in_progress", 0, None),
        ("response.mcp_list_tools.completed", 0, None),
        ("response.output_item.done", 0, "mcp_list_tools"),
        ("response.output_item.added", 1, "function_call"),
        ("response.function_call_arguments.done", 1, None),
        ("response.output_item.done", 1, "function_call"),
        ("response.completed", None, None),
    )
    assert {event.output_index for event in deltas} == {1}
    assert json.loads("".join(event.delta for event in deltas)) == _ASK_ARGUMENTS
    assert [event.sequence_number for event in events] == list(range(len(events)))
    streamed_call: Final = _item(events[-2].item)
    final_output: Final = events[-1].response.output
    assert _output_types(final_output) == ("mcp_list_tools", "function_call")
    assert (streamed_call["call_id"], streamed_call["name"], json.loads(streamed_call["arguments"])) == (
        _item(final_output[1])["call_id"],
        _item(final_output[1])["name"],
        _ASK_ARGUMENTS,
    )


@pytest.mark.asyncio
@respx.mock
async def test_an_allowlisted_server_receives_the_arguments_a_pre_call_guardrail_rewrote(
    monkeypatch: pytest.MonkeyPatch,
):
    from litellm.proxy import proxy_server

    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    _serve_responses(
        respx.mock,
        monkeypatch,
        (_tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"), _answer_round(_BEDROCK_MODEL)),
        _ROUTE_URLS[_BEDROCK_MODEL],
    )
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj,
        "pre_call_hook",
        AsyncMock(return_value={"modified_arguments": {"question": "[MASKED]"}}),
    )
    server: Final = _ScriptedMCPServer()

    await _gateway_hosted_mcp_request(_BEDROCK_MODEL, server)

    assert [(call.name, call.arguments) for call in server.calls] == [("ask_question", {"question": "[MASKED]"})]


def _stream_items(events: Sequence[ResponsesAPIStreamingResponse]) -> tuple[tuple[str, object, object], ...]:
    return tuple(
        (_event_type(event), getattr(event, "output_index", None), getattr(getattr(event, "item", None), "type", None))
        for event in events
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_failed", [False, True], ids=["tool_succeeds", "tool_fails"])
@respx.mock
async def test_an_allowlisted_hosted_mcp_url_streams_openai_mcp_events_in_order(
    monkeypatch: pytest.MonkeyPatch, tool_failed: bool
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    _serve_responses(respx.mock, monkeypatch, (_streamed_tool_call_round(), _streamed_answer_round()), _DEEPSEEK_URL)
    server: Final = _ScriptedMCPServer(tool_failed=tool_failed)

    stream: Final = await _gateway_hosted_mcp_request(_DEEPSEEK_MODEL, server, stream=True)
    events: Final = tuple([event async for event in stream])

    call_outcome: Final = "response.mcp_call.failed" if tool_failed else "response.mcp_call.completed"
    assert _stream_items(events) == (
        ("response.created", None, None),
        ("response.in_progress", None, None),
        ("response.output_item.added", 0, "mcp_list_tools"),
        ("response.mcp_list_tools.in_progress", 0, None),
        ("response.mcp_list_tools.completed", 0, None),
        ("response.output_item.done", 0, "mcp_list_tools"),
        ("response.output_item.added", 1, "mcp_call"),
        ("response.mcp_call.in_progress", 1, None),
        ("response.mcp_call_arguments.delta", 1, None),
        ("response.mcp_call_arguments.done", 1, None),
        (call_outcome, 1, None),
        ("response.output_item.done", 1, "mcp_call"),
        ("response.output_item.added", 2, "message"),
        ("response.content_part.added", 2, None),
        ("response.output_text.delta", 2, None),
        ("response.output_text.done", 2, None),
        ("response.content_part.done", 2, None),
        ("response.output_item.done", 2, "message"),
        ("response.completed", None, None),
    )
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert _item(events[6].item)["status"] == "in_progress"
    _assert_openai_mcp_output(events[-1].response.output, "failed" if tool_failed else "completed")
    assert [(call.name, call.arguments) for call in server.calls] == [("ask_question", _ASK_ARGUMENTS)]


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", ["always", None])
@respx.mock
async def test_an_allowlisted_hosted_mcp_tool_that_needs_approval_answers_400_or_is_dropped(
    monkeypatch: pytest.MonkeyPatch, approval: str | None
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    provider: Final = _serve_responses(respx.mock, monkeypatch, (_answer_round(_DEEPSEEK_MODEL),), _DEEPSEEK_URL)
    server: Final = _ScriptedMCPServer()
    tool: Final = {**_DEEPWIKI_TOOL, "require_approval": approval}

    with pytest.raises(litellm.BadRequestError) as refused:
        await _gateway_hosted_mcp_request(_DEEPSEEK_MODEL, server, tool)
    dropped: Final = await _gateway_hosted_mcp_request(_DEEPSEEK_MODEL, server, tool, drop_params=True)

    assert 'only with `require_approval: "never"`' in str(refused.value)
    assert isinstance(dropped, ResponsesAPIResponse)
    assert _output_types(dropped.output) == ("message",)
    assert _sent_tool_names(_sent_body(provider, 0)) == ()
    assert server.opened == ()


def _deepwiki_registered(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    from litellm.caching.caching import DualCache
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    registered: Final = MCPServer(
        server_id="srv-deepwiki",
        name="deepwiki",
        server_name="deepwiki",
        url="https://MCP.deepwiki.com/mcp/",
        transport=MCPTransport.http,
    )
    manager: Final = mcp_server_manager.global_mcp_server_manager
    mcp_client: Final = AsyncMock()
    mcp_client.list_tools_result.return_value = ListToolsResult(tools=[_ASK_TOOL])
    mcp_client.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text=_ASK_OUTPUT)])
    monkeypatch.setattr(manager, "registry", {registered.server_id: registered})
    monkeypatch.setattr(manager, "create_mcp_client", AsyncMock(return_value=mcp_client))
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(
            general_settings={},
            master_key="sk-master",
            proxy_logging_obj=None,
            prisma_client=None,
            user_api_key_cache=DualCache(),
        ),
    )
    return mcp_client


def _key_holding(server_id: str) -> object:
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth

    return UserAPIKeyAuth(
        api_key="sk-caller",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-key", mcp_servers=[server_id]),
    )


@pytest.mark.asyncio
@respx.mock
async def test_a_registered_hosted_mcp_url_runs_through_its_gateway_server_under_the_keys_permissions(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ())
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        (_tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"), _answer_round(_BEDROCK_MODEL)),
        _BEDROCK_URL,
    )
    server: Final = _ScriptedMCPServer()

    response: Final = await _gateway_hosted_mcp_request(
        _BEDROCK_MODEL, server, caller=_key_holding("srv-deepwiki")
    )

    assert isinstance(response, ResponsesAPIResponse)
    _assert_openai_mcp_output(response.output)
    assert mcp_client.call_tool.await_args is not None
    assert mcp_client.call_tool.await_args.args[0].name == "ask_question"
    assert _sent_tool_names(_sent_body(provider, 0)) == ("deepwiki-ask_question",)
    assert server.opened == ()


@pytest.mark.asyncio
@respx.mock
async def test_a_registered_hosted_mcp_url_answers_the_gateways_403_for_a_key_without_access(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException

    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(respx.mock, monkeypatch, (), _BEDROCK_URL)
    server: Final = _ScriptedMCPServer()

    with pytest.raises(HTTPException) as denied:
        await _gateway_hosted_mcp_request(
            _BEDROCK_MODEL, server, caller=_key_holding("srv-other")
        )

    assert denied.value.status_code == 403
    assert denied.value.detail == {"error": "The key is not allowed to access the requested MCP servers: deepwiki"}
    assert (server.opened, provider.call_count, mcp_client.call_tool.await_count) == ((), 0, 0)


def _virtual_key(**permission: object) -> object:
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth

    key: Final = UserAPIKeyAuth(
        api_key="sk-caller",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=(
            LiteLLM_ObjectPermissionTable(object_permission_id="op-key", **permission) if permission else None
        ),
    )
    return key.model_copy(update={"via_virtual_key": True})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("permission", "respects_mcp_scope", "denied"),
    [
        ({"mcp_servers": ["srv-other"]}, True, True),
        ({"mcp_servers": ["no-mcp-servers"]}, False, True),
        ({"mcp_servers": ["srv-other"]}, False, False),
        ({}, True, False),
    ],
    ids=["scoped_key_under_respects_scope", "opted_out_key", "scoped_key_by_default", "unscoped_key"],
)
@respx.mock
async def test_an_allowlisted_host_answers_the_gateways_403_to_a_key_kept_from_servers_open_to_every_key(
    monkeypatch: pytest.MonkeyPatch, permission: Mapping[str, object], respects_mcp_scope: bool, denied: bool
):
    from fastapi import HTTPException

    from litellm.proxy import proxy_server

    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    monkeypatch.setattr(
        proxy_server, "general_settings", {"mcp_allow_all_keys_respects_mcp_scope": respects_mcp_scope}
    )
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        (_tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"), _answer_round(_BEDROCK_MODEL)),
        _BEDROCK_URL,
    )
    server: Final = _ScriptedMCPServer()
    caller: Final = _virtual_key(**permission)

    if denied:
        with pytest.raises(HTTPException) as refused:
            await _gateway_hosted_mcp_request(_BEDROCK_MODEL, server, caller=caller)
        assert (refused.value.status_code, refused.value.detail) == (
            403,
            {"error": "The key is not allowed to access the requested MCP servers: deepwiki"},
        )
        assert (server.opened, provider.call_count) == ((), 0)
    else:
        await _gateway_hosted_mcp_request(_BEDROCK_MODEL, server, caller=caller)
        assert [call.name for call in server.calls] == ["ask_question"]


@pytest.mark.asyncio
@respx.mock
async def test_an_allowlisted_host_answers_the_gateways_403_to_a_managed_agent(monkeypatch: pytest.MonkeyPatch):
    from fastapi import HTTPException

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.types.agents import AgentResponse

    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    provider: Final = _serve_responses(respx.mock, monkeypatch, (), _BEDROCK_URL)
    server: Final = _ScriptedMCPServer()
    agent: Final = UserAPIKeyAuth(api_key="sk-agent", user_role=LitellmUserRoles.INTERNAL_USER, agent_id="agent-1")
    agent.managed_agent_policy = AgentResponse(agent_id="agent-1", agent_name="Agent", agent_card_params={})

    with pytest.raises(HTTPException) as refused:
        await _gateway_hosted_mcp_request(_BEDROCK_MODEL, server, caller=agent)

    assert (refused.value.status_code, refused.value.detail) == (
        403,
        {"error": "The key is not allowed to access the requested MCP servers: deepwiki"},
    )
    assert (server.opened, provider.call_count) == ((), 0)


@pytest.mark.asyncio
@respx.mock
async def test_an_allowlisted_tool_whose_description_a_guardrail_blocks_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException

    from litellm.proxy import proxy_server
    from litellm.types.utils import CallTypes

    async def block_listed_descriptions(**kwargs: object) -> object:
        if kwargs["call_type"] == CallTypes.list_mcp_tools.value:
            raise HTTPException(status_code=400, detail={"error": "tool description blocked"})
        return kwargs["data"]

    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "pre_call_hook", block_listed_descriptions)
    provider: Final = _serve_responses(respx.mock, monkeypatch, (_answer_round(_BEDROCK_MODEL),), _BEDROCK_URL)
    server: Final = _ScriptedMCPServer()

    response: Final = await _gateway_hosted_mcp_request(_BEDROCK_MODEL, server)

    assert isinstance(response, ResponsesAPIResponse)
    assert "deepwiki-ask_question" not in _sent_tool_names(_sent_body(provider, 0))
    assert [_item(item).get("tools") for item in response.output if _item(item)["type"] == "mcp_list_tools"] == [[]]
    assert server.calls == ()


_GITMCP_TOOL: Final = MappingProxyType(
    {"type": "mcp", "server_label": "gitmcp", "server_url": "https://gitmcp.io/docs", "require_approval": "never"}
)


def _deepwiki_reference(approval: str | None = None) -> Mapping[str, object]:
    reference: Final = {"type": "mcp", "server_label": "deepwiki", "server_url": "litellm_proxy/mcp/deepwiki"}
    return reference if approval is None else {**reference, "require_approval": approval}


def _output_summary(output: Sequence[object]) -> tuple[tuple[object, object, object], ...]:
    items: Final = tuple(_item(item) for item in output)
    return tuple((item["type"], item.get("server_label"), item.get("name")) for item in items)


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", [None, True], ids=["strict", "drop_params"])
@respx.mock
async def test_a_gateway_reference_that_needs_approval_beside_a_claimed_url_hands_its_call_back_unrun(
    monkeypatch: pytest.MonkeyPatch, drop_params: bool | None
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        (
            _tool_call_round(_BEDROCK_MODEL, "gitmcp-ask_question"),
            _tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"),
        ),
        _BEDROCK_URL,
    )
    server: Final = _ScriptedMCPServer()

    response: Final = await _gateway_hosted_mcp_request(
        _BEDROCK_MODEL,
        server,
        _GITMCP_TOOL,
        beside=(_deepwiki_reference(),),
        drop_params=drop_params,
        caller=_key_holding("srv-deepwiki"),
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_summary(response.output) == (
        ("mcp_list_tools", "gitmcp", None),
        ("mcp_call", "gitmcp", "ask_question"),
        ("message", None, None),
        ("function_call", None, "deepwiki-ask_question"),
    )
    assert [(call.name, call.arguments) for call in server.calls] == [("ask_question", _ASK_ARGUMENTS)]
    assert (mcp_client.call_tool.await_count, provider.call_count) == (0, 2)
    assert _sent_tool_names(_sent_body(provider, 0)) == ("deepwiki-ask_question", "gitmcp-ask_question")


@pytest.mark.asyncio
@respx.mock
async def test_a_never_gateway_reference_beside_a_claimed_url_runs_both_on_the_gateway(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    _serve_responses(
        respx.mock,
        monkeypatch,
        (
            _tool_call_round(_BEDROCK_MODEL, "gitmcp-ask_question"),
            _tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"),
            _answer_round(_BEDROCK_MODEL),
        ),
        _BEDROCK_URL,
    )
    server: Final = _ScriptedMCPServer()

    response: Final = await _gateway_hosted_mcp_request(
        _BEDROCK_MODEL,
        server,
        _GITMCP_TOOL,
        beside=(_deepwiki_reference("never"),),
        caller=_key_holding("srv-deepwiki"),
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_summary(response.output) == (
        ("mcp_list_tools", "deepwiki", None),
        ("mcp_list_tools", "gitmcp", None),
        ("mcp_call", "gitmcp", "ask_question"),
        ("mcp_call", "deepwiki", "ask_question"),
        ("message", None, None),
    )
    assert [call.name for call in server.calls] == ["ask_question"]
    assert mcp_client.call_tool.await_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_a_claimed_url_that_needs_approval_is_dropped_beside_a_gateway_reference_while_the_rest_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io", "mcp.example.com"))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        (_tool_call_round(_BEDROCK_MODEL, "gitmcp-ask_question"), _answer_round(_BEDROCK_MODEL)),
        _BEDROCK_URL,
    )
    server: Final = _ScriptedMCPServer()
    needs_approval: Final = {
        "type": "mcp",
        "server_label": "example",
        "server_url": "https://mcp.example.com/mcp",
        "require_approval": "always",
    }

    response: Final = await _gateway_hosted_mcp_request(
        _BEDROCK_MODEL,
        server,
        _GITMCP_TOOL,
        beside=(needs_approval, _deepwiki_reference()),
        drop_params=True,
        caller=_key_holding("srv-deepwiki"),
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_summary(response.output) == (
        ("mcp_list_tools", "gitmcp", None),
        ("mcp_call", "gitmcp", "ask_question"),
        ("message", None, None),
    )
    assert [opened.server_label for opened, _ in server.opened] == ["gitmcp"]
    assert [call.name for call in server.calls] == ["ask_question"]
    assert mcp_client.call_tool.await_count == 0
    assert _sent_tool_names(_sent_body(provider, 0)) == ("deepwiki-ask_question", "gitmcp-ask_question")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "beside", "clashing_name"),
    [
        (
            "deepwiki",
            {"type": "function", "name": "deepwiki-ask_question", "parameters": {"type": "object", "properties": {}}},
            "deepwiki-ask_question",
        ),
        (
            "deep wiki",
            {
                "type": "mcp",
                "server_label": "deep_wiki",
                "server_url": "https://mcp.example.com/mcp",
                "require_approval": "never",
            },
            "deep_wiki-ask_question",
        ),
    ],
    ids=["client_function", "second_direct_server"],
)
@respx.mock
async def test_a_direct_mcp_tool_whose_name_clashes_with_another_tool_answers_400_before_the_model_runs(
    monkeypatch: pytest.MonkeyPatch, label: str, beside: Mapping[str, object], clashing_name: str
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com", "mcp.example.com"))
    provider: Final = _serve_responses(respx.mock, monkeypatch, (), _BEDROCK_URL)
    server: Final = _ScriptedMCPServer()

    with pytest.raises(litellm.BadRequestError) as refused:
        await _gateway_hosted_mcp_request(
            _BEDROCK_MODEL, server, {**_DEEPWIKI_TOOL, "server_label": label}, beside=(beside,)
        )

    assert f"`{clashing_name}`" in str(refused.value)
    assert (server.calls, provider.call_count) == ((), 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "beside",
    [
        {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.example.com/mcp"},
        {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.unlisted.example/mcp"},
    ],
    ids=["second_direct_server", "unclaimed_server"],
)
@respx.mock
async def test_two_hosted_mcp_tools_with_the_same_server_label_answer_400_before_any_server_is_opened(
    monkeypatch: pytest.MonkeyPatch, beside: Mapping[str, object]
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com", "mcp.example.com"))
    provider: Final = _serve_responses(respx.mock, monkeypatch, (), _BEDROCK_URL)
    server: Final = _ScriptedMCPServer()

    with pytest.raises(litellm.BadRequestError) as refused:
        await _gateway_hosted_mcp_request(
            _BEDROCK_MODEL, server, beside=({**beside, "require_approval": "never"},), drop_params=True
        )

    assert "Duplicate MCP server_label `deepwiki`" in str(refused.value)
    assert (server.opened, provider.call_count) == ((), 0)


@pytest.mark.asyncio
@respx.mock
async def test_more_direct_mcp_servers_than_the_cap_answer_400_before_any_server_is_opened(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com", "mcp.example.com"))
    provider: Final = _serve_responses(respx.mock, monkeypatch, (), _BEDROCK_URL)
    server: Final = _ScriptedMCPServer()
    beside: Final = tuple(
        {
            "type": "mcp",
            "server_label": f"example-{index}",
            "server_url": "https://mcp.example.com/mcp",
            "require_approval": "never",
        }
        for index in range(MAX_GATEWAY_DIRECT_MCP_SERVERS)
    )

    with pytest.raises(litellm.BadRequestError) as refused:
        await _gateway_hosted_mcp_request(_BEDROCK_MODEL, server, beside=beside)

    assert f"more than the {MAX_GATEWAY_DIRECT_MCP_SERVERS} allowed" in str(refused.value)
    assert (server.opened, provider.call_count) == ((), 0)


@pytest.mark.asyncio
@respx.mock
async def test_a_model_that_keeps_calling_a_direct_mcp_tool_past_the_round_cap_gets_the_call_back_unrun(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        tuple(_tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question") for _ in range(3)),
        _BEDROCK_URL,
    )
    server: Final = _ScriptedMCPServer()

    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        response: Final = await litellm.aresponses(
            model=_BEDROCK_MODEL,
            input="Ask DeepWiki what LiteLLM is",
            tools=[dict(_DEEPWIKI_TOOL)],
            max_agentic_loops=1,
            **_ROUTE_KWARGS[_BEDROCK_MODEL],
        )

    assert [call.name for call in server.calls] == ["ask_question"]
    assert provider.call_count == 2
    assert _output_summary(response.output) == (
        ("mcp_list_tools", "deepwiki", None),
        ("mcp_call", "deepwiki", "ask_question"),
        ("message", None, None),
        ("function_call", None, "deepwiki-ask_question"),
    )


@pytest.mark.asyncio
@respx.mock
async def test_a_streamed_gateway_reference_that_needs_approval_beside_a_claimed_url_hands_its_call_back_unrun(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx.mock,
        monkeypatch,
        (_streamed_tool_call_round("gitmcp-ask_question"), _streamed_tool_call_round("deepwiki-ask_question")),
        _DEEPSEEK_URL,
    )
    server: Final = _ScriptedMCPServer()

    stream: Final = await _gateway_hosted_mcp_request(
        _DEEPSEEK_MODEL,
        server,
        _GITMCP_TOOL,
        beside=(_deepwiki_reference(),),
        stream=True,
        caller=_key_holding("srv-deepwiki"),
    )
    events: Final = tuple([event async for event in stream])

    assert _event_type(events[-1]) == "response.completed"
    assert _output_summary(events[-1].response.output) == (
        ("mcp_list_tools", "gitmcp", None),
        ("mcp_call", "gitmcp", "ask_question"),
        ("function_call", None, "deepwiki-ask_question"),
    )
    assert [call.name for call in server.calls] == ["ask_question"]
    assert (mcp_client.call_tool.await_count, provider.call_count) == (0, 2)


@pytest.mark.asyncio
@respx.mock
async def test_a_failed_gateway_tool_call_keeps_the_original_tool_execution_results_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    mcp_client.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="Repository not found")], isError=True
    )
    _serve_responses(
        respx.mock,
        monkeypatch,
        (_tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"), _answer_round(_BEDROCK_MODEL)),
        _BEDROCK_URL,
    )

    response: Final = await litellm.aresponses(
        model=_BEDROCK_MODEL,
        input="Ask DeepWiki what LiteLLM is",
        tools=[dict(_deepwiki_reference("never"))],
        litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
        **_ROUTE_KWARGS[_BEDROCK_MODEL],
    )

    assert isinstance(response, ResponsesAPIResponse)
    results_item: Final = next(
        _item(item) for item in response.output if _item(item)["type"] == "tool_execution_results"
    )
    assert json.loads(results_item["content"][0]["text"]) == [
        {"tool_call_id": "call_ask", "result": "Repository not found", "name": "deepwiki-ask_question"}
    ]


_NATIVE_ROUTES: Final = MappingProxyType(
    {
        "openai/gpt-5.6": (r"https://gateway-mcp\.example\.com/v1/responses", "tools", "server_url"),
        "anthropic/claude-haiku-5-5": (r"https://gateway-mcp\.example\.com/v1/messages", "mcp_servers", "url"),
    }
)


def _native_answer(model: str) -> httpx.Response:
    if model.startswith("anthropic/"):
        return httpx.Response(200, json=_anthropic_message("msg_native", [{"type": "text", "text": "ok"}], "end_turn"))
    return httpx.Response(
        200,
        json=_openai_response(
            "resp_native",
            "completed",
            [
                {
                    "type": "message",
                    "id": "msg_native",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", tuple(_NATIVE_ROUTES))
@respx.mock
async def test_a_route_that_runs_hosted_mcp_natively_still_gets_the_tool_with_the_gateway_switch_on(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    _gateway_runs_hosted_mcp(monkeypatch, ("mcp.deepwiki.com",))
    url, field, url_key = _NATIVE_ROUTES[model]
    provider: Final = _serve_responses(respx.mock, monkeypatch, (_native_answer(model),), url)
    server: Final = _ScriptedMCPServer()

    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        await litellm.aresponses(
            model=model,
            input="Ask DeepWiki what LiteLLM is",
            tools=[dict(_DEEPWIKI_TOOL)],
            api_key="sk-native",
            api_base="https://gateway-mcp.example.com/v1"
            if model.startswith("openai/")
            else "https://gateway-mcp.example.com",
        )

    assert [entry.get(url_key) for entry in _sent_body(provider, 0)[field]] == [_DEEPWIKI_URL]
    assert server.opened == ()


def _loads_no_db_object(object_type: str) -> bool:
    return False


def _proxy_database(monkeypatch: pytest.MonkeyPatch, database: object) -> None:
    proxy_server: Final = sys.modules["litellm.proxy.proxy_server"]
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "should_load_db_object", _loads_no_db_object, raising=False)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False, raising=False)


def _research_toolset(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.types.mcp_server.mcp_toolset import MCPToolset

    manager: Final = mcp_server_manager.global_mcp_server_manager
    toolset: Final = MCPToolset(
        toolset_id="ts-research",
        toolset_name="research",
        tools=[{"server_id": "srv-deepwiki", "tool_name": "ask_question"}],
    )
    _proxy_database(monkeypatch, SimpleNamespace())
    monkeypatch.setattr(manager, "get_toolset_by_name_cached", AsyncMock(return_value=toolset))
    monkeypatch.setattr(
        manager, "resolve_toolset_tool_permissions", AsyncMock(return_value={"srv-deepwiki": ["ask_question"]})
    )


def _key_holding_research() -> object:
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth

    return UserAPIKeyAuth(
        api_key="sk-caller",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-key", mcp_servers=["srv-deepwiki"], mcp_toolsets=["ts-research"]
        ),
    )


@pytest.mark.asyncio
@respx.mock
async def test_a_toolset_reference_that_needs_approval_beside_a_claimed_url_hands_its_servers_call_back_unrun(
    monkeypatch: pytest.MonkeyPatch,
):
    _gateway_runs_hosted_mcp(monkeypatch, ())
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    _research_toolset(monkeypatch)
    provider: Final = _serve_responses(
        respx.mock, monkeypatch, (_tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"),), _BEDROCK_URL
    )
    toolset_reference: Final = {"type": "mcp", "server_label": "research", "server_url": "litellm_proxy/mcp/research"}

    response: Final = await _gateway_hosted_mcp_request(
        _BEDROCK_MODEL, _ScriptedMCPServer(), beside=(toolset_reference,), caller=_key_holding_research()
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert ("function_call", None, "deepwiki-ask_question") in _output_summary(response.output)
    assert "mcp_call" not in _output_types(response.output)
    assert (mcp_client.call_tool.await_count, provider.call_count) == (0, 1)


_HANDED_BACK_OUTPUT: Final = "DeepWiki answered on the caller's side."


def _round_calling_both_servers(stream: bool) -> httpx.Response:
    calls: Final = tuple(
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(_ASK_ARGUMENTS)}}
        for call_id, name in (("call_git", "gitmcp-ask_question"), ("call_wiki", "deepwiki-ask_question"))
    )
    if not stream:
        return _chat_round(
            {"role": "assistant", "content": None, "reasoning_content": "Ask both servers.", "tool_calls": list(calls)}
        )
    return _chat_stream(
        _chat_chunk({"role": "assistant", "reasoning_content": "Ask both servers."}, None),
        _chat_chunk({"tool_calls": [{"index": index, **call} for index, call in enumerate(calls)]}, None),
        _chat_chunk({}, "tool_calls"),
    )


async def _client_response(
    response: ResponsesAPIResponse | AsyncIterator[ResponsesAPIStreamingResponse],
) -> tuple[ResponsesAPIResponse, tuple[ResponsesAPIStreamingResponse, ...]]:
    if isinstance(response, ResponsesAPIResponse):
        return response, ()
    events: Final = tuple([event async for event in response])
    assert _event_type(events[-1]) == "response.completed"
    return events[-1].response, events


def _added_items(events: Sequence[object]) -> tuple[tuple[int, Mapping[str, object]], ...]:
    return tuple(
        (event.output_index, _item(event.item))
        for event in events
        if _event_type(event) == "response.output_item.added"
    )


def _item_key(item: Mapping[str, object]) -> tuple[object, object]:
    return item["type"], item.get("id")


def _assert_streamed_like_an_openai_accumulator_reads_it(events: Sequence[object], output: Sequence[object]) -> None:
    added: Final = _added_items(events)
    assert [index for index, _ in added] == list(range(len(added)))
    added_at: Final = {
        event.output_index: position
        for position, event in enumerate(events)
        if _event_type(event) == "response.output_item.added"
    }
    assert all(
        added_at[event.output_index] <= position
        for position, event in enumerate(events)
        if isinstance(getattr(event, "output_index", None), int)
    )
    assert [_item_key(_item(item)) for item in output] == [_item_key(item) for _, item in added]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_a_round_calling_a_claimed_and_a_held_tool_runs_the_claimed_one_and_its_follow_up_pairs_every_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx_mock, monkeypatch, (_round_calling_both_servers(stream), _answer_round(_DEEPSEEK_MODEL)), _DEEPSEEK_URL
    )
    server: Final = _ScriptedMCPServer()
    recorder: Final = _RoundRecorder()
    tools: Final = [dict(_GITMCP_TOOL), dict(_deepwiki_reference())]

    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        response, events = await _client_response(
            await litellm.aresponses(
                model=_DEEPSEEK_MODEL,
                input="Ask both servers",
                tools=tools,
                stream=stream,
                litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
                litellm_logging_obj=_rounds_logging_obj(recorder, f"mixed-round-{stream}"),
                **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
            )
        )

    if stream:
        _assert_streamed_like_an_openai_accumulator_reads_it(events, response.output)
    items: Final = tuple(_item(item) for item in response.output)
    assert tuple(item["type"] for item in items) == ("mcp_list_tools", "reasoning", "function_call", "mcp_call")
    assert (items[2]["call_id"], items[2]["name"]) == ("call_wiki", "deepwiki-ask_question")
    assert (items[3]["server_label"], items[3]["name"], items[3]["output"]) == ("gitmcp", "ask_question", _ASK_OUTPUT)
    assert [call.name for call in server.calls] == ["ask_question"]
    assert mcp_client.call_tool.await_count == 0
    await _settle(lambda: len(recorder.logged) > 0)

    spend_logs: Final = _SpendLogs(
        (
            {
                "request_id": response.id,
                "session_id": "mixed-round-session",
                "proxy_server_request": {"input": "Ask both servers"},
                "response": recorder.logged[0]["response"],
            },
        )
    )
    _proxy_database(monkeypatch, SimpleNamespace(db=spend_logs))
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        follow_up: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input=[{"type": "function_call_output", "call_id": "call_wiki", "output": _HANDED_BACK_OUTPUT}],
            previous_response_id=response.id,
            tools=tools,
            litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )

    assert isinstance(follow_up, ResponsesAPIResponse)
    replayed: Final = _sent_body(provider, 1)["messages"]
    assert [message["role"] for message in replayed] == ["user", "assistant", "tool", "tool"]
    assistant: Final = replayed[1]
    assert assistant["reasoning_content"] == "Ask both servers."
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_wiki", items[3]["id"]]
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == [
        "deepwiki-ask_question",
        "gitmcp-ask_question",
    ]
    assert [(message["tool_call_id"], message["content"]) for message in replayed[2:]] == [
        (items[3]["id"], _ASK_OUTPUT),
        ("call_wiki", _HANDED_BACK_OUTPUT),
    ]


def _converse_round_thinking_first(
    thought: str, signature: str, block: Mapping[str, object], stop_reason: str
) -> httpx.Response:
    reasoning: Final = {"reasoningContent": {"reasoningText": {"text": thought, "signature": signature}}}
    return httpx.Response(
        200,
        json={
            "output": {"message": {"role": "assistant", "content": [reasoning, dict(block)]}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 40, "outputTokens": 8, "totalTokens": 48},
            "metrics": {"latencyMs": 10},
        },
    )


def _converse_block_shape(block: Mapping[str, object]) -> tuple[object, ...]:
    match block:
        case {"reasoningContent": {"reasoningText": {"text": text, "signature": signature}}}:
            return ("reasoning", text, signature)
        case {"toolUse": {"toolUseId": call_id, "name": name}}:
            return ("toolUse", call_id, name)
        case {"toolResult": {"toolUseId": call_id, "content": content}}:
            return ("toolResult", call_id, tuple(part.get("text") for part in content))
        case {"text": text}:
            return ("text", text)
        case _:
            return (tuple(block),)


def _converse_conversation(body: Mapping[str, object]) -> tuple[tuple[object, tuple[tuple[object, ...], ...]], ...]:
    return tuple(
        (message["role"], tuple(_converse_block_shape(block) for block in message["content"]))
        for message in body["messages"]
    )


def _stored_turn(
    response_id: str, logged: StandardLoggingPayload, request: Mapping[str, object]
) -> Mapping[str, object]:
    return {
        "request_id": response_id,
        "session_id": f"{response_id}-session",
        "custom_llm_provider": logged["custom_llm_provider"],
        "model": logged["model"],
        "api_base": logged["api_base"],
        "proxy_server_request": dict(request),
        "response": logged["response"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("replayed_from", ["stored_turn", "input", "stored_turn_without_tools"])
async def test_a_follow_up_to_a_claimed_round_with_thinking_on_both_sides_of_its_call_keeps_each_thought_in_place(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, replayed_from: str
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    without_tools: Final = replayed_from == "stored_turn_without_tools"
    monkeypatch.setattr(litellm, "bedrock_neutralize_orphaned_tool_blocks", not without_tools)
    monkeypatch.setattr(litellm, "modify_params", without_tools)
    provider: Final = _serve_responses(
        respx_mock,
        monkeypatch,
        (
            _converse_round_thinking_first(
                "Look it up.",
                "sig-before-call",
                {"toolUse": {"toolUseId": "tooluse_ask", "name": "gitmcp-ask_question", "input": _ASK_ARGUMENTS}},
                "tool_use",
            ),
            _converse_round_thinking_first("Answer now.", "sig-after-call", {"text": _ASK_ANSWER}, "end_turn"),
            _answer_round(_BEDROCK_MODEL),
        ),
        _BEDROCK_URL,
    )
    server: Final = _ScriptedMCPServer()
    recorder: Final = _RoundRecorder()
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        first: Final = await litellm.aresponses(
            model=_BEDROCK_MODEL,
            input="Ask DeepWiki what LiteLLM is",
            tools=[dict(_GITMCP_TOOL)],
            reasoning={"effort": "low"},
            litellm_logging_obj=_rounds_logging_obj(recorder, f"thinking-around-call-{replayed_from}"),
            **_ROUTE_KWARGS[_BEDROCK_MODEL],
        )
    assert isinstance(first, ResponsesAPIResponse)
    assert _output_types(first.output) == ("mcp_list_tools", "reasoning", "mcp_call", "reasoning", "message")
    await _settle(lambda: len(recorder.logged) > 0)

    stored_request: Final = {"input": "Ask DeepWiki what LiteLLM is", "tools": [dict(_GITMCP_TOOL)]}
    stored: Final = _stored_turn(first.id, recorder.logged[0], stored_request)
    _proxy_database(monkeypatch, SimpleNamespace(db=_SpendLogs((stored,))))
    continued: Final = (
        {
            "input": [
                {"type": "message", "role": "user", "content": "Ask DeepWiki what LiteLLM is"},
                *(_item(item) for item in first.output),
                {"type": "message", "role": "user", "content": "Who maintains it?"},
            ]
        }
        if replayed_from == "input"
        else {"previous_response_id": first.id, "input": "Who maintains it?"}
    )
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        follow_up: Final = await litellm.aresponses(
            model=_BEDROCK_MODEL,
            tools=[] if without_tools else [dict(_GITMCP_TOOL)],
            reasoning={"effort": "low"},
            **continued,
            **_ROUTE_KWARGS[_BEDROCK_MODEL],
        )

    assert isinstance(follow_up, ResponsesAPIResponse)
    call_id: Final = _item(first.output[2])["id"]
    assert _converse_conversation(_sent_body(provider, 2)) == (
        ("user", (("text", "Ask DeepWiki what LiteLLM is"),)),
        (
            "assistant",
            (("reasoning", "Look it up.", "sig-before-call"), ("toolUse", call_id, "gitmcp-ask_question")),
        ),
        ("user", (("toolResult", call_id, (_ASK_OUTPUT,)),)),
        ("assistant", (("reasoning", "Answer now.", "sig-after-call"), ("text", _ASK_ANSWER))),
        ("user", (("text", "Who maintains it?"),)),
    )
    assert [call.name for call in server.calls] == ["ask_question"]


def _ran_call(call_id: str, server_label: str, output: str = _ASK_OUTPUT) -> Mapping[str, object]:
    return {
        "type": "mcp_call",
        "id": call_id,
        "server_label": server_label,
        "name": "ask_question",
        "arguments": json.dumps(_ASK_ARGUMENTS),
        "output": output,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_a_resent_mcp_call_whose_tool_the_request_does_not_run_stays_out_of_a_bedrock_conversation(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    provider: Final = _serve_responses(respx_mock, monkeypatch, (_answer_round(_BEDROCK_MODEL),), _BEDROCK_URL)

    response: Final = await litellm.aresponses(
        model=_BEDROCK_MODEL,
        input=[
            {"type": "message", "role": "user", "content": "Ask DeepWiki what LiteLLM is"},
            _ran_call("mcp_resent", "gitmcp"),
            {"type": "message", "role": "assistant", "content": _ASK_ANSWER},
            {"type": "message", "role": "user", "content": "Who maintains it?"},
        ],
        **_ROUTE_KWARGS[_BEDROCK_MODEL],
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert _converse_conversation(_sent_body(provider, 0)) == (
        ("user", (("text", "Ask DeepWiki what LiteLLM is"),)),
        ("assistant", (("text", _ASK_ANSWER),)),
        ("user", (("text", "Who maintains it?"),)),
    )


@pytest.mark.asyncio
async def test_a_follow_up_without_tools_to_a_stored_gateway_reference_stream_replays_none_of_its_calls(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _serve_responses(respx_mock, monkeypatch, (_streamed(_ROUND_ONE_BODY), _streamed(_ROUND_TWO_BODY)))
    recorder: Final = _RoundRecorder()
    first: Final = _final_response(await _stream_auto_execute(_rounds_logging_obj(recorder, "legacy-stream-follow-up")))
    await _settle(lambda: len(recorder.logged) > 0)
    assert "mcp_call" in _output_types(recorder.logged[0]["response"]["output"])
    stored_request: Final = {"input": "Call echo with PING", "tools": [dict(_ECHO_MCP_TOOL)]}
    stored: Final = _stored_turn(first.id, recorder.logged[0], stored_request)
    _proxy_database(monkeypatch, SimpleNamespace(db=_SpendLogs((stored,))))
    provider: Final = _serve_responses(respx_mock, monkeypatch, (_answer_round(_BEDROCK_MODEL),), _BEDROCK_URL)

    follow_up: Final = await litellm.aresponses(
        model=_BEDROCK_MODEL,
        input="Who maintains it?",
        previous_response_id=first.id,
        **_ROUTE_KWARGS[_BEDROCK_MODEL],
    )

    assert isinstance(follow_up, ResponsesAPIResponse)
    assert _converse_conversation(_sent_body(provider, 0)) == (
        ("user", (("text", "Call echo with PING"),)),
        ("assistant", (("text", "PING"),)),
        ("user", (("text", "Who maintains it?"),)),
    )


@pytest.mark.asyncio
async def test_a_follow_up_without_tools_to_a_turn_a_native_provider_ran_its_hosted_tool_in_replays_none_of_its_calls(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    native_turn: Final = _openai_response(
        "resp_native_gitmcp",
        "completed",
        (
            {"type": "mcp_list_tools", "id": "mcpl_native", "server_label": "gitmcp", "tools": []},
            _ran_call("mcp_native", "gitmcp"),
            {
                "type": "message",
                "id": "msg_native",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _ASK_ANSWER, "annotations": []}],
            },
        ),
    )
    _serve_responses(
        respx_mock, monkeypatch, (httpx.Response(200, json=native_turn),), _NATIVE_ROUTES["openai/gpt-5.6"][0]
    )
    server: Final = _ScriptedMCPServer()
    recorder: Final = _RoundRecorder()
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        first: Final = await litellm.aresponses(
            model="openai/gpt-5.6",
            input="Ask DeepWiki what LiteLLM is",
            tools=[dict(_GITMCP_TOOL)],
            api_key="sk-native",
            api_base="https://gateway-mcp.example.com/v1",
            litellm_logging_obj=_rounds_logging_obj(recorder, "native-direct-tool"),
        )
    assert isinstance(first, ResponsesAPIResponse)
    await _settle(lambda: len(recorder.logged) > 0)
    stored_request: Final = {"input": "Ask DeepWiki what LiteLLM is", "tools": [dict(_GITMCP_TOOL)]}
    stored: Final = _stored_turn(first.id, recorder.logged[0], stored_request)
    _proxy_database(monkeypatch, SimpleNamespace(db=_SpendLogs((stored,))))
    provider: Final = _serve_responses(respx_mock, monkeypatch, (_answer_round(_BEDROCK_MODEL),), _BEDROCK_URL)

    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        follow_up: Final = await litellm.aresponses(
            model=_BEDROCK_MODEL,
            input="Who maintains it?",
            previous_response_id=first.id,
            **_ROUTE_KWARGS[_BEDROCK_MODEL],
        )

    assert isinstance(follow_up, ResponsesAPIResponse)
    assert _converse_conversation(_sent_body(provider, 0)) == (
        ("user", (("text", "Ask DeepWiki what LiteLLM is"),)),
        ("assistant", (("text", _ASK_ANSWER),)),
        ("user", (("text", "Who maintains it?"),)),
    )
    assert server.opened == ()


@pytest.mark.asyncio
async def test_a_follow_up_without_tools_to_a_round_that_ran_a_url_and_a_gateway_reference_replays_only_the_url_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx_mock,
        monkeypatch,
        (_round_calling_both_servers(False), _answer_round(_DEEPSEEK_MODEL), _answer_round(_DEEPSEEK_MODEL)),
        _DEEPSEEK_URL,
    )
    server: Final = _ScriptedMCPServer()
    recorder: Final = _RoundRecorder()
    tools: Final = [dict(_GITMCP_TOOL), dict(_deepwiki_reference("never"))]
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        first: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input="Ask both servers",
            tools=tools,
            litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
            litellm_logging_obj=_rounds_logging_obj(recorder, "url-and-reference"),
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )
    assert isinstance(first, ResponsesAPIResponse)
    calls: Final = tuple(_item(item) for item in first.output if _item(item)["type"] == "mcp_call")
    ran: Final = {call["server_label"]: call["id"] for call in calls}
    assert set(ran) == {"gitmcp", "deepwiki"}
    assert mcp_client.call_tool.await_count == 1
    await _settle(lambda: len(recorder.logged) > 0)
    stored: Final = _stored_turn(first.id, recorder.logged[0], {"input": "Ask both servers", "tools": tools})
    _proxy_database(monkeypatch, SimpleNamespace(db=_SpendLogs((stored,))))

    follow_up: Final = await litellm.aresponses(
        model=_DEEPSEEK_MODEL,
        input="Who maintains it?",
        previous_response_id=first.id,
        **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
    )

    assert isinstance(follow_up, ResponsesAPIResponse)
    replayed: Final = _sent_body(provider, 2)["messages"]
    assert [_message_shape(message)[2] for message in replayed] == [(), ("gitmcp-ask_question",), (), (), ()]
    assert [(message["tool_call_id"], message["content"]) for message in replayed if message["role"] == "tool"] == [
        (ran["gitmcp"], _ASK_OUTPUT)
    ]


@pytest.mark.asyncio
async def test_a_resent_mcp_call_next_to_one_the_request_runs_stays_out_while_the_run_one_keeps_its_pair(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    provider: Final = _serve_responses(respx_mock, monkeypatch, (_answer_round(_DEEPSEEK_MODEL),), _DEEPSEEK_URL)
    server: Final = _ScriptedMCPServer()

    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        response: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input=[
                {"type": "message", "role": "user", "content": "Ask both servers"},
                _ran_call("mcp_git", "gitmcp"),
                _ran_call("mcp_other", "other-docs", "Other docs say it is a proxy"),
                {"type": "message", "role": "assistant", "content": _ASK_ANSWER},
                {"type": "message", "role": "user", "content": "Who maintains it?"},
            ],
            tools=[dict(_GITMCP_TOOL)],
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )

    assert isinstance(response, ResponsesAPIResponse)
    sent: Final = _sent_body(provider, 0)["messages"]
    assert [_message_shape(message)[2] for message in sent] == [(), ("gitmcp-ask_question",), (), (), ()]
    assert [(message["tool_call_id"], message["content"]) for message in sent if message["role"] == "tool"] == [
        ("mcp_git", _ASK_OUTPUT)
    ]
    assert server.calls == ()


_DOCS_TOOL: Final = MappingProxyType({**_GITMCP_TOOL, "server_label": "react-docs"})


def _labelled_reference(kind: str) -> Mapping[str, object]:
    name: Final = "research" if kind == "toolset" else "deepwiki"
    return {"type": "mcp", "server_label": "notes", "server_url": f"litellm_proxy/mcp/{name}", "require_approval": "never"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["toolset", "server"])
async def test_every_mcp_item_of_a_gateway_reference_carries_the_label_the_caller_gave_it(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, kind: str
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    if kind == "toolset":
        _research_toolset(monkeypatch)
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_tool_call_round(_DEEPSEEK_MODEL, "deepwiki-ask_question"), _answer_round(_DEEPSEEK_MODEL)),
        _DEEPSEEK_URL,
    )

    response: Final = await _gateway_hosted_mcp_request(
        _DEEPSEEK_MODEL,
        _ScriptedMCPServer(),
        _DOCS_TOOL,
        beside=(_labelled_reference(kind),),
        caller=_key_holding_research() if kind == "toolset" else _key_holding("srv-deepwiki"),
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_summary(response.output) == (
        ("mcp_list_tools", "notes", None),
        ("mcp_list_tools", "react-docs", None),
        ("mcp_call", "notes", "ask_question"),
        ("message", None, None),
    )
    assert [tool["name"] for tool in _item(response.output[0])["tools"]] == ["ask_question"]
    assert mcp_client.call_tool.await_count == 1


def _mirror_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager: Final = mcp_server_manager.global_mcp_server_manager
    mirror: Final = MCPServer(
        server_id="srv-mirror",
        name="mirror",
        server_name="mirror",
        url="https://mirror.example.com/mcp",
        transport=MCPTransport.http,
    )
    monkeypatch.setattr(manager, "registry", {**manager.registry, mirror.server_id: mirror})
    monkeypatch.setattr(
        manager,
        "resolve_toolset_tool_permissions",
        AsyncMock(return_value={"srv-deepwiki": ["ask_question"], "srv-mirror": ["ask_question"]}),
    )


def _key_holding_research_on_both_servers() -> object:
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth

    return UserAPIKeyAuth(
        api_key="sk-caller",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-key", mcp_servers=["srv-deepwiki", "srv-mirror"], mcp_toolsets=["ts-research"]
        ),
    )


@pytest.mark.asyncio
async def test_a_toolset_spanning_two_servers_with_the_same_tool_lists_both_under_its_label_by_their_gateway_names(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    _research_toolset(monkeypatch)
    _mirror_registered(monkeypatch)
    _serve_responses(
        respx_mock,
        monkeypatch,
        (_tool_call_round(_DEEPSEEK_MODEL, "mirror-ask_question"), _answer_round(_DEEPSEEK_MODEL)),
        _DEEPSEEK_URL,
    )

    response: Final = await _gateway_hosted_mcp_request(
        _DEEPSEEK_MODEL,
        _ScriptedMCPServer(),
        _DOCS_TOOL,
        beside=(_labelled_reference("toolset"),),
        caller=_key_holding_research_on_both_servers(),
    )

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_summary(response.output) == (
        ("mcp_list_tools", "notes", None),
        ("mcp_list_tools", "react-docs", None),
        ("mcp_call", "notes", "mirror-ask_question"),
        ("message", None, None),
    )
    assert sorted(tool["name"] for tool in _item(response.output[0])["tools"]) == [
        "deepwiki-ask_question",
        "mirror-ask_question",
    ]
    assert mcp_client.call_tool.await_count == 1


_ASK_TOOL_LISTED: Final = MappingProxyType(
    {
        "name": "ask_question",
        "description": _ASK_TOOL.description,
        "input_schema": _ASK_TOOL.input_schema,
        "annotations": {"read_only": False},
    }
)


async def _first_turn_with_both_listings(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, server: _ScriptedMCPServer
) -> tuple[ResponsesAPIResponse, respx.Route, Mapping[str, object]]:
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx_mock, monkeypatch, (_answer_round(_DEEPSEEK_MODEL), _answer_round(_DEEPSEEK_MODEL)), _DEEPSEEK_URL
    )
    recorder: Final = _RoundRecorder()
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        response: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input="Which servers can you use?",
            tools=[dict(_GITMCP_TOOL), dict(_deepwiki_reference("never"))],
            litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
            litellm_logging_obj=_rounds_logging_obj(recorder, "listed-turn"),
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )
    assert isinstance(response, ResponsesAPIResponse)
    await _settle(lambda: len(recorder.logged) > 0)
    return response, provider, recorder.logged[0]["response"]


def _unlisted(output: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    return tuple(item for item in map(_item, output) if item["type"] != "mcp_list_tools")


async def _follow_up(
    monkeypatch: pytest.MonkeyPatch,
    server: _ScriptedMCPServer,
    first: ResponsesAPIResponse,
    stored_response: Mapping[str, object],
    context: str,
    listings: Sequence[Mapping[str, object]],
) -> ResponsesAPIResponse:
    question: Final = {"type": "message", "role": "user", "content": "Ask again"}
    spend_logs: Final = _SpendLogs(
        (
            {
                "request_id": first.id,
                "session_id": "listed-session",
                "proxy_server_request": {"input": "Which servers can you use?"},
                "response": {**stored_response, "output": [*listings, *_unlisted(stored_response["output"])]},
            },
        )
    )
    _proxy_database(monkeypatch, SimpleNamespace(db=spend_logs))
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        follow_up: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input=[question] if context == "stored_turn" else [*listings, question],
            previous_response_id=first.id if context == "stored_turn" else None,
            tools=[dict(_GITMCP_TOOL), dict(_deepwiki_reference("never"))],
            litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )
    assert isinstance(follow_up, ResponsesAPIResponse)
    return follow_up


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["stored_turn", "input"])
async def test_a_follow_up_lists_no_server_again_whose_listing_the_conversation_already_holds(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, context: str
):
    server: Final = _ScriptedMCPServer()
    first, provider, stored_response = await _first_turn_with_both_listings(respx_mock, monkeypatch, server)
    listings: Final = tuple(_item(item) for item in stored_response["output"][:2])
    assert listings == tuple(_item(item) for item in first.output[:2])
    assert _output_summary(listings) == (("mcp_list_tools", "deepwiki", None), ("mcp_list_tools", "gitmcp", None))

    follow_up: Final = await _follow_up(monkeypatch, server, first, stored_response, context, listings)

    assert _output_types(follow_up.output) == ("message",)
    assert len(server.opened) == 2
    assert _sent_tool_names(_sent_body(provider, 1)) == ("deepwiki-ask_question", "gitmcp-ask_question")


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["stored_turn", "input"])
async def test_a_follow_up_lists_a_server_again_when_its_tools_changed_since_the_conversation_listed_them(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, context: str
):
    server: Final = _ScriptedMCPServer()
    first, provider, stored_response = await _first_turn_with_both_listings(respx_mock, monkeypatch, server)
    deepwiki_listing, gitmcp_listing = (_item(item) for item in first.output[:2])
    revoked: Final = {**deepwiki_listing, "tools": [*deepwiki_listing["tools"], {**_ASK_TOOL_LISTED, "name": "drop_repo"}]}

    follow_up: Final = await _follow_up(
        monkeypatch, server, first, stored_response, context, (revoked, gitmcp_listing)
    )

    assert _output_summary(follow_up.output) == (("mcp_list_tools", "deepwiki", None), ("message", None, None))
    assert _item(follow_up.output[0])["tools"] == deepwiki_listing["tools"]
    assert _item(follow_up.output[0])["id"] != deepwiki_listing["id"]
    assert len(server.opened) == 2


class _StoredTurnReads:
    def __init__(self, stored_turns: StoredTurns) -> None:
        self.stored_turns: Final = stored_turns
        self.asked: tuple[tuple[str, tuple[str, ...]], ...] = ()

    async def __call__(self, previous_response_id: str, client_response_ids: tuple[str, ...]) -> StoredTurns:
        self.asked = (*self.asked, (previous_response_id, client_response_ids))
        return self.stored_turns


@pytest.mark.asyncio
async def test_a_follow_up_reads_its_stored_turn_once_for_both_its_listings_and_its_history(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    server: Final = _ScriptedMCPServer()
    first, provider, stored_response = await _first_turn_with_both_listings(respx_mock, monkeypatch, server)
    stored_turn: Final = {
        "request_id": first.id,
        "session_id": "listed-session",
        "proxy_server_request": {"input": "Which servers can you use?"},
        "response": stored_response,
    }
    reads: Final = _StoredTurnReads(
        StoredTurns(spend_logs=(stored_turn,), proxy_server_requests=(stored_turn["proxy_server_request"],))
    )

    with using_responses_dependencies(
        ResponsesDependencies(direct_client_factory=server.client_for, stored_turns_source=reads)
    ):
        follow_up: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input=[{"type": "message", "role": "user", "content": "Ask again"}],
            previous_response_id=first.id,
            tools=[dict(_GITMCP_TOOL), dict(_deepwiki_reference("never"))],
            litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )

    assert reads.asked == ((first.id, (first.id,)),)
    assert isinstance(follow_up, ResponsesAPIResponse)
    assert _output_types(follow_up.output) == ("message",)
    replayed: Final = _sent_body(provider, 1)["messages"]
    assert [message["content"] for message in replayed if message["role"] == "user"] == [
        "Which servers can you use?",
        "Ask again",
    ]


_PRIOR_RESPONSE_ID: Final = "resp_prior_turn"
_PRIOR_ANSWER: Final = "An AI gateway"


def _prior_turn_reads() -> _StoredTurnReads:
    stored_turn: Final = {
        "request_id": _PRIOR_RESPONSE_ID,
        "session_id": "prior-session",
        "proxy_server_request": {"input": "What is LiteLLM?"},
        "response": {
            "id": _PRIOR_RESPONSE_ID,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "deepseek-v4-flash",
            "output": [
                {
                    "type": "message",
                    "id": "msg_prior",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": _PRIOR_ANSWER, "annotations": []}],
                }
            ],
        },
    }
    return _StoredTurnReads(
        StoredTurns(spend_logs=(stored_turn,), proxy_server_requests=(stored_turn["proxy_server_request"],))
    )


@pytest.mark.asyncio
async def test_a_sync_follow_up_made_inside_a_running_loop_runs_its_claimed_tool_with_the_injected_sources(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    provider: Final = _serve_responses(
        respx_mock,
        monkeypatch,
        (_tool_call_round(_DEEPSEEK_MODEL, "gitmcp-ask_question"), _answer_round(_DEEPSEEK_MODEL)),
        _DEEPSEEK_URL,
    )
    server: Final = _ScriptedMCPServer()
    reads: Final = _prior_turn_reads()

    with using_responses_dependencies(
        ResponsesDependencies(direct_client_factory=server.client_for, stored_turns_source=reads)
    ):
        response: Final = litellm.responses(
            model=_DEEPSEEK_MODEL,
            input="Who maintains it?",
            previous_response_id=_PRIOR_RESPONSE_ID,
            tools=[dict(_GITMCP_TOOL)],
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )

    assert isinstance(response, ResponsesAPIResponse)
    assert _output_types(response.output) == ("mcp_list_tools", "mcp_call", "message")
    assert [call.name for call in server.calls] == ["ask_question"]
    assert reads.asked == ((_PRIOR_RESPONSE_ID, (_PRIOR_RESPONSE_ID,)),)
    for index in range(provider.call_count):
        assert [_message_shape(message) for message in _sent_body(provider, index)["messages"]][:3] == [
            ("user", "What is LiteLLM?", ()),
            ("assistant", _PRIOR_ANSWER, ()),
            ("user", "Who maintains it?", ()),
        ]


class _DependenciesSeen:
    def __init__(self, rounds: Sequence[httpx.Response]) -> None:
        self._rounds: Final = iter(rounds)
        self.seen: tuple[ResponsesDependencies, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen = (*self.seen, responses_dependencies())
        return next(self._rounds)


@pytest.mark.asyncio
async def test_a_claimed_follow_up_keeps_no_request_state_in_the_dependencies_its_provider_calls_see(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    _serve_responses(respx_mock, monkeypatch, (), _DEEPSEEK_URL)
    provider: Final = _DependenciesSeen(
        (_tool_call_round(_DEEPSEEK_MODEL, "gitmcp-ask_question"), _answer_round(_DEEPSEEK_MODEL))
    )
    respx_mock.post(url__regex=_DEEPSEEK_URL).mock(side_effect=provider)
    server: Final = _ScriptedMCPServer()
    injected: Final = ResponsesDependencies(direct_client_factory=server.client_for, stored_turns_source=_prior_turn_reads())

    with using_responses_dependencies(injected):
        response: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input="Who maintains it?",
            previous_response_id=_PRIOR_RESPONSE_ID,
            tools=[dict(_GITMCP_TOOL)],
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )

    assert isinstance(response, ResponsesAPIResponse)
    assert [call.name for call in server.calls] == ["ask_question"]
    assert len(provider.seen) == 2
    assert all(seen is injected for seen in provider.seen)


def _streamed_answer_round_from(chat_id: str) -> httpx.Response:
    return _chat_stream(
        *(
            {**chunk, "id": chat_id}
            for chunk in (
                _chat_chunk({"role": "assistant", "content": _ASK_ANSWER}, None),
                _chat_chunk({}, "stop"),
            )
        )
    )


async def _streamed_turn_running_a_tool(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, server: _ScriptedMCPServer
) -> tuple[tuple[ResponsesAPIStreamingResponse, ...], respx.Route, StandardLoggingPayload]:
    turn: Final = await _streamed_turn(
        respx_mock,
        monkeypatch,
        server,
        (
            _streamed_tool_call_round("gitmcp-ask_question"),
            _streamed_answer_round_from("chatcmpl-gateway-mcp-answer"),
            _answer_round(_DEEPSEEK_MODEL),
        ),
    )
    assert [call.name for call in server.calls] == ["ask_question"]
    return turn


async def _streamed_turn(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    server: _ScriptedMCPServer,
    rounds: tuple[httpx.Response, ...],
) -> tuple[tuple[ResponsesAPIStreamingResponse, ...], respx.Route, StandardLoggingPayload]:
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(respx_mock, monkeypatch, rounds, _DEEPSEEK_URL)
    recorder: Final = _RoundRecorder()
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        stream: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input="Ask DeepWiki what LiteLLM is",
            tools=[dict(_GITMCP_TOOL)],
            stream=True,
            litellm_logging_obj=_rounds_logging_obj(recorder, "streamed-tool-turn"),
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )
    events: Final = tuple([event async for event in stream])
    assert _event_type(events[-1]) == "response.completed"
    await _settle(lambda: len(recorder.logged) > 0)
    assert len(recorder.logged) == 1
    return events, provider, recorder.logged[0]


def _message_shape(message: Mapping[str, object]) -> tuple[object, object, tuple[object, ...]]:
    calls: Final = message.get("tool_calls")
    return (
        message["role"],
        message.get("content"),
        tuple(call["function"]["name"] for call in calls) if isinstance(calls, list) else (),
    )


async def _follow_up_from_stored_row(
    monkeypatch: pytest.MonkeyPatch, server: _ScriptedMCPServer, response_id: str, row: StandardLoggingPayload
) -> ResponsesAPIResponse:
    stored_turn: Final = {
        "request_id": row["id"],
        "session_id": "streamed-tool-session",
        "proxy_server_request": {"input": "Ask DeepWiki what LiteLLM is"},
        "response": row["response"],
    }
    _proxy_database(monkeypatch, SimpleNamespace(db=_SpendLogs((stored_turn,))))
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        follow_up: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input="Who maintains it?",
            previous_response_id=response_id,
            tools=[dict(_GITMCP_TOOL)],
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )
    assert isinstance(follow_up, ResponsesAPIResponse)
    return follow_up


@pytest.mark.asyncio
async def test_a_streamed_turn_that_runs_a_tool_keeps_one_response_id_that_a_follow_up_continues_from(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    server: Final = _ScriptedMCPServer()
    events, provider, row = await _streamed_turn_running_a_tool(respx_mock, monkeypatch, server)
    response_id: Final = events[0].response.id

    assert {event.response.id for event in events if getattr(event, "response", None) is not None} == {response_id}
    assert row["id"] == response_id

    await _follow_up_from_stored_row(monkeypatch, server, response_id, row)

    replayed: Final = _sent_body(provider, 2)["messages"]
    assert [_message_shape(message) for message in replayed] == [
        ("user", "Ask DeepWiki what LiteLLM is", ()),
        ("assistant", None, ("gitmcp-ask_question",)),
        ("tool", _ASK_OUTPUT, ()),
        ("assistant", _ASK_ANSWER, ()),
        ("user", "Who maintains it?", ()),
    ]
    assert replayed[2]["tool_call_id"] == replayed[1]["tool_calls"][0]["id"]


@pytest.mark.asyncio
async def test_a_streamed_turn_that_runs_a_tool_stores_every_item_it_streamed_so_a_follow_up_lists_nothing_again(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    server: Final = _ScriptedMCPServer()
    events, _, row = await _streamed_turn_running_a_tool(respx_mock, monkeypatch, server)
    streamed: Final = tuple(_item(item) for item in events[-1].response.output)
    stored: Final = tuple(_item(item) for item in row["response"].get("output", ()))

    assert _output_summary(stored) == (
        ("mcp_list_tools", "gitmcp", None),
        ("mcp_call", "gitmcp", "ask_question"),
        ("message", None, None),
    )
    assert [_item_key(item) for item in stored] == [_item_key(item) for item in streamed]
    assert (stored[1]["arguments"], stored[1]["output"]) == (json.dumps(_ASK_ARGUMENTS), _ASK_OUTPUT)

    follow_up: Final = await _follow_up_from_stored_row(monkeypatch, server, events[0].response.id, row)

    assert _output_types(follow_up.output) == ("message",)


@pytest.mark.asyncio
async def test_a_streamed_turn_that_runs_no_tool_stores_its_listing_so_a_follow_up_lists_nothing_again(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    server: Final = _ScriptedMCPServer()
    events, _, row = await _streamed_turn(
        respx_mock,
        monkeypatch,
        server,
        (_streamed_answer_round_from("chatcmpl-gateway-mcp-answer"), _answer_round(_DEEPSEEK_MODEL)),
    )
    stored: Final = tuple(_item(item) for item in row["response"].get("output", ()))

    assert server.calls == ()
    assert _output_summary(stored) == (("mcp_list_tools", "gitmcp", None), ("message", None, None))

    follow_up: Final = await _follow_up_from_stored_row(monkeypatch, server, events[0].response.id, row)

    assert _output_types(follow_up.output) == ("message",)


@pytest.mark.asyncio
@respx.mock
async def test_a_held_gateway_reference_is_still_listed_as_the_access_check_but_gets_no_mcp_list_tools_item(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException

    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx.mock, monkeypatch, (_tool_call_round(_BEDROCK_MODEL, "deepwiki-ask_question"),), _BEDROCK_URL
    )
    server: Final = _ScriptedMCPServer()

    with pytest.raises(HTTPException) as denied:
        await _gateway_hosted_mcp_request(
            _BEDROCK_MODEL, server, _GITMCP_TOOL, beside=(_deepwiki_reference(),), caller=_key_holding("srv-other")
        )
    response: Final = await _gateway_hosted_mcp_request(
        _BEDROCK_MODEL, server, _GITMCP_TOOL, beside=(_deepwiki_reference(),), caller=_key_holding("srv-deepwiki")
    )

    assert denied.value.status_code == 403
    assert isinstance(response, ResponsesAPIResponse)
    assert _output_summary(response.output) == (
        ("mcp_list_tools", "gitmcp", None),
        ("message", None, None),
        ("function_call", None, "deepwiki-ask_question"),
    )
    assert mcp_client.list_tools_result.await_count == 1
    assert _sent_tool_names(_sent_body(provider, 0)) == ("deepwiki-ask_question", "gitmcp-ask_question")
    assert (mcp_client.call_tool.await_count, server.calls) == (0, ())


_GATEWAY_DEPENDENCY_NAMES: Final = (
    "stored_turns_source",
    "direct_client_factory",
    "hosted_mcp_claims",
    "direct_tools",
    "stream_shaper",
)


def _rounds_for(turn: str, stream: bool) -> tuple[httpx.Response, ...]:
    answer: Final = _streamed_answer_round() if stream else _answer_round(_DEEPSEEK_MODEL)
    if turn != "hosted_tool":
        return (answer,)
    tool_call: Final = (
        _streamed_tool_call_round("gitmcp-ask_question")
        if stream
        else _tool_call_round(_DEEPSEEK_MODEL, "gitmcp-ask_question")
    )
    return tool_call, answer


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("turn", ["first", "follow_up", "hosted_tool"])
async def test_a_body_key_named_like_a_gateway_dependency_reaches_the_provider_like_any_other_key(
    monkeypatch: pytest.MonkeyPatch, turn: str, stream: bool
):
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    provider: Final = _serve_responses(respx.mock, monkeypatch, _rounds_for(turn, stream), _DEEPSEEK_URL)
    server: Final = _ScriptedMCPServer()
    reads: Final = _prior_turn_reads()
    body_keys: Final = {name: f"body-{name}" for name in _GATEWAY_DEPENDENCY_NAMES}

    with using_responses_dependencies(
        ResponsesDependencies(direct_client_factory=server.client_for, stored_turns_source=reads)
    ):
        response, _ = await _client_response(
            await litellm.aresponses(
                model=_DEEPSEEK_MODEL,
                input="Ask DeepWiki what LiteLLM is",
                previous_response_id=_PRIOR_RESPONSE_ID if turn == "follow_up" else None,
                tools=[dict(_GITMCP_TOOL)] if turn == "hosted_tool" else None,
                stream=stream,
                **body_keys,
                **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
            )
        )

    sent: Final = tuple(_sent_body(provider, index) for index in range(provider.call_count))
    assert [{name: body.get(name) for name in _GATEWAY_DEPENDENCY_NAMES} for body in sent] == [body_keys] * len(
        _rounds_for(turn, stream)
    )
    assert [call.name for call in server.calls] == (["ask_question"] if turn == "hosted_tool" else [])
    assert reads.asked == (((_PRIOR_RESPONSE_ID, (_PRIOR_RESPONSE_ID,)),) if turn == "follow_up" else ())
    assert _output_types(response.output)[-1] == "message"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["caller_input", "names_non_string_fields", "mcp_call_names"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@respx.mock
async def test_a_body_key_named_like_a_bridge_transform_keyword_still_gets_the_providers_answer(
    monkeypatch: pytest.MonkeyPatch, name: str, stream: bool
):
    provider: Final = _serve_responses(respx.mock, monkeypatch, _rounds_for("first", stream), _DEEPSEEK_URL)

    response, _ = await _client_response(
        await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input="Say ok",
            stream=stream,
            **{name: {}},
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )
    )

    assert provider.call_count == 1
    assert _output_types(response.output) == ("message",)



async def _turn_running_a_hosted_tool(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, server: _ScriptedMCPServer, stream: bool
) -> tuple[str, _RoundRecorder]:
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    rounds: Final = (
        (_streamed_tool_call_round("gitmcp-ask_question"), _streamed_answer_round_from("chatcmpl-gateway-mcp-answer"))
        if stream
        else (_tool_call_round(_DEEPSEEK_MODEL, "gitmcp-ask_question"), _answer_round(_DEEPSEEK_MODEL))
    )
    _serve_responses(respx_mock, monkeypatch, (*rounds, _answer_round(_DEEPSEEK_MODEL)), _DEEPSEEK_URL)
    recorder: Final = _RoundRecorder()
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        response, _ = await _client_response(
            await litellm.aresponses(
                model=_DEEPSEEK_MODEL,
                input="Ask DeepWiki what LiteLLM is",
                tools=[dict(_GITMCP_TOOL)],
                stream=stream,
                litellm_logging_obj=_rounds_logging_obj(recorder, f"hosted-tool-turn-{stream}"),
                **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
            )
        )
    assert [call.name for call in server.calls] == ["ask_question"]
    await _settle(lambda: len(recorder.logged) > 0)
    assert len(recorder.logged) == 1
    return response.id, recorder



_TOOL_CONTENT: Final = (_ASK_OUTPUT, _ASK_ARGUMENTS["question"])


def _tool_content_in(record: object) -> tuple[str, ...]:
    text: Final = record if isinstance(record, str) else json.dumps(_item(record), default=str)
    return tuple(content for content in _TOOL_CONTENT if content in text)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_with_message_logging_off_a_turn_that_runs_a_hosted_tool_logs_and_stores_none_of_its_content(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.legacy_spend_log_rows import _get_response_for_spend_logs_payload

    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    monkeypatch.setitem(proxy_server.general_settings, "store_prompts_in_spend_logs", True)
    server: Final = _ScriptedMCPServer()

    _, recorder = await _turn_running_a_hosted_tool(respx_mock, monkeypatch, server, stream)

    logged: Final = recorder.logged[0]
    stored_row: Final = _get_response_for_spend_logs_payload(logged, {"litellm_params": {}})
    assert _output_summary(json.loads(stored_row)["output"]) == (
        ("mcp_list_tools", "gitmcp", None),
        ("mcp_call", "gitmcp", "ask_question"),
        ("message", None, None),
    )
    assert tuple(map(_tool_content_in, (logged, stored_row, recorder.results[0]))) == ((), (), ())


@pytest.mark.asyncio
async def test_a_follow_up_from_a_round_stored_with_message_logging_off_replays_every_call_and_checks_access(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from fastapi import HTTPException

    from litellm.constants import REDACTED_BY_LITELLM, REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER

    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    _gateway_runs_hosted_mcp(monkeypatch, ("gitmcp.io",))
    mcp_client: Final = _deepwiki_registered(monkeypatch)
    provider: Final = _serve_responses(
        respx_mock, monkeypatch, (_round_calling_both_servers(False), _answer_round(_DEEPSEEK_MODEL)), _DEEPSEEK_URL
    )
    server: Final = _ScriptedMCPServer()
    recorder: Final = _RoundRecorder()
    tools: Final = [dict(_GITMCP_TOOL), dict(_deepwiki_reference())]
    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        response: Final = await litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input="Ask both servers",
            tools=tools,
            litellm_metadata={"user_api_key_auth": _key_holding("srv-deepwiki")},
            litellm_logging_obj=_rounds_logging_obj(recorder, "redacted-mixed-round"),
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )
    await _settle(lambda: len(recorder.logged) > 0)
    stored_turn: Final = {
        "request_id": response.id,
        "session_id": "redacted-mixed-round-session",
        "proxy_server_request": {"input": "Ask both servers"},
        "response": recorder.logged[0]["response"],
    }
    _proxy_database(monkeypatch, SimpleNamespace(db=_SpendLogs((stored_turn,))))

    def follow_up_from(caller: object) -> Coroutine[object, object, object]:
        return litellm.aresponses(
            model=_DEEPSEEK_MODEL,
            input=[{"type": "function_call_output", "call_id": "call_wiki", "output": _HANDED_BACK_OUTPUT}],
            previous_response_id=response.id,
            tools=tools,
            litellm_metadata={"user_api_key_auth": caller},
            **_ROUTE_KWARGS[_DEEPSEEK_MODEL],
        )

    with using_responses_dependencies(ResponsesDependencies(direct_client_factory=server.client_for)):
        follow_up: Final = await follow_up_from(_key_holding("srv-deepwiki"))
        with pytest.raises(HTTPException) as denied:
            await follow_up_from(_key_holding("srv-other"))

    assert isinstance(follow_up, ResponsesAPIResponse)
    assert _output_types(follow_up.output) == ("message",)
    replayed: Final = _sent_body(provider, 1)["messages"]
    ran_call_id: Final = _item(response.output[3])["id"]
    assert [(call["id"], call["function"]["arguments"]) for call in replayed[1]["tool_calls"]] == [
        ("call_wiki", REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER),
        (ran_call_id, REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER),
    ]
    assert [(message["tool_call_id"], message["content"]) for message in replayed[2:]] == [
        (ran_call_id, REDACTED_BY_LITELLM),
        ("call_wiki", _HANDED_BACK_OUTPUT),
    ]
    assert mcp_client.list_tools_result.await_count == 2
    assert (denied.value.status_code, provider.call_count) == (403, 2)
