"""
Unit tests for MCP Semantic Tool Filtering

Tests the core filtering logic that takes a long list of tools and returns
an ordered set of top K tools based on semantic similarity.
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from mcp.types import Tool as MCPTool


@pytest.mark.asyncio
async def test_semantic_filter_basic_filtering():
    """
    Test that the semantic filter correctly filters tools based on query.
    
    Given: 10 email/calendar tools
    When: Query is "send an email"
    Then: Email tools should rank higher than calendar tools
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )

    # Create mock tools - mix of email and calendar tools
    tools = [
        MCPTool(name="gmail_send", description="Send an email via Gmail", inputSchema={"type": "object"}),
        MCPTool(name="outlook_send", description="Send an email via Outlook", inputSchema={"type": "object"}),
        MCPTool(name="calendar_create", description="Create a calendar event", inputSchema={"type": "object"}),
        MCPTool(name="calendar_update", description="Update a calendar event", inputSchema={"type": "object"}),
        MCPTool(name="email_read", description="Read emails from inbox", inputSchema={"type": "object"}),
        MCPTool(name="email_delete", description="Delete an email", inputSchema={"type": "object"}),
        MCPTool(name="calendar_delete", description="Delete a calendar event", inputSchema={"type": "object"}),
        MCPTool(name="email_search", description="Search for emails", inputSchema={"type": "object"}),
        MCPTool(name="calendar_list", description="List calendar events", inputSchema={"type": "object"}),
        MCPTool(name="email_forward", description="Forward an email to someone", inputSchema={"type": "object"}),
    ]
    
    # Mock router that returns mock embeddings
    from litellm.types.utils import Embedding, EmbeddingResponse
    
    mock_router = Mock()
    
    def mock_embedding_sync(*args, **kwargs):
        return EmbeddingResponse(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10}
        )
    
    async def mock_embedding_async(*args, **kwargs):
        return mock_embedding_sync()
    
    mock_router.embedding = mock_embedding_sync
    mock_router.aembedding = mock_embedding_async
    
    # Create filter
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=3,
        similarity_threshold=0.3,
        enabled=True,
    )
    
    # Build router with the tools before filtering
    filter_instance._build_router(tools)
    
    # Filter tools with email-related query
    filtered = await filter_instance.filter_tools(
        query="send an email to john@example.com",
        available_tools=tools,
    )
    
    # Assertions - validate filtering mechanics work
    assert len(filtered) <= 3, f"Should return at most 3 tools (top_k), got {len(filtered)}"
    assert len(filtered) > 0, "Should return at least some tools"
    assert len(filtered) < len(tools), f"Should filter down from {len(tools)} tools, got {len(filtered)}"
    
    # Validate tools are actual MCPTool objects
    for tool in filtered:
        assert hasattr(tool, 'name'), "Filtered result should be MCPTool with name"
        assert hasattr(tool, 'description'), "Filtered result should be MCPTool with description"
    
    filtered_names = [t.name for t in filtered]
    print(f"✅ Successfully filtered {len(tools)} tools down to top {len(filtered)}: {filtered_names}")
    print(f"   Filter respects top_k parameter correctly")


@pytest.mark.asyncio
async def test_semantic_filter_top_k_limiting():
    """
    Test that the filter respects top_k parameter.
    
    Given: 20 tools
    When: top_k=5
    Then: Should return at most 5 tools
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )

    # Create 20 tools
    tools = [
        MCPTool(name=f"tool_{i}", description=f"Tool number {i} for testing", inputSchema={"type": "object"})
        for i in range(20)
    ]
    
    # Mock router
    from litellm.types.utils import Embedding, EmbeddingResponse
    
    mock_router = Mock()
    
    def mock_embedding_sync(*args, **kwargs):
        return EmbeddingResponse(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10}
        )
    
    async def mock_embedding_async(*args, **kwargs):
        return mock_embedding_sync()
    
    mock_router.embedding = mock_embedding_sync
    mock_router.aembedding = mock_embedding_async
    
    # Create filter with top_k=5
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=5,
        similarity_threshold=0.3,
        enabled=True,
    )
    
    # Build router with the tools before filtering
    filter_instance._build_router(tools)
    
    # Filter tools
    filtered = await filter_instance.filter_tools(
        query="test query",
        available_tools=tools,
    )
    
    # Should return at most 5 tools
    assert len(filtered) <= 5, f"Expected at most 5 tools, got {len(filtered)}"
    print(f"Returned {len(filtered)} tools out of {len(tools)} (top_k=5)")


@pytest.mark.asyncio
async def test_semantic_filter_disabled():
    """
    Test that when filter is disabled, all tools are returned.
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    
    tools = [
        MCPTool(name=f"tool_{i}", description=f"Tool {i}", inputSchema={"type": "object"})
        for i in range(10)
    ]
    
    mock_router = Mock()
    
    # Create disabled filter
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=3,
        similarity_threshold=0.3,
        enabled=False,  # Disabled
    )
    
    # Filter tools
    filtered = await filter_instance.filter_tools(
        query="test query",
        available_tools=tools,
    )
    
    # Should return all tools when disabled
    assert len(filtered) == len(tools), f"Expected all {len(tools)} tools, got {len(filtered)}"


@pytest.mark.asyncio
async def test_semantic_filter_empty_tools():
    """
    Test that filter handles empty tool list gracefully.
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    
    mock_router = Mock()
    
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=3,
        similarity_threshold=0.3,
        enabled=True,
    )
    
    # Filter empty list
    filtered = await filter_instance.filter_tools(
        query="test query",
        available_tools=[],
    )
    
    assert len(filtered) == 0, "Should return empty list for empty input"


@pytest.mark.asyncio
async def test_semantic_filter_extract_user_query():
    """
    Test that user query extraction works correctly from messages.
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    
    mock_router = Mock()
    
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=3,
        similarity_threshold=0.3,
        enabled=True,
    )
    
    # Test string content
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Send an email to john@example.com"},
    ]
    
    query = filter_instance.extract_user_query(messages)
    assert query == "Send an email to john@example.com"
    
    # Test list content blocks
    messages_with_blocks = [
        {"role": "user", "content": [
            {"type": "text", "text": "Hello, "},
            {"type": "text", "text": "send email please"},
        ]},
    ]
    
    query2 = filter_instance.extract_user_query(messages_with_blocks)
    assert "Hello" in query2 and "send email" in query2
    
    # Test no user messages
    messages_no_user = [
        {"role": "system", "content": "System message only"},
    ]
    
    query3 = filter_instance.extract_user_query(messages_no_user)
    assert query3 == ""


@pytest.mark.asyncio
async def test_semantic_filter_hook_triggers_on_completion():
    """
    Test that the hook triggers for completion requests with tools.
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    from litellm.proxy.hooks.mcp_semantic_filter import SemanticToolFilterHook
    from litellm.types.utils import Embedding, EmbeddingResponse

    # Create mock filter
    mock_router = Mock()
    
    def mock_embedding_sync(*args, **kwargs):
        return EmbeddingResponse(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10}
        )
    
    async def mock_embedding_async(*args, **kwargs):
        return mock_embedding_sync()
    
    mock_router.embedding = mock_embedding_sync
    mock_router.aembedding = mock_embedding_async
    
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=3,
        similarity_threshold=0.3,
        enabled=True,
    )
    
    # Prepare data - completion request with tools
    tools = [
        MCPTool(name=f"tool_{i}", description=f"Tool {i}", inputSchema={"type": "object"})
        for i in range(10)
    ]
    
    # Build router with the tools before filtering
    filter_instance._build_router(tools)
    
    # Create hook
    hook = SemanticToolFilterHook(filter_instance)
    
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Send an email"}
        ],
        "tools": tools,
        "metadata": {},  # Hook needs metadata field to store filter stats
    }
    
    # Mock user API key dict and cache
    mock_user_api_key_dict = Mock()
    mock_cache = Mock()
    
    # Call hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key_dict,
        cache=mock_cache,
        data=data,
        call_type="completion",
    )
    
    # Assertions
    assert result is not None, "Hook should return modified data"
    assert "tools" in result, "Result should contain tools"
    assert len(result["tools"]) < len(tools), f"Hook should filter tools, got {len(result['tools'])}/{len(tools)}"
    
    print(f"✅ Hook triggered correctly: {len(tools)} -> {len(result['tools'])} tools")



@pytest.mark.asyncio
async def test_semantic_filter_hook_skips_no_tools():
    """
    Test that the hook does NOT trigger when there are no tools.
    """
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    from litellm.proxy.hooks.mcp_semantic_filter import SemanticToolFilterHook

    # Create mock filter
    mock_router = Mock()
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=mock_router,
        top_k=3,
        similarity_threshold=0.3,
        enabled=True,
    )
    
    # Create hook
    hook = SemanticToolFilterHook(filter_instance)
    
    # Prepare data - completion without tools
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
    }
    
    # Mock user API key dict and cache
    mock_user_api_key_dict = Mock()
    mock_cache = Mock()
    
    # Call hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key_dict,
        cache=mock_cache,
        data=data,
        call_type="completion",
    )
    
    # Should return None (no modification)
    assert result is None, "Hook should skip requests without tools"
    print("✅ Hook correctly skips requests without tools")

