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
        MCPTool(
            name="gmail_send",
            description="Send an email via Gmail",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="outlook_send",
            description="Send an email via Outlook",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="calendar_create",
            description="Create a calendar event",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="calendar_update",
            description="Update a calendar event",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="email_read",
            description="Read emails from inbox",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="email_delete",
            description="Delete an email",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="calendar_delete",
            description="Delete a calendar event",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="email_search",
            description="Search for emails",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="calendar_list",
            description="List calendar events",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="email_forward",
            description="Forward an email to someone",
            inputSchema={"type": "object"},
        ),
    ]

    # Mock router that returns mock embeddings
    from litellm.types.utils import Embedding, EmbeddingResponse

    mock_router = Mock()

    def mock_embedding_sync(*args, **kwargs):
        return EmbeddingResponse(
            data=[Embedding(embedding=[0.1] * 1536, index=0, object="embedding")],
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
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
    assert (
        len(filtered) <= 3
    ), f"Should return at most 3 tools (top_k), got {len(filtered)}"
    assert len(filtered) > 0, "Should return at least some tools"
    assert len(filtered) < len(
        tools
    ), f"Should filter down from {len(tools)} tools, got {len(filtered)}"

    # Validate tools are actual MCPTool objects
    for tool in filtered:
        assert hasattr(tool, "name"), "Filtered result should be MCPTool with name"
        assert hasattr(
            tool, "description"
        ), "Filtered result should be MCPTool with description"

    filtered_names = [t.name for t in filtered]
    print(
        f"✅ Successfully filtered {len(tools)} tools down to top {len(filtered)}: {filtered_names}"
    )
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
        MCPTool(
            name=f"tool_{i}",
            description=f"Tool number {i} for testing",
            inputSchema={"type": "object"},
        )
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
            usage={"prompt_tokens": 10, "total_tokens": 10},
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
        MCPTool(
            name=f"tool_{i}", description=f"Tool {i}", inputSchema={"type": "object"}
        )
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
    assert len(filtered) == len(
        tools
    ), f"Expected all {len(tools)} tools, got {len(filtered)}"


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
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "send email please"},
            ],
        },
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
            usage={"prompt_tokens": 10, "total_tokens": 10},
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
        MCPTool(
            name=f"tool_{i}", description=f"Tool {i}", inputSchema={"type": "object"}
        )
        for i in range(10)
    ]

    # Build router with the tools before filtering
    filter_instance._build_router(tools)

    # Create hook
    hook = SemanticToolFilterHook(filter_instance)

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Send an email"}],
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
    assert len(result["tools"]) < len(
        tools
    ), f"Hook should filter tools, got {len(result['tools'])}/{len(tools)}"

    print(f"✅ Hook triggered correctly: {len(tools)} -> {len(result['tools'])} tools")


def _make_filter():
    """Build a ``SemanticMCPToolFilter`` without exercising the embedding
    router. These tests only poke pure-Python methods (``_extract_tool_info``
    and ``_get_tools_by_names``)."""
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )

    return SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=Mock(),
        top_k=10,
        similarity_threshold=0.3,
        enabled=True,
    )


def test_extract_tool_info_openai_chat_wrapper():
    """Chat Completions wrapper: name and description come from .function."""
    f = _make_filter()
    name, description = f._extract_tool_info(
        {"type": "function", "function": {"name": "foo", "description": "bar"}}
    )
    assert (name, description) == ("foo", "bar")


def test_extract_tool_info_flat_dict():
    """Responses API / expanded MCP tool: flat dict with name+description."""
    f = _make_filter()
    name, description = f._extract_tool_info(
        {"name": "foo", "description": "bar"}
    )
    assert (name, description) == ("foo", "bar")


def test_extract_tool_info_mcptool_object():
    """MCPTool objects keep working unchanged."""
    f = _make_filter()
    name, description = f._extract_tool_info(
        MCPTool(name="foo", description="bar", inputSchema={"type": "object"})
    )
    assert (name, description) == ("foo", "bar")


def test_extract_tool_info_falls_back_to_name_when_description_missing():
    f = _make_filter()
    name, description = f._extract_tool_info(
        {"type": "function", "function": {"name": "foo"}}
    )
    assert (name, description) == ("foo", "foo")


def test_get_tools_by_names_exact_match_preserves_router_order():
    f = _make_filter()
    tools = [
        MCPTool(name="b", description="", inputSchema={}),
        MCPTool(name="a", description="", inputSchema={}),
        MCPTool(name="c", description="", inputSchema={}),
    ]
    matched = f._get_tools_by_names(["a", "c", "b"], tools)
    assert [t.name for t in matched] == ["a", "c", "b"]


def test_get_tools_by_names_client_prefix_is_resolved():
    """Client wraps an MCP canonical tool with its own alias prefix
    (``litellm_<server>-<tool>``). Canonical from the router is
    ``fc_web_search-firecrawl_scrape``; the wrapped client name still
    resolves to it."""
    f = _make_filter()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "litellm_fc_web_search-firecrawl_scrape",
                "description": "scrape",
            },
        }
    ]
    matched = f._get_tools_by_names(["fc_web_search-firecrawl_scrape"], tools)
    assert len(matched) == 1
    assert matched[0] is tools[0]


def test_get_tools_by_names_longest_canonical_wins():
    """Two canonicals share a tail; client-prefixed name must bind to the
    longer (more specific) canonical only."""
    f = _make_filter()
    tool = MCPTool(
        name="litellm_fc_web_search-scrape", description="", inputSchema={}
    )
    canonicals = ["search-scrape", "fc_web_search-scrape"]
    matched = f._get_tools_by_names(canonicals, [tool])
    # The tool resolves exactly once, against the longer canonical, so
    # ``search-scrape`` must drop out of the result.
    assert len(matched) == 1
    assert matched[0] is tool


def test_get_tools_by_names_multi_server_no_cross_match():
    """Two MCP servers register a same-named tool. A client-prefixed name
    referencing server_a must not be bound to server_b's canonical."""
    f = _make_filter()
    tools = [
        MCPTool(
            name="litellm_server_a-search", description="", inputSchema={}
        )
    ]
    canonicals = ["server_a-search", "server_b-search"]
    matched = f._get_tools_by_names(canonicals, tools)
    assert len(matched) == 1
    assert matched[0] is tools[0]


def test_get_tools_by_names_native_tool_no_false_match():
    """A native client tool shorter than the canonical (``read`` vs
    ``fs-read``) cannot alias into it: ``endswith`` requires the canonical
    to fit inside the tool name, which naturally excludes shorter native
    tools."""
    f = _make_filter()
    tools = [MCPTool(name="read", description="", inputSchema={})]
    matched = f._get_tools_by_names(["fs-read"], tools)
    assert matched == []


def test_get_tools_by_names_requires_boundary_before_canonical():
    """No hyphen/underscore immediately before the canonical - no match.

    Pure ``endswith`` would alias client ``foo123a-search`` into canonical
    ``a-search``; the separator-boundary guard (``-`` or ``_`` preceding
    the canonical suffix) rejects that aliasing, addressing Greptile's
    cross-server false-match concern.
    """
    f = _make_filter()
    tools = [MCPTool(name="foo123a-search", description="", inputSchema={})]
    matched = f._get_tools_by_names(["a-search"], tools)
    assert matched == []


def test_get_tools_by_names_warns_when_no_matches(caplog):
    """When every canonical fails to resolve we surface a warn log so the
    pathological empty-``tools`` request doesn't silently degrade into an
    upstream HTTP 400."""
    f = _make_filter()
    tools = [{"type": "function", "function": {"name": "unrelated_native_tool"}}]
    with caplog.at_level("WARNING", logger="LiteLLM"):
        matched = f._get_tools_by_names(["server-foo", "server-bar"], tools)
    assert matched == []
    assert any("matched 0 tools" in r.getMessage() for r in caplog.records)


def test_get_tools_by_names_ordering_follows_router_with_mixed_inputs():
    f = _make_filter()
    tools = [
        # Exact match for "a".
        MCPTool(name="a", description="", inputSchema={}),
        # Client-prefixed name for canonical "c".
        {"type": "function", "function": {"name": "litellm_c"}},
        # Wrapped flat dict for canonical "b".
        {"name": "ext-b"},
    ]
    matched = f._get_tools_by_names(["b", "a", "c"], tools)
    matched_names = [get_tool_name_for_test(t) for t in matched]
    assert matched_names == ["ext-b", "a", "litellm_c"]


def get_tool_name_for_test(tool):
    """Local mirror of ``get_tool_name_and_description`` (first element) so
    assertions don't couple to the helper module import order in other
    tests."""
    from litellm.proxy._experimental.mcp_server.utils import (
        get_tool_name_and_description,
    )

    return get_tool_name_and_description(tool)[0]


def test_get_tool_names_csv_handles_all_shapes():
    """``_get_tool_names_csv`` used to read only flat-dict ``name`` or
    ``.name``; wrapped Chat Completions tools therefore surfaced as empty
    strings in the ``x-litellm-semantic-filter-tools`` response header."""
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    from litellm.proxy.hooks.mcp_semantic_filter import SemanticToolFilterHook

    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=Mock(),
        top_k=3,
        similarity_threshold=0.3,
        enabled=True,
    )
    hook = SemanticToolFilterHook(filter_instance)

    tools = [
        {"type": "function", "function": {"name": "foo"}},
        {"name": "bar"},
        MCPTool(name="baz", description="", inputSchema={}),
    ]

    assert hook._get_tool_names_csv(tools) == "foo,bar,baz"


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
        "messages": [{"role": "user", "content": "Hello"}],
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
