"""
End-to-end test for MCP Semantic Tool Filtering
"""
import asyncio
import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from mcp.types import Tool as MCPTool

# Check if semantic-router is available
try:
    import semantic_router
    SEMANTIC_ROUTER_AVAILABLE = True
except ImportError:
    SEMANTIC_ROUTER_AVAILABLE = False


@pytest.mark.asyncio
@pytest.mark.skipif(
    not SEMANTIC_ROUTER_AVAILABLE,
    reason="semantic-router not installed. Install with: pip install 'litellm[semantic-router]'"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set in environment"
)
async def test_e2e_semantic_filter():
    """E2E: Load router/filter and verify hook filters tools."""
    from litellm import Router
    from litellm.proxy.hooks.mcp_semantic_filter import SemanticToolFilterHook
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    
    # Create router and filter
    router = Router(
        model_list=[{
            "model_name": "text-embedding-3-small",
            "litellm_params": {"model": "openai/text-embedding-3-small"},
        }]
    )
    
    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=router,
        top_k=3,
        enabled=True,
    )
    
    # Create 10 tools
    tools = [
        MCPTool(name="gmail_send", description="Send an email via Gmail", inputSchema={"type": "object"}),
        MCPTool(name="calendar_create", description="Create a calendar event", inputSchema={"type": "object"}),
        MCPTool(name="file_upload", description="Upload a file", inputSchema={"type": "object"}),
        MCPTool(name="web_search", description="Search the web", inputSchema={"type": "object"}),
        MCPTool(name="slack_send", description="Send Slack message", inputSchema={"type": "object"}),
        MCPTool(name="doc_read", description="Read document", inputSchema={"type": "object"}),
        MCPTool(name="db_query", description="Query database", inputSchema={"type": "object"}),
        MCPTool(name="api_call", description="Make API call", inputSchema={"type": "object"}),
        MCPTool(name="task_create", description="Create task", inputSchema={"type": "object"}),
        MCPTool(name="note_add", description="Add note", inputSchema={"type": "object"}),
    ]
    
    # Build router with test tools
    filter_instance._build_router(tools)
    
    hook = SemanticToolFilterHook(filter_instance)
    
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Send an email and create a calendar event"}],
        "tools": tools,
        "metadata": {},  # Initialize metadata dict for hook to store filter stats
    }
    
    # Call hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=Mock(),
        cache=Mock(),
        data=data,
        call_type="completion",
    )

    # Single assertion: hook filtered tools
    assert result and len(result["tools"]) < len(tools), f"Expected filtered tools, got {len(result['tools'])} tools (original: {len(tools)})"
    
    print(f"✅ E2E test passed: Filtering reduced tools from {len(tools)} to {len(result['tools'])}")
    print(f"   Filtered tools: {[t.name for t in result['tools']]}")
