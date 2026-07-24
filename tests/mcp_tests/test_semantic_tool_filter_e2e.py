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
    reason="semantic-router not installed. Install the `litellm[semantic-router]` extra.",
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set in environment"
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
        model_list=[
            {
                "model_name": "text-embedding-3-small",
                "litellm_params": {"model": "openai/text-embedding-3-small"},
            }
        ]
    )

    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=router,
        top_k=3,
        enabled=True,
    )

    # Create 10 tools
    tools = [
        MCPTool(
            name="gmail_send",
            description="Send an email via Gmail",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="calendar_create",
            description="Create a calendar event",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="file_upload",
            description="Upload a file",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="web_search",
            description="Search the web",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="slack_send",
            description="Send Slack message",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="doc_read", description="Read document", inputSchema={"type": "object"}
        ),
        MCPTool(
            name="db_query",
            description="Query database",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="api_call", description="Make API call", inputSchema={"type": "object"}
        ),
        MCPTool(
            name="task_create",
            description="Create task",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="note_add", description="Add note", inputSchema={"type": "object"}
        ),
    ]

    # Build router with test tools
    filter_instance._build_router(tools)

    hook = SemanticToolFilterHook(filter_instance)

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Send an email and create a calendar event"}
        ],
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
    assert result and len(result["tools"]) < len(
        tools
    ), f"Expected filtered tools, got {len(result['tools'])} tools (original: {len(tools)})"

    print(
        f"✅ E2E test passed: Filtering reduced tools from {len(tools)} to {len(result['tools'])}"
    )
    print(f"   Filtered tools: {[t.name for t in result['tools']]}")


@pytest.mark.asyncio
async def test_expand_mcp_tools_uses_chat_format():
    """_expand_mcp_tools must request the Chat Completions tool shape
    ({"type": "function", "function": {...}}), not the flat Responses API
    shape, since it feeds /v1/chat/completions providers with strict
    schema validation (e.g. hosted_vllm). Regression test for #32281.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.hooks.mcp_semantic_filter.hook import SemanticToolFilterHook

    fake_tools = [
        MCPTool(
            name="matomo-matomo_site_list",
            description="List sites",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    with patch(
        "litellm.responses.mcp.litellm_proxy_mcp_handler.LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager",
        new=AsyncMock(return_value=(fake_tools, ["matomo"])),
    ):
        hook = SemanticToolFilterHook(Mock())
        result = await hook._expand_mcp_tools(
            tools=[
                {
                    "type": "mcp",
                    "server_url": "litellm_proxy/mcp/matomo",
                    "server_label": "matomo_mcp",
                    "require_approval": "never",
                }
            ],
            user_api_key_dict=Mock(),
        )

    assert len(result) == 1
    tool = result[0]
    assert tool["type"] == "function"
    assert "function" in tool, (
        "Expected the Chat Completions wrapper shape "
        f"{{'type': 'function', 'function': {{...}}}}, got flat keys: {list(tool.keys())}"
    )
    assert tool["function"]["name"] == "matomo-matomo_site_list"
