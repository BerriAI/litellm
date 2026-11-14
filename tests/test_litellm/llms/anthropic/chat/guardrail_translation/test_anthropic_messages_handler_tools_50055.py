import copy
from unittest.mock import AsyncMock, MagicMock
import pytest

from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler


@pytest.mark.asyncio
async def test_anthropic_guardrail_preserves_unchanged_tools_50055():
    """
    Regression test for #50055:
    When a guardrail runs on /v1/messages and returns inputs unchanged,
    web_search_* server tools, mcp_toolset entries, and custom tool fields
    (such as long tool names and defer_loading) must not be dropped or
    rewritten into litellm_unnamed_tool_0.
    """
    handler = AnthropicMessagesHandler()
    original_tools = [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 1},
        {"type": "mcp_toolset", "mcp_server_name": "github"},
        {"name": "Read", "description": "d", "input_schema": {"type": "object", "properties": {}}},
        {
            "name": "mcp__atlassian__create_confluence_footer_comment_with_a_very_long_name_x",
            "description": "d",
            "defer_loading": True,
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    data = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": copy.deepcopy(original_tools),
    }

    guardrail = MagicMock()
    guardrail.guardrail_name = "passthrough"
    guardrail.should_skip_system_message_in_guardrail = MagicMock(return_value=False)
    guardrail.should_skip_tool_message_in_guardrail = MagicMock(return_value=False)
    guardrail.should_scan_only_tool_results = MagicMock(return_value=False)
    guardrail.structured_messages_cover_full_request = MagicMock(return_value=False)

    async def passthrough_apply(inputs, request_data, input_type, logging_obj=None):
        return copy.deepcopy(inputs)

    guardrail.apply_guardrail = AsyncMock(side_effect=passthrough_apply)

    result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)
    assert result["tools"] == original_tools
