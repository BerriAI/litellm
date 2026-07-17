import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages_handler,
)
from litellm.llms.anthropic.experimental_pass_through.messages.mcp_handler import (
    _build_tool_result_message,
    _extract_tool_use_blocks,
)

MCP_REFERENCE = {
    "type": "mcp",
    "server_label": "litellm",
    "server_url": "litellm_proxy/mcp/deepwiki",
    "require_approval": "never",
}


def test_anthropic_messages_handler_routes_litellm_proxy_mcp_to_the_gateway():
    """
    Regression test (LIT-4517): /v1/messages must expand a litellm_proxy MCP
    reference through the MCP gateway.

    Given: A /v1/messages request whose tools carry a litellm_proxy MCP reference
    When:  The handler dispatches
    Then:  It hands off to the MCP gateway instead of the provider

    Without this hook the reference is forwarded to Anthropic verbatim and the API
    rejects the request ("Input tag 'mcp' found using 'type' does not match any of
    the expected tags"), because only /v1/chat/completions and /v1/responses ever
    had a gateway entry point. This pins the wiring, not the helper: deleting the
    dispatch makes the whole feature unreachable while every unit test still passes.
    """
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.mcp_handler.anthropic_messages_with_mcp",
        new=AsyncMock(return_value={"routed": True}),
    ) as routed:
        result = anthropic_messages_handler(
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5",
            tools=[MCP_REFERENCE],
            custom_llm_provider="anthropic",
        )

    assert routed.called, "A litellm_proxy MCP reference must be dispatched to the MCP gateway"
    assert routed.call_args.kwargs["tools"] == [MCP_REFERENCE]
    assert routed.call_args.kwargs["model"] == "claude-sonnet-4-5"
    assert result is not None


def test_anthropic_messages_handler_skips_the_gateway_on_recursion():
    """The gateway's own follow-up call must not re-enter the gateway."""
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.mcp_handler.anthropic_messages_with_mcp",
        new=AsyncMock(return_value={"routed": True}),
    ) as routed:
        with pytest.raises(Exception):
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4-5",
                tools=[MCP_REFERENCE],
                custom_llm_provider="anthropic",
                _skip_mcp_handler=True,
            )

    assert not routed.called, "_skip_mcp_handler must stop the gateway from recursing"


def test_anthropic_messages_handler_leaves_native_tools_alone():
    """A plain Anthropic tool is not an MCP reference and must not reach the gateway."""
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.mcp_handler.anthropic_messages_with_mcp",
        new=AsyncMock(return_value={"routed": True}),
    ) as routed:
        with pytest.raises(Exception):
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4-5",
                tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
                custom_llm_provider="anthropic",
            )

    assert not routed.called, "Only litellm_proxy MCP references belong to the gateway"


def test_extract_tool_use_blocks_ignores_text_blocks():
    """Only tool_use blocks drive the loop; text blocks are the model's prose."""
    response = {
        "content": [
            {"type": "text", "text": "let me look that up"},
            {"type": "tool_use", "id": "toolu_1", "name": "read_wiki_structure", "input": {"repoName": "a/b"}},
        ]
    }

    blocks = _extract_tool_use_blocks(response)

    assert len(blocks) == 1
    assert blocks[0]["name"] == "read_wiki_structure"


def test_build_tool_result_message_uses_anthropic_tool_result_blocks():
    """
    Results must go back as tool_result blocks in a user message.

    Anthropic pairs each result to its request by tool_use_id; the OpenAI shape
    (a role="tool" message keyed by tool_call_id) is rejected here.
    """
    message = _build_tool_result_message([{"tool_call_id": "toolu_1", "result": "9 sections", "name": "read_wiki"}])

    assert message["role"] == "user"
    assert list(message["content"]) == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "9 sections"}
    ]
