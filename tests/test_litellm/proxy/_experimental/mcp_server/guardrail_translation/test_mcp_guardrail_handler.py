"""Tests for the MCP guardrail translation handler."""

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)


class MockGuardrail(CustomGuardrail):
    """Simple guardrail mock that records invocations."""

    def __init__(self):
        super().__init__(guardrail_name="mock-mcp-guardrail")
        self.call_count = 0
        self.last_inputs = None
        self.last_request_data = None

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.call_count += 1
        self.last_inputs = inputs
        self.last_request_data = request_data
        return None  # Guardrail doesn't modify for MCP tools


@pytest.mark.asyncio
async def test_process_input_messages_updates_content():
    """Handler should pass tool definition to guardrail when mcp_tool_name is present."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "mcp_tool_name": "weather",
        "mcp_arguments": {"city": "tokyo"},
        "mcp_tool_description": "Get weather for a city",
    }

    result = await handler.process_input_messages(data, guardrail)

    # Handler passes data through unchanged
    assert result == data
    # Guardrail was called
    assert guardrail.call_count == 1
    # Guardrail received tools (not texts) with tool definition
    assert guardrail.last_inputs is not None
    tools = guardrail.last_inputs.get("tools", [])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "weather"
    # Request data was passed to guardrail
    assert guardrail.last_request_data == data


@pytest.mark.asyncio
async def test_process_input_messages_skips_when_no_tool_name():
    """Handler should skip guardrail invocation if mcp_tool_name is missing."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    # No mcp_tool_name means nothing to process
    data = {"some_other_field": "value"}
    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 0


@pytest.mark.asyncio
async def test_process_input_messages_handles_minimal_data():
    """Handler should work with just mcp_tool_name (minimal required field)."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {"mcp_tool_name": "simple_tool"}

    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 1
    tools = guardrail.last_inputs.get("tools", [])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "simple_tool"
