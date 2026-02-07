"""Tests for the MCP guardrail translation handler."""

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)


class MockGuardrail(CustomGuardrail):
    """Simple guardrail mock that records invocations."""

    def __init__(self, return_texts=None):
        super().__init__(guardrail_name="mock-mcp-guardrail")
        self.return_texts = return_texts
        self.call_count = 0
        self.last_inputs = None
        self.last_request_data = None

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.call_count += 1
        self.last_inputs = inputs
        self.last_request_data = request_data

        if self.return_texts is not None:
            return {"texts": self.return_texts}

        # Return original inputs (no modification for tool-based guardrails)
        return inputs


@pytest.mark.asyncio
async def test_process_input_messages_calls_guardrail_with_tool():
    """Handler should call guardrail with tool definition when mcp_tool_name is present."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "mcp_tool_name": "weather",
        "mcp_arguments": {"city": "tokyo"},
        "mcp_tool_description": "Get weather for a city",
    }

    result = await handler.process_input_messages(data, guardrail)

    # Guardrail should be called once
    assert guardrail.call_count == 1

    # Guardrail should receive tool definition in inputs
    assert "tools" in guardrail.last_inputs
    assert len(guardrail.last_inputs["tools"]) == 1

    tool = guardrail.last_inputs["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "weather"
    assert tool["function"]["description"] == "Get weather for a city"

    # Request data should be passed through
    assert guardrail.last_request_data == data

    # Result should be the original data (unchanged)
    assert result == data


@pytest.mark.asyncio
async def test_process_input_messages_skips_when_no_tool_name():
    """Handler should skip guardrail invocation if mcp_tool_name is missing."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    # No mcp_tool_name in data - guardrail should not be called
    data = {"some_other_field": "value"}
    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 0


@pytest.mark.asyncio
async def test_process_input_messages_handles_name_alias():
    """Handler should accept 'name' as an alias for 'mcp_tool_name'."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "name": "calendar",
        "arguments": {"date": "2024-12-25"},
    }

    result = await handler.process_input_messages(data, guardrail)

    assert guardrail.call_count == 1
    assert guardrail.last_inputs["tools"][0]["function"]["name"] == "calendar"


@pytest.mark.asyncio
async def test_process_input_messages_handles_missing_arguments():
    """Handler should handle missing mcp_arguments gracefully."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "mcp_tool_name": "simple_tool",
        # No mcp_arguments provided
    }

    result = await handler.process_input_messages(data, guardrail)

    assert guardrail.call_count == 1
    assert guardrail.last_inputs["tools"][0]["function"]["name"] == "simple_tool"
