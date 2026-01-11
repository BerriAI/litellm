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

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.call_count += 1
        self.last_inputs = inputs

        if self.return_texts is not None:
            return {"texts": self.return_texts}

        texts = inputs.get("texts", [])
        return {"texts": [f"{text} [SAFE]" for text in texts]}


@pytest.mark.asyncio
async def test_process_input_messages_updates_content():
    """Handler should update the synthetic message content when guardrail modifies text."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    original_content = "Tool: weather\nArguments: {'city': 'tokyo'}"
    data = {
        "messages": [{"role": "user", "content": original_content}],
        "mcp_tool_name": "weather",
    }

    result = await handler.process_input_messages(data, guardrail)

    assert result["messages"][0]["content"].endswith("[SAFE]")
    assert guardrail.last_inputs == {"texts": [original_content]}
    assert guardrail.call_count == 1


@pytest.mark.asyncio
async def test_process_input_messages_skips_when_no_messages():
    """Handler should skip guardrail invocation if messages array is missing or empty."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {"mcp_tool_name": "noop"}
    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 0


@pytest.mark.asyncio
async def test_process_input_messages_handles_empty_guardrail_result():
    """Handler should leave content untouched when guardrail returns no text updates."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail(return_texts=[])

    original_content = "Tool: calendar\nArguments: {'date': '2024-12-25'}"
    data = {
        "messages": [{"role": "user", "content": original_content}],
        "mcp_tool_name": "calendar",
    }

    result = await handler.process_input_messages(data, guardrail)

    assert result["messages"][0]["content"] == original_content
    assert guardrail.call_count == 1
