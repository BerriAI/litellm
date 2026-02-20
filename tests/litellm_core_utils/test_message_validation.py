"""Tests for pre-flight message validation.

Tests validate detection of common issues that cause LLM API errors:
- Orphan tool calls (tool_call without response)
- Duplicate tool responses
- Invalid message sequencing
"""

import pytest

from litellm.litellm_core_utils.message_validation import (
    validate_messages,
    validate_responses_input,
)


class TestOpenAIValidation:
    """Test OpenAI Chat Completions message validation."""

    def test_valid_tool_call_sequence(self):
        """Valid: tool_call followed by matching tool response."""
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "file.txt"},
        ]
        errors = validate_messages(messages, provider="openai", tools_defined=True)
        assert errors == []

    def test_orphan_tool_call(self):
        """Invalid: tool_call without matching tool response."""
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_orphan",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "What happened?"},
        ]
        errors = validate_messages(messages, provider="openai", tools_defined=True)
        assert len(errors) >= 1
        assert any("unresolved" in e.lower() for e in errors)

    def test_duplicate_tool_response(self):
        """Invalid: multiple tool responses for same tool_call_id."""
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_dup", "type": "function", "function": {"name": "x", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_dup", "content": "first"},
            {"role": "tool", "tool_call_id": "call_dup", "content": "duplicate"},
        ]
        errors = validate_messages(messages, provider="openai", tools_defined=True)
        assert len(errors) >= 1
        assert any("duplicate" in e.lower() for e in errors)

    def test_orphan_tool_response(self):
        """Invalid: tool response without matching tool_call."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "unknown_id", "content": "result"},
        ]
        errors = validate_messages(messages, provider="openai", tools_defined=True)
        assert len(errors) >= 1
        assert any("unknown" in e.lower() for e in errors)

    def test_multiple_parallel_tool_calls(self):
        """Valid: multiple tool_calls in one message, all resolved."""
        messages = [
            {"role": "user", "content": "Run commands"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {"role": "tool", "tool_call_id": "c2", "content": "result2"},
        ]
        errors = validate_messages(messages, provider="openai", tools_defined=True)
        assert errors == []


class TestAnthropicValidation:
    """Test Anthropic Messages API validation."""

    def test_valid_tool_use_sequence(self):
        """Valid: tool_use followed by tool_result in next user message."""
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running command"},
                    {"type": "tool_use", "id": "tu_123", "name": "terminal", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_123", "content": "file.txt"},
                ],
            },
        ]
        errors = validate_messages(messages, provider="anthropic", tools_defined=True)
        assert errors == []

    def test_orphan_tool_use(self):
        """Invalid: tool_use without matching tool_result."""
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_orphan", "name": "terminal", "input": {}},
                ],
            },
            {"role": "user", "content": "What happened?"},
        ]
        errors = validate_messages(messages, provider="anthropic", tools_defined=True)
        assert len(errors) >= 1
        assert any("unresolved" in e.lower() for e in errors)

    def test_duplicate_tool_result(self):
        """Invalid: multiple tool_results for same tool_use_id."""
        messages = [
            {"role": "user", "content": "Run ls"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_dup", "name": "x", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_dup", "content": "first"},
                    {"type": "tool_result", "tool_use_id": "tu_dup", "content": "dup"},
                ],
            },
        ]
        errors = validate_messages(messages, provider="anthropic", tools_defined=True)
        assert len(errors) >= 1
        assert any("duplicate" in e.lower() for e in errors)


class TestResponsesAPIValidation:
    """Test OpenAI Responses API input validation."""

    def test_valid_function_call_sequence(self):
        """Valid: function_call followed by function_call_output."""
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Run ls"}],
            },
            {"type": "function_call", "call_id": "fc_123", "name": "terminal", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc_123", "output": "file.txt"},
        ]
        errors = validate_responses_input(input_items, tools_defined=True)
        assert errors == []

    def test_orphan_function_call(self):
        """Invalid: function_call without function_call_output."""
        input_items = [
            {"type": "function_call", "call_id": "fc_orphan", "name": "x", "arguments": "{}"},
        ]
        errors = validate_responses_input(input_items, tools_defined=True)
        assert len(errors) >= 1
        assert any("unresolved" in e.lower() for e in errors)

    def test_duplicate_function_output(self):
        """Invalid: multiple outputs for same call_id."""
        input_items = [
            {"type": "function_call", "call_id": "fc_dup", "name": "x", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc_dup", "output": "first"},
            {"type": "function_call_output", "call_id": "fc_dup", "output": "dup"},
        ]
        errors = validate_responses_input(input_items, tools_defined=True)
        assert len(errors) >= 1
        assert any("duplicate" in e.lower() for e in errors)


class TestAutoDetection:
    """Test automatic provider detection."""

    def test_detects_openai_format(self):
        """Detects OpenAI format from tool_calls structure."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "x"},
        ]
        errors = validate_messages(messages, provider="auto", tools_defined=True)
        # Should not error - valid OpenAI format
        assert errors == []

    def test_detects_anthropic_format(self):
        """Detects Anthropic format from tool_use/tool_result blocks."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu1", "name": "x", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "x"}],
            },
        ]
        errors = validate_messages(messages, provider="auto", tools_defined=True)
        # Should not error - valid Anthropic format
        assert errors == []
