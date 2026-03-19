"""
Tests for $schema stripping in the Anthropic experimental pass-through
/v1/messages endpoint.

Addresses: https://github.com/BerriAI/litellm/issues/24121
"""

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


class TestStripSchemaFromTools:
    """Verify that ``$schema`` is removed from tool ``input_schema`` dicts."""

    def test_removes_dollar_schema_key(self):
        """$schema must be stripped from input_schema before forwarding."""
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {
                    "type": "object",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "properties": {
                        "param1": {"type": "string", "description": "A parameter"}
                    },
                    "required": ["param1"],
                    "additionalProperties": False,
                },
            }
        ]

        result = AnthropicMessagesConfig._strip_schema_from_tools(tools)

        assert len(result) == 1
        assert "$schema" not in result[0]["input_schema"]
        assert result[0]["input_schema"]["type"] == "object"
        assert "param1" in result[0]["input_schema"]["properties"]
        assert result[0]["input_schema"]["required"] == ["param1"]

    def test_does_not_mutate_original(self):
        """The original tools list and dicts must not be modified."""
        original_schema = {
            "type": "object",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"x": {"type": "string"}},
        }
        tools = [
            {
                "name": "tool",
                "description": "desc",
                "input_schema": original_schema,
            }
        ]

        result = AnthropicMessagesConfig._strip_schema_from_tools(tools)

        # Original must still have $schema
        assert "$schema" in tools[0]["input_schema"]
        # Result must not
        assert "$schema" not in result[0]["input_schema"]

    def test_no_schema_key_is_noop(self):
        """Tools without $schema should pass through unchanged."""
        tools = [
            {
                "name": "tool",
                "description": "desc",
                "input_schema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                },
            }
        ]

        result = AnthropicMessagesConfig._strip_schema_from_tools(tools)

        assert result == tools

    def test_multiple_tools(self):
        """All tools in the list should be sanitised."""
        tools = [
            {
                "name": "t1",
                "description": "d1",
                "input_schema": {
                    "type": "object",
                    "$schema": "http://example.com/schema",
                    "properties": {},
                },
            },
            {
                "name": "t2",
                "description": "d2",
                "input_schema": {
                    "type": "object",
                    "properties": {"b": {"type": "boolean"}},
                },
            },
            {
                "name": "t3",
                "description": "d3",
                "input_schema": {
                    "type": "object",
                    "$schema": "http://example.com/schema2",
                    "properties": {"c": {"type": "string"}},
                },
            },
        ]

        result = AnthropicMessagesConfig._strip_schema_from_tools(tools)

        assert len(result) == 3
        assert "$schema" not in result[0]["input_schema"]
        assert "$schema" not in result[2]["input_schema"]
        # Second tool had no $schema, should be unchanged
        assert result[1]["input_schema"] == tools[1]["input_schema"]

    def test_tool_without_input_schema(self):
        """Tools that lack input_schema entirely should not crash."""
        tools = [
            {"name": "bare_tool", "description": "no schema"},
        ]

        result = AnthropicMessagesConfig._strip_schema_from_tools(tools)

        assert result == tools

    def test_empty_tools_list(self):
        """An empty tools list should return an empty list."""
        assert AnthropicMessagesConfig._strip_schema_from_tools([]) == []
