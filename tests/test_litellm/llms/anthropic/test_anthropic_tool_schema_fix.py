"""
Test Anthropic Tool Schema Type Validation

This test ensures that translate_anthropic_tools_to_openai correctly adds
type='object' to input_schema when translating Anthropic tools to OpenAI format.

Issue: SAP GenAI Hub rejects tools missing type field in parameters.
The fix ensures Anthropic's input_schema gets type='object' added before
being translated to OpenAI's parameters field.
"""

import pytest
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)


class TestTranslateAnthropicToolsToOpenAI:
    """Test that translate_anthropic_tools_to_openai adds type='object' to input_schema."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance."""
        return LiteLLMAnthropicMessagesAdapter()

    def test_should_add_type_object_when_input_schema_missing_type(self, adapter):
        """Anthropic tool with input_schema missing type should get type='object' added."""
        anthropic_tools = [
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(anthropic_tools)

        # Verify tool was translated correctly
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "web_search"

        # Verify parameters has type='object' (critical for SAP compatibility)
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "query" in parameters["properties"]
        assert parameters["required"] == ["query"]

    def test_should_preserve_existing_type_object_in_input_schema(self, adapter):
        """Anthropic tool with input_schema already having type='object' should preserve it."""
        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]

        result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(anthropic_tools)

        # Verify tool maintains type='object'
        assert len(result) == 1
        tool = result[0]
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "location" in parameters["properties"]
        assert parameters["required"] == ["location"]

    def test_should_handle_empty_input_schema(self, adapter):
        """Anthropic tool with empty input_schema should get type='object' added."""
        anthropic_tools = [
            {
                "name": "no_params_tool",
                "description": "Tool without parameters",
                "input_schema": {}
            }
        ]

        result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(anthropic_tools)

        # Verify tool gets type='object' even with empty schema
        assert len(result) == 1
        tool = result[0]
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"

    def test_should_handle_multiple_tools(self, adapter):
        """Multiple Anthropic tools should all get type='object' if missing."""
        anthropic_tools = [
            {
                "name": "tool1",
                "description": "First tool",
                "input_schema": {
                    "properties": {
                        "param1": {"type": "string"}
                    }
                }
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "input_schema": {
                    "type": "object",  # Already has type
                    "properties": {
                        "param2": {"type": "number"}
                    }
                }
            }
        ]

        result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(anthropic_tools)

        # Verify both tools have type='object'
        assert len(result) == 2
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert result[1]["function"]["parameters"]["type"] == "object"

    def test_should_preserve_additional_schema_properties(self, adapter):
        """Additional JSON schema properties should be preserved during translation."""
        anthropic_tools = [
            {
                "name": "advanced_tool",
                "description": "Tool with complex schema",
                "input_schema": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 3
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        ]

        result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(anthropic_tools)

        # Verify all schema properties preserved + type='object' added
        assert len(result) == 1
        parameters = result[0]["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["required"] == ["query"]
        assert parameters["additionalProperties"] is False
        assert parameters["properties"]["query"]["minLength"] == 3


class TestSAPCompatibility:
    """Test scenarios specific to SAP GenAI Hub compatibility requirements."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance."""
        return LiteLLMAnthropicMessagesAdapter()

    def test_should_fix_claude_code_web_search_tool(self, adapter):
        """
        Simulate the exact scenario that was failing in Claude Code.
        Claude Code's web search tools were being rejected by SAP API.
        """
        # This is approximately what Claude Code sends (simplified)
        anthropic_tools = [
            {
                "name": "web_search",
                "description": "Perform a web search",
                "input_schema": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        result, tool_name_mapping = adapter.translate_anthropic_tools_to_openai(anthropic_tools)

        # SAP requires parameters.type to be 'object'
        # Without the fix, this would be missing and SAP would reject with:
        # "400 - LLM Module: tools.0.custom.input_schema.type: Input should be 'object'"
        assert len(result) == 1
        parameters = result[0]["function"]["parameters"]
        assert "type" in parameters, "Missing type field - SAP will reject this"
        assert parameters["type"] == "object", "Type must be 'object' for SAP compatibility"
        assert "properties" in parameters
        assert "query" in parameters["properties"]
