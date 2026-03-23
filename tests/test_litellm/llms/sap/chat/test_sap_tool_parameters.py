"""
Test SAP Tool Parameters Validation

This test ensures that tools passed to the SAP Orchestration Service
have `parameters.type` set to 'object' as required by the SAP API.

Issue: SAP GenAI Hub Orchestration Service rejects tools with:
"tools.0.custom.input_schema.type: Input should be 'object'"
"""

import pytest

from litellm.llms.sap.chat.models import FunctionTool, ChatCompletionTool


class TestFunctionToolParametersValidation:
    """Test that FunctionTool ensures parameters has type='object'."""

    def test_should_add_type_object_when_parameters_empty(self):
        """Empty parameters should get type='object' and properties={}."""
        tool = FunctionTool(name="test_tool", parameters={})
        
        assert tool.parameters.get("type") == "object"
        assert "properties" in tool.parameters

    def test_should_add_type_object_when_parameters_missing_type(self):
        """Parameters without type should get type='object' added."""
        tool = FunctionTool(
            name="test_tool",
            parameters={"properties": {"query": {"type": "string"}}}
        )
        
        assert tool.parameters.get("type") == "object"
        assert tool.parameters.get("properties") == {"query": {"type": "string"}}

    def test_should_preserve_existing_type_object(self):
        """Parameters with type='object' should be preserved."""
        tool = FunctionTool(
            name="test_tool",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        )
        
        assert tool.parameters.get("type") == "object"
        assert tool.parameters.get("properties") == {"query": {"type": "string"}}

    def test_should_add_properties_when_missing(self):
        """Parameters without properties should get properties={} added."""
        tool = FunctionTool(
            name="test_tool",
            parameters={"type": "object"}
        )
        
        assert tool.parameters.get("type") == "object"
        assert "properties" in tool.parameters

    def test_should_preserve_additional_schema_properties(self):
        """Additional JSON schema properties should be preserved."""
        tool = FunctionTool(
            name="test_tool",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False
            }
        )
        
        assert tool.parameters.get("type") == "object"
        assert tool.parameters.get("required") == ["query"]
        assert tool.parameters.get("additionalProperties") is False


class TestChatCompletionToolValidation:
    """Test that ChatCompletionTool correctly validates nested FunctionTool."""

    def test_should_validate_empty_parameters_in_function(self):
        """ChatCompletionTool with empty parameters should get type='object'."""
        tool_dict = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {}
            }
        }
        completion_tool = ChatCompletionTool(**tool_dict)
        
        assert completion_tool.function.parameters.get("type") == "object"
        assert "properties" in completion_tool.function.parameters

    def test_should_validate_missing_type_in_function_parameters(self):
        """ChatCompletionTool function parameters without type should get type='object'."""
        tool_dict = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    }
                }
            }
        }
        completion_tool = ChatCompletionTool(**tool_dict)
        
        assert completion_tool.function.parameters.get("type") == "object"


class TestToolTransformationIntegration:
    """Test the full tool transformation flow similar to SAP transformation.py."""

    def test_should_transform_openai_format_tool_correctly(self):
        """Simulate transformation.py tool validation flow."""
        from litellm.llms.sap.chat.transformation import validate_dict
        
        # OpenAI format tool with empty parameters (common case that was failing)
        openai_tool = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a web search"
            }
        }
        
        validated_tool = validate_dict(openai_tool, ChatCompletionTool)

        assert validated_tool["function"]["name"] == "web_search"


    def test_should_transform_tool_with_existing_parameters(self):
        """Tool with parameters should preserve them while ensuring type='object'."""
        from litellm.llms.sap.chat.transformation import validate_dict
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
        
        validated_tool = validate_dict(openai_tool, ChatCompletionTool)
        
        assert validated_tool["function"]["parameters"]["type"] == "object"
        assert "location" in validated_tool["function"]["parameters"]["properties"]
        assert validated_tool["function"]["parameters"]["required"] == ["location"]
