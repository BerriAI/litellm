"""
Test for Anthropic tool type leakage into OpenAI parameters schema.

Regression test for issue #30557:
When an Anthropic tool has type: "custom", that top-level type was being
merged into the OpenAI function parameters schema, overwriting parameters.type
from "object" to "custom", causing validation errors from downstream providers.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.llms.anthropic import AnthropicMessagesRequest


def test_translate_anthropic_custom_tool_type_does_not_leak():
    """
    Anthropic tool with type: "custom" should not overwrite parameters.type.

    The top-level Anthropic tool "type" field must not be merged into the
    OpenAI function parameters schema. Parameters.type should remain "object"
    as defined in input_schema.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()

    tools = [
        {
            "type": "custom",
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    result, _ = adapter.translate_anthropic_tools_to_openai(tools=tools, model=None)

    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["description"] == "Get weather information"

    # The bug: parameters.type was being set to "custom" instead of "object"
    parameters = result[0]["function"]["parameters"]
    assert parameters["type"] == "object", (
        f"Expected parameters.type='object', got '{parameters['type']}'. "
        "Tool top-level type leaked into parameters schema."
    )
    assert parameters["properties"]["location"]["type"] == "string"
    assert parameters["required"] == ["location"]


def test_translate_anthropic_tool_without_type_field():
    """
    Tools without a type field should work correctly.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()

    tools = [{"name": "calculate", "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}}}]

    result, _ = adapter.translate_anthropic_tools_to_openai(tools=tools, model=None)

    assert len(result) == 1
    assert result[0]["function"]["parameters"]["type"] == "object"


def test_translate_anthropic_request_with_custom_tool():
    """
    End-to-end test: full Anthropic request translation with custom tool.

    Reproduces the exact scenario from issue #30557.
    """
    adapter = LiteLLMAnthropicMessagesAdapter()

    req = AnthropicMessagesRequest(
        model="gpt-4o",
        max_tokens=64,
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[
            {
                "type": "custom",
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ],
    )

    body, _ = adapter.translate_anthropic_to_openai(anthropic_message_request=req)

    assert "tools" in body
    assert len(body["tools"]) == 1

    tool_params = body["tools"][0]["function"]["parameters"]
    assert tool_params["type"] == "object", f"Expected object, got {tool_params['type']}. Reproduce issue #30557."
    assert "city" in tool_params["properties"]
