"""
Test custom_tool_call adaptation for apply_patch and other custom tools.

This test verifies that when Codex sends custom tools (type="custom"),
LiteLLM bridge correctly:
1. Converts them to function tools for Chat Completions providers
2. Converts function_call responses back to custom_tool_call output items
3. Unwraps the JSON-wrapping arguments to extract the actual input content
"""

import json
import pytest
from typing import Dict, Any, List

from openai.types.responses import ResponseFunctionToolCall

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.litellm_completion_transformation.custom_tools import (
    extract_custom_tool_names,
    is_custom_tool_call,
    unwrap_custom_tool_arguments,
    build_tool_call_item_kwargs,
    convert_custom_tool_to_function_tool,
    _MAX_ARGUMENTS_LEN,
)

from litellm.types.responses.main import CustomToolCallOutputItem


class TestCustomToolUtilities:
    """Test the custom_tools utility functions."""

    def test_extract_custom_tool_names(self):
        """Test extraction of custom tool names from tools list."""
        tools = [
            {"type": "function", "name": "regular_tool"},
            {"type": "custom", "name": "apply_patch"},
            {"type": "function", "name": "another_tool"},
            {"type": "custom", "name": "custom_format"},
        ]

        names = extract_custom_tool_names(tools)
        assert names == {"apply_patch", "custom_format"}

    def test_extract_custom_tool_names_empty(self):
        """Test extraction with no custom tools."""
        tools = [
            {"type": "function", "name": "tool1"},
            {"type": "function", "name": "tool2"},
        ]

        names = extract_custom_tool_names(tools)
        assert names == set()

    def test_extract_custom_tool_names_none(self):
        """Test extraction with None input."""
        names = extract_custom_tool_names(None)
        assert names == set()

    def test_is_custom_tool_call_true(self):
        """Test identification of custom tool call."""
        custom_names = {"apply_patch", "custom_format"}
        assert is_custom_tool_call("apply_patch", custom_names) is True
        assert is_custom_tool_call("custom_format", custom_names) is True

    def test_is_custom_tool_call_false(self):
        """Test identification of non-custom tool call."""
        custom_names = {"apply_patch"}
        assert is_custom_tool_call("regular_tool", custom_names) is False
        assert is_custom_tool_call("unknown_tool", custom_names) is False

    def test_unwrap_custom_tool_arguments(self):
        """Test unwrapping of JSON-wrapped arguments."""
        # Test with valid JSON
        wrapped = json.dumps({"content": "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"})
        unwrapped = unwrap_custom_tool_arguments(wrapped)
        assert unwrapped == "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"

    def test_unwrap_custom_tool_arguments_invalid_json(self):
        """Test unwrapping with invalid JSON returns original."""
        raw = "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"
        unwrapped = unwrap_custom_tool_arguments(raw)
        assert unwrapped == raw

    def test_unwrap_custom_tool_arguments_no_content_key(self):
        """Test unwrapping with JSON but no content key."""
        wrapped = json.dumps({"other_key": "value"})
        unwrapped = unwrap_custom_tool_arguments(wrapped)
        assert unwrapped == wrapped

    def test_build_tool_call_item_kwargs_custom_completed(self):
        """A completed custom tool call unwraps the content into `input`."""
        wrapped = json.dumps({"content": "patch body"})
        kwargs = build_tool_call_item_kwargs(
            call_id="c1",
            name="apply_patch",
            arguments_or_input=wrapped,
            status="completed",
            custom_tool_names={"apply_patch"},
        )
        assert kwargs["type"] == "custom_tool_call"
        assert kwargs["input"] == "patch body"
        assert "arguments" not in kwargs

    def test_build_tool_call_item_kwargs_custom_in_progress(self):
        """An in-progress custom tool call seeds an empty input string."""
        kwargs = build_tool_call_item_kwargs(
            call_id="c2",
            name="apply_patch",
            arguments_or_input="ignored-until-completed",
            status="in_progress",
            custom_tool_names={"apply_patch"},
        )
        assert kwargs["input"] == ""

    def test_build_tool_call_item_kwargs_regular_function(self):
        """A regular function call keeps raw arguments and uses function_call type."""
        raw = json.dumps({"k": "v"})
        kwargs = build_tool_call_item_kwargs(
            call_id="c3",
            name="get_weather",
            arguments_or_input=raw,
            status="completed",
            custom_tool_names=set(),
        )
        assert kwargs["type"] == "function_call"
        assert kwargs["arguments"] == raw
        assert "input" not in kwargs

    def test_unwrap_custom_tool_arguments_oversized_returns_raw(self):
        """Arguments larger than the safety cap are returned unchanged to avoid
        OOM on JSON parsing a pathologically large string."""
        oversized = "x" * (_MAX_ARGUMENTS_LEN + 1)
        assert unwrap_custom_tool_arguments(oversized) == oversized

    def test_unwrap_custom_tool_arguments_empty(self):
        """Empty arguments unwrap to an empty string, not the raw input."""
        assert unwrap_custom_tool_arguments("") == ""

    def test_convert_custom_tool_to_function_tool_with_format(self):
        """The grammar definition is embedded in the description so the model can
        produce correctly-formatted output."""
        tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: begin_patch",
            },
        }
        result = convert_custom_tool_to_function_tool(tool)
        assert result is not None
        assert result["type"] == "function"
        assert "begin_patch" in result["function"]["description"]
        assert result["function"]["parameters"]["required"] == ["content"]

    def test_convert_custom_tool_to_function_tool_non_custom_returns_none(self):
        """Non-custom tools are not convertible; the caller keeps them as-is."""
        assert convert_custom_tool_to_function_tool({"type": "function"}) is None


class TestTransformationCustomTools:
    """Test custom tool handling in transformation logic."""

    def test_transform_apply_patch_function_call_to_custom_tool_call(self):
        """Test that apply_patch function_call is converted to custom_tool_call."""
        # Simulate a Chat Completion response with apply_patch function call
        from litellm.types.utils import ModelResponse, Choices, Message, ChatCompletionMessageToolCall, Function

        tool_call = ChatCompletionMessageToolCall(
            id="call_abc123",
            type="function",
            function=Function(
                name="apply_patch",
                arguments=json.dumps({"content": "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"}),
            ),
        )

        message = Message(role="assistant", content=None, tool_calls=[tool_call])

        choices = [Choices(index=0, message=message, finish_reason="tool_calls")]

        response = ModelResponse(
            id="test_response", choices=choices, created=1234567890, model="gpt-4", object="chat.completion"
        )

        # Transform with custom tool names
        responses_api_request = {
            "tools": [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "regular_tool"}]
        }

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            response, responses_api_request=responses_api_request
        )

        # Should return a CustomToolCallOutputItem object; ResponsesAPIResponse
        # accepts it directly via its output item union.
        assert len(result) == 1
        item = result[0]
        assert isinstance(item, CustomToolCallOutputItem)
        assert item.type == "custom_tool_call"
        assert item.call_id == "call_abc123"
        assert item.name == "apply_patch"
        assert item.input == "*** Begin Patch\n*** Add File: test.py\n+hello\n*** End Patch"
        assert item.status == "completed"

    def test_custom_tool_call_input_item_recovers_payload_from_input(self):
        """A custom_tool_call input item stores its payload in `input`; the
        assistant tool call must carry it as a JSON content envelope whether
        `arguments` is missing or an empty string."""
        for arguments in (None, ""):
            item = {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n+hello\n*** End Patch",
            }
            if arguments is not None:
                item["arguments"] = arguments
            messages = (
                LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
                    function_call=item
                )
            )
            tool_call = messages[0]["tool_calls"][0]
            assert tool_call["function"]["arguments"] == json.dumps(
                {"content": "*** Begin Patch\n+hello\n*** End Patch"}
            )

    def test_function_call_input_item_with_empty_arguments_keeps_them_empty(self):
        """A plain function_call input item with empty or missing `arguments`
        must produce an empty arguments string, never a `{"content": ...}`
        envelope (that recovery is reserved for custom_tool_call items) and
        never the literal string "None"."""
        for item in (
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "get_weather",
                "arguments": "",
                "input": "stray value",
            },
            {
                "type": "function_call",
                "call_id": "call_3",
                "name": "get_weather",
            },
        ):
            messages = (
                LiteLLMCompletionResponsesConfig._transform_responses_api_function_call_to_chat_completion_message(
                    function_call=item
                )
            )
            tool_call = messages[0]["tool_calls"][0]
            assert tool_call["function"]["arguments"] == ""

    def test_transform_regular_function_call_unchanged(self):
        """Test that regular function calls remain as ResponseFunctionToolCall."""
        from litellm.types.utils import ModelResponse, Choices, Message, ChatCompletionMessageToolCall, Function

        tool_call = ChatCompletionMessageToolCall(
            id="call_xyz789",
            type="function",
            function=Function(name="regular_tool", arguments=json.dumps({"param": "value"})),
        )

        message = Message(role="assistant", content=None, tool_calls=[tool_call])

        choices = [Choices(index=0, message=message, finish_reason="tool_calls")]

        response = ModelResponse(
            id="test_response", choices=choices, created=1234567890, model="gpt-4", object="chat.completion"
        )

        # Transform with custom tool names (regular_tool is NOT custom)
        responses_api_request = {
            "tools": [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "regular_tool"}]
        }

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            response, responses_api_request=responses_api_request
        )

        # Should return ResponseFunctionToolCall
        assert len(result) == 1
        item = result[0]
        assert isinstance(item, ResponseFunctionToolCall)
        assert item.type == "function_call"
        assert item.name == "regular_tool"
        assert item.arguments == json.dumps({"param": "value"})

    def test_transform_mixed_tool_calls(self):
        """Test transformation with both custom and regular tool calls."""
        from litellm.types.utils import ModelResponse, Choices, Message, ChatCompletionMessageToolCall, Function

        custom_call = ChatCompletionMessageToolCall(
            id="call_001",
            type="function",
            function=Function(name="apply_patch", arguments=json.dumps({"content": "patch content"})),
        )

        regular_call = ChatCompletionMessageToolCall(
            id="call_002", type="function", function=Function(name="get_weather", arguments=json.dumps({"city": "SF"}))
        )

        message = Message(role="assistant", content=None, tool_calls=[custom_call, regular_call])

        choices = [Choices(index=0, message=message, finish_reason="tool_calls")]

        response = ModelResponse(
            id="test_response", choices=choices, created=1234567890, model="gpt-4", object="chat.completion"
        )

        responses_api_request = {
            "tools": [{"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "get_weather"}]
        }

        result = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
            response, responses_api_request=responses_api_request
        )

        assert len(result) == 2

        # First should be custom_tool_call object
        first = result[0]
        assert isinstance(first, CustomToolCallOutputItem)
        assert first.type == "custom_tool_call"
        assert first.name == "apply_patch"
        assert first.input == "patch content"

        # Second should be function_call
        second = result[1]
        assert isinstance(second, ResponseFunctionToolCall)
        assert second.type == "function_call"
        assert second.name == "get_weather"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
