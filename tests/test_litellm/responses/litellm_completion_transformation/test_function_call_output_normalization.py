"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

import json
from typing import Final

from openai.types.responses import ResponseFunctionToolCall

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)


def test_function_call_output_list_input_text_is_converted_to_tool_string_content():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": " world"},
            ],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "hello world"


def test_function_call_output_string_passthrough():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        }
    )
    assert len(out) == 1
    assert out[0]["content"] == '{"ok":true}'


def test_multi_tool_use_parallel_expanded_in_responses_tools():
    tool_call: Final = ChatCompletionMessageToolCall(
        id="call_azure_123",
        type="function",
        function=Function(
            name="multi_tool_use.parallel",
            arguments=json.dumps(
                {  # mutable-ok: test payload
                    "tool_uses": [
                        {
                            "recipient_name": "functions.zoekt_search",
                            "parameters": {"query": "litellm"},
                        },
                        {
                            "recipient_name": "functions.file_lookup",
                            "parameters": {"path": "README.md"},
                        },
                    ]
                }
            ),
        ),
    )
    response: Final = ModelResponse(
        id="test_resp",
        choices=[
            Choices(index=0, message=Message(content=None, role="assistant", tool_calls=[tool_call]))
        ],  # mutable-ok: test payload
        created=1234567890,
        model="azure/gpt-4o",
        object="chat.completion",
    )

    result: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(response)

    assert len(result) == 2
    assert isinstance(result[0], ResponseFunctionToolCall)
    assert result[0].name == "zoekt_search"
    assert result[0].id == "call_azure_123_0"
    assert result[0].call_id == "call_azure_123_0"
    assert json.loads(result[0].arguments) == {"query": "litellm"}  # mutable-ok: comparison

    assert isinstance(result[1], ResponseFunctionToolCall)
    assert result[1].name == "file_lookup"
    assert result[1].id == "call_azure_123_1"
    assert result[1].call_id == "call_azure_123_1"
    assert json.loads(result[1].arguments) == {"path": "README.md"}  # mutable-ok: comparison


def test_multi_tool_use_parallel_invalid_json_fallback():
    tool_call: Final = ChatCompletionMessageToolCall(
        id="call_azure_123",
        type="function",
        function=Function(
            name="multi_tool_use.parallel",
            arguments="invalid-json-content",
        ),
    )
    response: Final = ModelResponse(
        id="test_resp",
        choices=[
            Choices(index=0, message=Message(content=None, role="assistant", tool_calls=[tool_call]))
        ],  # mutable-ok: test payload
        created=1234567890,
        model="azure/gpt-4o",
        object="chat.completion",
    )

    result: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(response)

    assert len(result) == 1
    assert isinstance(result[0], ResponseFunctionToolCall)
    assert result[0].name == "multi_tool_use.parallel"
    assert result[0].id == "call_azure_123"


def test_multi_tool_use_parallel_malformed_recipient_fallback():
    tool_call: Final = ChatCompletionMessageToolCall(
        id="call_azure_123",
        type="function",
        function=Function(
            name="multi_tool_use.parallel",
            arguments=json.dumps(
                {  # mutable-ok: test payload
                    "tool_uses": [
                        {
                            "recipient_name": None,
                            "parameters": {"query": "litellm"},
                        }
                    ]
                }
            ),
        ),
    )
    response: Final = ModelResponse(
        id="test_resp",
        choices=[
            Choices(index=0, message=Message(content=None, role="assistant", tool_calls=[tool_call]))
        ],  # mutable-ok: test payload
        created=1234567890,
        model="azure/gpt-4o",
        object="chat.completion",
    )

    result: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(response)

    assert len(result) == 1
    assert isinstance(result[0], ResponseFunctionToolCall)
    assert result[0].name == "multi_tool_use.parallel"
    assert result[0].id == "call_azure_123"
