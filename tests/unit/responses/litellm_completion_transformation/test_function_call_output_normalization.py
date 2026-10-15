"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

from collections.abc import Mapping
from typing import Final, cast

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_tool_result
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ChatCompletionToolMessage, ToolMessageContentPart


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


@pytest.mark.parametrize(
    ("output", "expected_content"),
    [
        (
            [{"type": "tool_reference", "tool_name": "get_weather"}],
            [{"type": "tool_reference", "tool_name": "get_weather"}],
        ),
        (
            [
                {"type": "input_text", "text": "Loaded tools"},
                {"type": "tool_reference", "tool_name": "get_weather"},
                {"type": "input_text", "text": " and "},
                {"type": "tool_reference", "tool_name": "get_forecast"},
            ],
            [
                {"type": "text", "text": "Loaded tools"},
                {"type": "tool_reference", "tool_name": "get_weather"},
                {"type": "text", "text": " and "},
                {"type": "tool_reference", "tool_name": "get_forecast"},
            ],
        ),
    ],
    ids=["reference-only", "interleaved-text-and-references"],
)
def test_function_call_output_preserves_tool_references_for_anthropic(
    output: list[Mapping[str, object]], expected_content: list[ToolMessageContentPart]
) -> None:
    messages: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
            {"type": "function_call_output", "call_id": "call_load_tools", "output": output}
        )
    )

    assert messages == [{"role": "tool", "tool_call_id": "call_load_tools", "content": expected_content}]
    message: Final = cast(ChatCompletionToolMessage, messages[0])  # cast-ok: complete tool message asserted above
    assert convert_to_anthropic_tool_result(message) == {
        "type": "tool_result",
        "tool_use_id": "call_load_tools",
        "content": expected_content,
    }
