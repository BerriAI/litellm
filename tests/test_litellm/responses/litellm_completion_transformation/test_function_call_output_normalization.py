"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    TOOL_CALLS_CACHE,
    LiteLLMCompletionResponsesConfig,
)


@pytest.fixture(autouse=True)
def clear_cached_tool_use_definition():
    """
    TOOL_CALLS_CACHE is module-level state shared by every test in a worker:
    transforming a completion that carries tool_calls stores the definition
    under the call id, and this transformation prepends that assistant turn
    when the ids match. Under xdist there is no per-module litellm reload to
    clear it, so drop the id around each test -- otherwise the message count
    below depends on whichever tests happened to run earlier in the worker.
    """
    TOOL_CALLS_CACHE.delete_cache(key="call_1")
    yield
    TOOL_CALLS_CACHE.delete_cache(key="call_1")


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
