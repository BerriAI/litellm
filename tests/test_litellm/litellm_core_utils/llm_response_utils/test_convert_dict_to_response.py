import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _handle_invalid_parallel_tool_calls,
    _should_convert_tool_call_to_json_mode,
    convert_to_model_response_object,
)
from litellm.types.utils import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Function,
    ModelResponse,
)

OPENAI_CUSTOM_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-abc",
    "created": 1784657740,
    "model": "gpt-5.6",
    "object": "chat.completion",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_njxQ",
                        "type": "custom",
                        "custom": {
                            "name": "ApplyPatch",
                            "input": "*** Begin Patch\n*** Update File: main.py\n@@\n+def hello():\n+    print(\"Hello\")\n*** End Patch\n",
                        },
                    }
                ],
                "refusal": None,
                "annotations": [],
            },
        }
    ],
    "usage": {"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15},
}


def test_convert_openai_custom_tool_call_response():
    result = convert_to_model_response_object(
        response_object=OPENAI_CUSTOM_TOOL_CALL_RESPONSE,
        model_response_object=ModelResponse(),
        response_type="completion",
    )
    tool_calls = result.choices[0].message.tool_calls
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ChatCompletionMessageCustomToolCall)
    dumped = tool_calls[0].model_dump()
    assert dumped == OPENAI_CUSTOM_TOOL_CALL_RESPONSE["choices"][0]["message"]["tool_calls"][0]
    assert result.choices[0].finish_reason == "tool_calls"


def test_should_convert_tool_call_to_json_mode_ignores_custom_tool_call():
    custom_tool_call = ChatCompletionMessageCustomToolCall(
        id="call_c",
        custom={"name": "ApplyPatch", "input": "patch"},
    )
    assert (
        _should_convert_tool_call_to_json_mode(
            tool_calls=[custom_tool_call],
            convert_tool_call_to_json_mode=True,
        )
        is False
    )


def test_should_convert_tool_call_to_json_mode_still_matches_response_format_tool():
    response_format_call = ChatCompletionMessageToolCall(
        id="call_f",
        type="function",
        function=Function(name=RESPONSE_FORMAT_TOOL_NAME, arguments='{"answer": 4}'),
    )
    assert (
        _should_convert_tool_call_to_json_mode(
            tool_calls=[response_format_call],
            convert_tool_call_to_json_mode=True,
        )
        is True
    )


def test_handle_invalid_parallel_tool_calls_skips_custom_tool_calls():
    custom_tool_call = ChatCompletionMessageCustomToolCall(
        id="call_c",
        custom={"name": "ApplyPatch", "input": "patch"},
    )
    function_tool_call = ChatCompletionMessageToolCall(
        id="call_f",
        type="function",
        function=Function(name="get_weather", arguments='{"city": "SF"}'),
    )
    result = _handle_invalid_parallel_tool_calls([custom_tool_call, function_tool_call])
    assert result == [custom_tool_call, function_tool_call]
