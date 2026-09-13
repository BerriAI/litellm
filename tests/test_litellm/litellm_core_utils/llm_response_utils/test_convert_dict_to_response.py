from typing import Final

import pytest

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


def test_convert_empty_choices_response() -> None:
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_streaming_response,
    )

    resp: Final = {
        "id": "x",
        "created": 1,
        "model": "gemini-3.5-flash",
        "object": "chat.completion",
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        "vertex_ai_safety_results": ["blocked"],
    }
    result: Final = convert_to_model_response_object(
        response_object=resp,
        model_response_object=ModelResponse(),
        response_type="completion",
    )
    assert result.choices == []
    assert getattr(result, "vertex_ai_safety_results") == ["blocked"]

    sync_stream: Final = list(convert_to_streaming_response(response_object=resp))
    assert len(sync_stream) == 1
    assert sync_stream[0].choices == []


@pytest.mark.asyncio
async def test_convert_empty_choices_response_async() -> None:
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_streaming_response_async,
    )

    resp: Final = {
        "id": "x",
        "created": 1,
        "model": "gemini-3.5-flash",
        "object": "chat.completion",
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }
    async_chunks: Final = [chunk async for chunk in convert_to_streaming_response_async(response_object=resp)]
    assert len(async_chunks) == 1
    assert async_chunks[0].choices == []


def test_convert_missing_choices_raises_api_error() -> None:
    from litellm.exceptions import APIError

    resp: Final = {
        "id": "x",
        "created": 1,
        "model": "gemini-3.5-flash",
        "object": "chat.completion",
    }
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object=resp,
            model_response_object=ModelResponse(),
            response_type="completion",
        )
    assert "no 'choices'" in str(exc_info.value)


@pytest.mark.parametrize(("choices", "type_name"), [({}, "dict"), ("", "str"), (None, "NoneType"), (0, "int")])
@pytest.mark.asyncio
async def test_convert_non_list_choices_raises_api_error(choices: object, type_name: str) -> None:
    from litellm.exceptions import APIError
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_streaming_response,
        convert_to_streaming_response_async,
    )

    resp: Final = {
        "id": "x",
        "created": 1,
        "model": "gemini-3.5-flash",
        "object": "chat.completion",
        "choices": choices,
    }
    expected: Final = f"'choices' that is not a list \\({type_name}\\)"
    with pytest.raises(APIError, match=expected):
        convert_to_model_response_object(
            response_object=resp,
            model_response_object=ModelResponse(),
            response_type="completion",
        )
    with pytest.raises(APIError, match=expected):
        list(convert_to_streaming_response(response_object=resp))
    with pytest.raises(APIError, match=expected):
        async for _ in convert_to_streaming_response_async(response_object=resp):
            pass
