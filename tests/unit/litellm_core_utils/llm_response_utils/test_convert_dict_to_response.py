from typing import Final

import pytest

from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.exceptions import APIError
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _coerce_missing_choices_status,
    _get_missing_choices_error_args,
    _handle_invalid_parallel_tool_calls,
    _should_convert_tool_call_to_json_mode,
    convert_to_model_response_object,
    convert_to_streaming_response,
    convert_to_streaming_response_async,
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
                            "input": '*** Begin Patch\n*** Update File: main.py\n@@\n+def hello():\n+    print("Hello")\n*** End Patch\n',
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


def test_non_openai_error_coerces_numeric_string_status():
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={
                "status": "400",
                "response": "Token is invalid [2]",
                "choices": None,
            },
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 400
    assert "Token is invalid [2]" in exc_info.value.message


def test_non_openai_error_ignores_non_decimal_unicode_status():
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={
                "status": "²00",
                "response": "Provider returned an invalid status",
                "choices": None,
            },
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 500
    assert "Provider returned an invalid status" in exc_info.value.message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, None),
        (200, 200),
        (99, None),
        (" 201 ", 201),
        ("²00", None),
        ("١٠٠", None),
        ("99", None),
        (None, None),
        (3.14, None),
    ],
)
def test_coerce_missing_choices_status(value: object, expected: int | None) -> None:
    assert _coerce_missing_choices_status(value) == expected


@pytest.mark.parametrize(
    ("response_object", "expected"),
    [
        ({"status": 429, "response": "Too many requests"}, (429, "Too many requests")),
        ({"status": None, "status_code": " 430 ", "message": "Service unavailable"}, (430, "Service unavailable")),
        (
            {"error": {"status_code": 431, "message": "Nested failure"}, "response": ""},
            (431, "Nested failure"),
        ),
        ({"error": {"status": 432, "response": "Nested response"}}, (432, "Nested response")),
        ({"error": "Provider request failed"}, (500, "Provider request failed")),
        ({"error": object()}, (500, "LiteLLM: provider returned a response with no 'choices'. Raw keys: ['error']")),
        ({"message": "  "}, (500, "LiteLLM: provider returned a response with no 'choices'. Raw keys: ['message']")),
    ],
)
def test_get_missing_choices_error_args(response_object: dict, expected: tuple[int, str]) -> None:
    assert _get_missing_choices_error_args(response_object) == expected


@pytest.mark.parametrize("code", [401, "401"])
def test_non_openai_error_uses_nested_error_object(code):
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={
                "error": {"message": "Invalid credentials", "code": code},
                "choices": None,
            },
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.message


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_message"),
    [
        ({"response": "Provider unavailable", "status": "503"}, 503, "Provider unavailable"),
        ("Provider request failed", 500, "Provider request failed"),
    ],
)
def test_non_openai_error_uses_nested_error_message_variants(error, expected_status, expected_message):
    with pytest.raises(APIError) as exc_info:
        convert_to_model_response_object(
            response_object={"error": error, "choices": None},
            model_response_object=ModelResponse(),
        )

    assert exc_info.value.status_code == expected_status
    assert expected_message in exc_info.value.message


def test_streaming_non_openai_error_uses_provider_status_and_message():
    with pytest.raises(APIError) as exc_info:
        next(convert_to_streaming_response({"error": {"message": "Invalid credentials", "code": 401}, "choices": None}))

    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.message


@pytest.mark.asyncio
async def test_async_streaming_non_openai_error_uses_provider_status_and_message():
    with pytest.raises(APIError) as exc_info:
        await convert_to_streaming_response_async(
            {"error": {"message": "Invalid credentials", "code": 401}, "choices": None}
        ).__anext__()

    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.message


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
