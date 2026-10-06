from typing import Final

import asyncio, importlib, litellm, pytest

from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    LiteLLMResponseObjectHandler,
    safe_convert_created_field,
    handle_invalid_parallel_tool_calls,
    should_convert_tool_call_to_json_mode,
    convert_to_model_response_object,
)
from litellm.types.utils import(
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Function,
    ImageResponse,
    ModelResponse,
)
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome

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


def test_safe_convert_created_field_preserves_large_integer_precision():
    created_value = 2**53 + 1

    assert safe_convert_created_field(created_value) == created_value


def test_safe_convert_created_field_accepts_float_convertible_non_strings():
    class FloatConvertible:
        def __float__(self) -> float:
            return 1.5

    assert safe_convert_created_field(FloatConvertible()) == 1


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
        should_convert_tool_call_to_json_mode(
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
        should_convert_tool_call_to_json_mode(
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
    result = handle_invalid_parallel_tool_calls([custom_tool_call, function_tool_call])
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


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_basic():
    # Test basic conversion with minimal input
    response_dict = {
        "created": 1234567890,
        "data": [{"url": "http://example.com/image.jpg"}],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].url == "http://example.com/image.jpg"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_with_hidden_params():
    # Test with hidden params
    response_dict = {
        "created": 1234567890,
        "data": [{"url": "http://example.com/image.jpg"}],
    }
    hidden_params = {"api_key": "test_key"}

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict, hidden_params=hidden_params)

    assert result._hidden_params == {"api_key": "test_key"}

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_multiple_images():
    # Test handling multiple images in response
    response_dict = {
        "created": 1234567890,
        "data": [
            {"url": "http://example.com/image1.jpg"},
            {"url": "http://example.com/image2.jpg"},
        ],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert len(result.data) == 2
    assert result.data[0].url == "http://example.com/image1.jpg"
    assert result.data[1].url == "http://example.com/image2.jpg"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_with_b64_json():
    # Test handling b64_json in response
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert result.data[0].b64_json == "base64encodedstring"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_with_extra_fields():
    response_dict = {
        "created": 1234567890,
        "data": [
            {
                "url": "http://example.com/image1.jpg",
                "content_filter_results": {"category": "violence", "flagged": True},
            },
            {
                "url": "http://example.com/image2.jpg",
                "content_filter_results": {"category": "violence", "flagged": True},
            },
        ],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert result.data[0].url == "http://example.com/image1.jpg"
    assert result.data[1].url == "http://example.com/image2.jpg"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_with_extra_fields_2():
    """
    Date from a non-OpenAI API could have some obscure field in addition to the expected ones. This should not break the conversion.
    """
    response_dict = {
        "created": 1234567890,
        "data": [
            {
                "url": "http://example.com/image1.jpg",
                "very_obscure_field": "some_value",
            },
            {
                "url": "http://example.com/image2.jpg",
                "very_obscure_field2": "some_other_value",
            },
        ],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert result.data[0].url == "http://example.com/image1.jpg"
    assert result.data[1].url == "http://example.com/image2.jpg"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_with_none_usage_fields():
    """
    Test handling of None values in usage fields, specifically for gpt-image-1 responses.

    This test verifies the fix for the bug where gpt-image-1 returns None values
    for usage statistics fields, which caused Pydantic validation errors.
    The fix should clean these None values and let ImageResponse constructor
    handle the default values.
    """
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
        "usage": {
            "input_tokens": None,  # gpt-image-1 returns None instead of integer
            "input_tokens_details": None,  # gpt-image-1 returns None instead of object
            "output_tokens": None,  # gpt-image-1 returns None instead of integer
            "total_tokens": None,  # gpt-image-1 returns None instead of integer
        },
    }

    # This should not raise a ValidationError
    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].b64_json == "base64encodedstring"

    # Usage should be properly initialized with default values
    assert result.usage is not None
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.total_tokens == 0
    assert result.usage.input_tokens_details is not None
    assert result.usage.input_tokens_details.image_tokens == 0
    assert result.usage.input_tokens_details.text_tokens == 0

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_with_partial_none_usage_fields():
    """
    Test handling of mixed None and valid values in usage fields.
    """
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
        "usage": {
            "input_tokens": 10,  # Valid value
            "input_tokens_details": None,  # None value (should be cleaned)
            "output_tokens": None,  # None value (should be cleaned)
            "total_tokens": 10,  # Valid value
        },
    }

    # This should not raise a ValidationError
    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].b64_json == "base64encodedstring"

    # Usage should be properly initialized with defaults where needed
    # Valid values should be preserved, None values should be cleaned and use defaults
    assert result.usage is not None
    assert result.usage.input_tokens == 10  # Valid value should be preserved
    assert result.usage.output_tokens == 0  # None value should become 0
    assert result.usage.total_tokens == 10  # Calculated as input_tokens + output_tokens (10 + 0)
    assert result.usage.input_tokens_details is not None
    assert result.usage.input_tokens_details.image_tokens == 0
    assert result.usage.input_tokens_details.text_tokens == 0

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_convert_to_image_response_with_valid_usage_fields():
    """
    Test that valid usage fields are preserved correctly.
    """
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
        "usage": {
            "input_tokens": 50,
            "input_tokens_details": {
                "image_tokens": 30,
                "text_tokens": 20,
            },
            "output_tokens": 10,
            "total_tokens": 60,
        },
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].b64_json == "base64encodedstring"

    # Valid usage fields should be preserved
    assert result.usage is not None
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 10
    assert result.usage.total_tokens == 60
    assert result.usage.input_tokens_details is not None
    assert result.usage.input_tokens_details.image_tokens == 30
    assert result.usage.input_tokens_details.text_tokens == 20
