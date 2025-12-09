import copy
import logging
import sys
import time
from datetime import datetime
from unittest import mock

from dotenv import load_dotenv

from litellm.types.utils import StandardCallbackDynamicParams

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, headers
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.litellm_core_utils.duration_parser import (
    get_last_day_of_month,
    _extract_from_regex,
)
from litellm.utils import (
    check_valid_key,
    create_pretrained_tokenizer,
    create_tokenizer,
    function_to_dict,
    get_llm_provider,
    get_max_tokens,
    get_supported_openai_params,
    get_token_count,
    get_valid_models,
    trim_messages,
    validate_environment,
)
from unittest.mock import AsyncMock, MagicMock, patch


# Assuming your trim_messages, shorten_message_to_fit_limit, and get_token_count functions are all in a module named 'message_utils'
@pytest.fixture(autouse=True)
def reset_mock_cache():
    from litellm.utils import _model_cache

    _model_cache.flush_cache()


# Test 1: Check trimming of normal message
def test_basic_trimming():
    litellm._turn_on_debug()
    messages = [
        {
            "role": "user",
            "content": "This is a long message that definitely exceeds the token limit.",
        }
    ]
    trimmed_messages = trim_messages(messages, model="claude-2", max_tokens=8)
    print("trimmed messages")
    print(trimmed_messages)
    # print(get_token_count(messages=trimmed_messages, model="claude-2"))
    assert (get_token_count(messages=trimmed_messages, model="claude-2")) <= 8


# test_basic_trimming()


def test_basic_trimming_no_max_tokens_specified():
    messages = [
        {
            "role": "user",
            "content": "This is a long message that is definitely under the token limit.",
        }
    ]
    trimmed_messages = trim_messages(messages, model="gpt-4")
    print("trimmed messages for gpt-4")
    print(trimmed_messages)
    # print(get_token_count(messages=trimmed_messages, model="claude-2"))
    assert (
        get_token_count(messages=trimmed_messages, model="gpt-4")
    ) <= litellm.model_cost["gpt-4"]["max_tokens"]


# test_basic_trimming_no_max_tokens_specified()


def test_multiple_messages_trimming():
    messages = [
        {
            "role": "user",
            "content": "This is a long message that will exceed the token limit.",
        },
        {
            "role": "user",
            "content": "This is another long message that will also exceed the limit.",
        },
    ]
    trimmed_messages = trim_messages(
        messages=messages, model="gpt-3.5-turbo", max_tokens=20
    )
    # print(get_token_count(messages=trimmed_messages, model="gpt-3.5-turbo"))
    assert (get_token_count(messages=trimmed_messages, model="gpt-3.5-turbo")) <= 20


# test_multiple_messages_trimming()


def test_multiple_messages_no_trimming():
    messages = [
        {
            "role": "user",
            "content": "This is a long message that will exceed the token limit.",
        },
        {
            "role": "user",
            "content": "This is another long message that will also exceed the limit.",
        },
    ]
    trimmed_messages = trim_messages(
        messages=messages, model="gpt-3.5-turbo", max_tokens=100
    )
    print("Trimmed messages")
    print(trimmed_messages)
    assert messages == trimmed_messages


# test_multiple_messages_no_trimming()


def test_large_trimming_multiple_messages():
    messages = [
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
    ]
    trimmed_messages = trim_messages(messages, max_tokens=20, model="gpt-4-0613")
    print("trimmed messages")
    print(trimmed_messages)
    assert (get_token_count(messages=trimmed_messages, model="gpt-4-0613")) <= 20


# test_large_trimming()


def test_large_trimming_single_message():
    messages = [
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."}
    ]
    trimmed_messages = trim_messages(messages, max_tokens=5, model="gpt-4-0613")
    assert (get_token_count(messages=trimmed_messages, model="gpt-4-0613")) <= 5
    assert (get_token_count(messages=trimmed_messages, model="gpt-4-0613")) > 0


def test_trimming_with_system_message_within_max_tokens():
    # This message is 33 tokens long
    messages = [
        {"role": "system", "content": "This is a short system message"},
        {
            "role": "user",
            "content": "This is a medium normal message, let's say litellm is awesome.",
        },
    ]
    trimmed_messages = trim_messages(
        messages, max_tokens=30, model="gpt-4-0613"
    )  # The system message should fit within the token limit
    assert len(trimmed_messages) == 2
    assert trimmed_messages[0]["content"] == "This is a short system message"


def test_trimming_with_system_message_exceeding_max_tokens():
    # This message is 33 tokens long. The system message is 13 tokens long.
    messages = [
        {"role": "system", "content": "This is a short system message"},
        {
            "role": "user",
            "content": "This is a medium normal message, let's say litellm is awesome.",
        },
    ]
    trimmed_messages = trim_messages(messages, max_tokens=12, model="gpt-4-0613")
    assert len(trimmed_messages) == 1


def test_trimming_with_tool_calls():
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        },
        Message(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    function=Function(
                        arguments='{"location": "San Francisco, CA", "unit": "celsius"}',
                        name="get_current_weather",
                    ),
                    id="call_G11shFcS024xEKjiAOSt6Tc9",
                    type="function",
                ),
                ChatCompletionMessageToolCall(
                    function=Function(
                        arguments='{"location": "Tokyo, Japan", "unit": "celsius"}',
                        name="get_current_weather",
                    ),
                    id="call_e0ss43Bg7H8Z9KGdMGWyZ9Mj",
                    type="function",
                ),
                ChatCompletionMessageToolCall(
                    function=Function(
                        arguments='{"location": "Paris, France", "unit": "celsius"}',
                        name="get_current_weather",
                    ),
                    id="call_nRjLXkWTJU2a4l9PZAf5as6g",
                    type="function",
                ),
            ],
            function_call=None,
        ),
        {
            "tool_call_id": "call_G11shFcS024xEKjiAOSt6Tc9",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
        },
        {
            "tool_call_id": "call_e0ss43Bg7H8Z9KGdMGWyZ9Mj",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}',
        },
        {
            "tool_call_id": "call_nRjLXkWTJU2a4l9PZAf5as6g",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "Paris", "temperature": "22", "unit": "celsius"}',
        },
    ]
    num_tool_calls = 3

    result = trim_messages(messages=messages, max_tokens=1)

    print(result)

    # only trailing tool calls are returned
    assert len(result) == num_tool_calls
    assert result == messages[-num_tool_calls:]

    result = trim_messages(messages=messages, max_tokens=999)
    # message length is below max_tokens, so output should match input
    assert messages == result


def test_trimming_should_not_change_original_messages():
    messages = [
        {"role": "system", "content": "This is a short system message"},
        {
            "role": "user",
            "content": "This is a medium normal message, let's say litellm is awesome.",
        },
    ]
    messages_copy = copy.deepcopy(messages)
    trimmed_messages = trim_messages(messages, max_tokens=12, model="gpt-4-0613")
    assert messages == messages_copy


@pytest.mark.parametrize("model", ["gpt-4-0125-preview", "claude-3-opus-20240229"])
def test_trimming_with_model_cost_max_input_tokens(model):
    messages = [
        {"role": "system", "content": "This is a normal system message"},
        {
            "role": "user",
            "content": "This is a sentence" * 100000,
        },
    ]
    trimmed_messages = trim_messages(messages, model=model)
    assert (
        get_token_count(trimmed_messages, model=model)
        < litellm.model_cost[model]["max_input_tokens"]
    )


def test_trimming_with_untokenizable_field(caplog: pytest.LogCaptureFixture) -> None:
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?",
            # non-string values will cause the tokenizer to raise an exception
            "user_id": 123,
        },
        Message(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    function=Function(
                        arguments='{"location": "San Francisco, CA", "unit": "celsius"}',
                        name="get_current_weather",
                    ),
                    id="call_G11shFcS024xEKjiAOSt6Tc9",
                    type="function",
                ),
            ],
            function_call=None,
        ),
        {
            "tool_call_id": "call_G11shFcS024xEKjiAOSt6Tc9",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
        },
    ]

    # trim_messages() catches the exception raised by the tokenizer and logs an error
    with caplog.at_level(level=logging.ERROR, logger="LiteLLM"):
        trimmed_messages = trim_messages(messages, max_tokens=999)

    assert trimmed_messages == messages


def test_aget_valid_models():
    old_environ = os.environ
    os.environ = {"OPENAI_API_KEY": "temp"}  # mock set only openai key in environ

    valid_models = get_valid_models()
    print(valid_models)

    # list of openai supported llms on litellm
    expected_models = (
        litellm.open_ai_chat_completion_models | litellm.open_ai_text_completion_models
    )

    assert set(valid_models) == set(expected_models)

    # reset replicate env key
    os.environ = old_environ

    # GEMINI
    expected_models = litellm.gemini_models
    old_environ = os.environ
    os.environ = {"GEMINI_API_KEY": "temp"}  # mock set only openai key in environ

    valid_models = get_valid_models()

    print(valid_models)
    assert set(valid_models) == set(expected_models)

    # reset replicate env key
    os.environ = old_environ


@pytest.mark.parametrize("custom_llm_provider", ["anthropic", "xai"])
def test_get_valid_models_with_custom_llm_provider(custom_llm_provider):
    from litellm.utils import ProviderConfigManager
    from litellm.types.utils import LlmProviders

    provider_config = ProviderConfigManager.get_provider_model_info(
        model=None,
        provider=LlmProviders(custom_llm_provider),
    )
    assert provider_config is not None
    valid_models = get_valid_models(
        check_provider_endpoint=True, custom_llm_provider=custom_llm_provider
    )
    print(valid_models)
    assert len(valid_models) > 0
    assert set(provider_config.get_models()) == set(valid_models)


# test_get_valid_models()


def test_bad_key():
    key = "bad-key"
    response = check_valid_key(model="gpt-3.5-turbo", api_key=key)
    print(response, key)
    assert response == False


def test_good_key():
    key = os.environ["OPENAI_API_KEY"]
    response = check_valid_key(model="gpt-3.5-turbo", api_key=key)
    assert response == True


# test validate environment


def test_validate_environment_empty_model():
    api_key = validate_environment()
    if api_key is None:
        raise Exception()


def test_validate_environment_api_key():
    response_obj = validate_environment(model="gpt-3.5-turbo", api_key="sk-my-test-key")
    assert (
        response_obj["keys_in_environment"] is True
    ), f"Missing keys={response_obj['missing_keys']}"


def test_validate_environment_api_version():
    response_obj = validate_environment(
        model="azure/openai-deployment",
        api_key="sk-my-test-key",
        api_base="https://fake.openai.azure.com/",
        api_version="2024-02-15",
    )
    assert (
        response_obj["keys_in_environment"] is True
    ), f"Missing keys={response_obj['missing_keys']}"


def test_validate_environment_api_base_dynamic():
    for provider in ["ollama", "ollama_chat"]:
        kv = validate_environment(provider + "/mistral", api_base="https://example.com")
        assert kv["keys_in_environment"]
        assert kv["missing_keys"] == []


@mock.patch.dict(os.environ, {"OLLAMA_API_BASE": "foo"}, clear=True)
def test_validate_environment_ollama():
    for provider in ["ollama", "ollama_chat"]:
        kv = validate_environment(provider + "/mistral")
        assert kv["keys_in_environment"]
        assert kv["missing_keys"] == []


@mock.patch.dict(os.environ, {}, clear=True)
def test_validate_environment_ollama_failed():
    for provider in ["ollama", "ollama_chat"]:
        kv = validate_environment(provider + "/mistral")
        assert not kv["keys_in_environment"]
        assert kv["missing_keys"] == ["OLLAMA_API_BASE"]


def test_function_to_dict():
    print("testing function to dict for get current weather")

    def get_current_weather(location: str, unit: str):
        """Get the current weather in a given location

        Parameters
        ----------
        location : str
            The city and state, e.g. San Francisco, CA
        unit : {'celsius', 'fahrenheit'}
            Temperature unit

        Returns
        -------
        str
            a sentence indicating the weather
        """
        if location == "Boston, MA":
            return "The weather is 12F"

    function_json = litellm.utils.function_to_dict(get_current_weather)
    print(function_json)

    expected_output = {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit",
                    "enum": "['fahrenheit', 'celsius']",
                },
            },
            "required": ["location", "unit"],
        },
    }
    print(expected_output)

    assert function_json["name"] == expected_output["name"]
    assert function_json["description"] == expected_output["description"]
    assert function_json["parameters"]["type"] == expected_output["parameters"]["type"]
    assert (
        function_json["parameters"]["properties"]["location"]
        == expected_output["parameters"]["properties"]["location"]
    )

    # the enum can change it can be - which is why we don't assert on unit
    # {'type': 'string', 'description': 'Temperature unit', 'enum': "['fahrenheit', 'celsius']"}
    # {'type': 'string', 'description': 'Temperature unit', 'enum': "['celsius', 'fahrenheit']"}

    assert (
        function_json["parameters"]["required"]
        == expected_output["parameters"]["required"]
    )

    print("passed")


# test_function_to_dict()


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("gpt-3.5-turbo", True),
        ("azure/gpt-4-1106-preview", True),
        ("groq/gemma-7b-it", True),
        ("gemini/gemini-1.5-flash", True),
    ],
)
def test_supports_function_calling(model, expected_bool):
    try:
        assert litellm.supports_function_calling(model=model) == expected_bool
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("gpt-4o-mini-search-preview", True),
        ("openai/gpt-4o-mini-search-preview", True),
        ("gpt-4o-search-preview", True),
        ("openai/gpt-4o-search-preview", True),
        ("groq/deepseek-r1-distill-llama-70b", False),
        ("groq/llama-3.3-70b-versatile", False),
        ("codestral/codestral-latest", False),
    ],
)
def test_supports_web_search(model, expected_bool):
    try:
        assert litellm.supports_web_search(model=model) == expected_bool
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("openai/o3-mini", True),
        ("o3-mini", True),
        ("xai/grok-3-mini-beta", True),
        ("xai/grok-3-mini-fast-beta", True),
        ("xai/grok-2", False),
        ("gpt-3.5-turbo", False),
    ],
)
def test_supports_reasoning(model, expected_bool):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        assert litellm.supports_reasoning(model=model) == expected_bool
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_get_max_token_unit_test():
    """
    More complete testing in `test_completion_cost.py`
    """
    model = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"

    max_tokens = get_max_tokens(
        model
    )  # Returns a number instead of throwing an Exception

    assert isinstance(max_tokens, int)


def test_get_supported_openai_params() -> None:
    # Mapped provider
    assert isinstance(get_supported_openai_params("gpt-4"), list)

    # Unmapped provider
    assert get_supported_openai_params("nonexistent") is None


def test_get_chat_completion_prompt():
    """
    Unit test to ensure get_chat_completion_prompt updates messages in logging object.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    updated_message = "hello world"

    litellm_logging_obj.get_chat_completion_prompt(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": updated_message}],
        non_default_params={},
        prompt_id="1234",
        prompt_variables=None,
    )

    assert litellm_logging_obj.messages == [
        {"role": "user", "content": updated_message}
    ]


def test_redact_msgs_from_logs():
    """
    Tests that turn_off_message_logging does not modify the response_obj

    On the proxy some users were seeing the redaction impact client side responses
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.redact_messages import (
        redact_message_input_output_from_logging,
    )

    litellm.turn_off_message_logging = True

    response_obj = litellm.ModelResponse(
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner.",
                    "role": "assistant",
                },
            }
        ]
    )

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )

    # Assert the response_obj content is NOT modified
    assert (
        response_obj.choices[0].message.content
        == "I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner."
    )

    litellm.turn_off_message_logging = False
    print("Test passed")


def test_redact_embedding_response():
    """
    Tests that EmbeddingResponse redaction preserves critical metadata while clearing sensitive data

    This test ensures that:
    1. usage field is preserved for token/cost tracking
    2. model field is preserved for response structure integrity
    3. data field (containing embeddings) is cleared for privacy
    4. original response object is not modified
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.redact_messages import (
        redact_message_input_output_from_logging,
    )

    litellm.turn_off_message_logging = True

    # Create a test EmbeddingResponse with usage data
    original_usage = litellm.Usage(
        prompt_tokens=10, completion_tokens=0, total_tokens=10
    )
    original_data = [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
        {"object": "embedding", "index": 1, "embedding": [0.6, 0.7, 0.8, 0.9, 1.0]},
    ]

    response_obj = litellm.EmbeddingResponse(
        model="text-embedding-ada-002",
        data=original_data,
        usage=original_usage,
        object="list",
    )

    litellm_logging_obj = Logging(
        model="text-embedding-ada-002",
        messages=[{"role": "user", "content": "test input"}],
        stream=False,
        call_type="embedding",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )

    # Assert the original response_obj is NOT modified
    assert response_obj.data == original_data
    assert response_obj.usage == original_usage
    assert response_obj.model == "text-embedding-ada-002"
    assert response_obj.object == "list"

    # Assert the redacted response preserves critical metadata
    assert _redacted_response_obj.usage == original_usage  # usage should be preserved
    assert (
        _redacted_response_obj.model == "text-embedding-ada-002"
    )  # model should be preserved
    assert _redacted_response_obj.object == "list"  # object should be preserved

    # Assert sensitive data is cleared
    assert _redacted_response_obj.data == []  # data should be cleared

    # Assert it's still an EmbeddingResponse instance
    assert isinstance(_redacted_response_obj, litellm.EmbeddingResponse)

    litellm.turn_off_message_logging = False
    print("Test passed")


def test_redact_msgs_from_logs_with_dynamic_params():
    """
    Tests redaction behavior based on standard_callback_dynamic_params setting:
    In all tests litellm.turn_off_message_logging is True


    1. When standard_callback_dynamic_params.turn_off_message_logging is False (or not set): No redaction should occur. User has opted out of redaction.
    2. When standard_callback_dynamic_params.turn_off_message_logging is True: Redaction should occur. User has opted in to redaction.
    3. standard_callback_dynamic_params.turn_off_message_logging not set, litellm.turn_off_message_logging is True: Redaction should occur.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.redact_messages import (
        redact_message_input_output_from_logging,
    )

    litellm.turn_off_message_logging = True
    test_content = "I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner."
    response_obj = litellm.ModelResponse(
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": test_content,
                    "role": "assistant",
                },
            }
        ]
    )

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    # Test Case 1: standard_callback_dynamic_params = False (or not set)
    standard_callback_dynamic_params = StandardCallbackDynamicParams(
        turn_off_message_logging=False
    )
    litellm_logging_obj.model_call_details["standard_callback_dynamic_params"] = (
        standard_callback_dynamic_params
    )
    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )
    # Assert no redaction occurred
    assert _redacted_response_obj.choices[0].message.content == test_content

    # Test Case 2: standard_callback_dynamic_params = True
    standard_callback_dynamic_params = StandardCallbackDynamicParams(
        turn_off_message_logging=True
    )
    litellm_logging_obj.model_call_details["standard_callback_dynamic_params"] = (
        standard_callback_dynamic_params
    )
    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )
    # Assert redaction occurred
    assert _redacted_response_obj.choices[0].message.content == "redacted-by-litellm"

    # Test Case 3: standard_callback_dynamic_params does not override litellm.turn_off_message_logging
    # since litellm.turn_off_message_logging is True redaction should occur
    standard_callback_dynamic_params = StandardCallbackDynamicParams()
    litellm_logging_obj.model_call_details["standard_callback_dynamic_params"] = (
        standard_callback_dynamic_params
    )
    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )
    # Assert no redaction occurred
    assert _redacted_response_obj.choices[0].message.content == "redacted-by-litellm"

    # Reset settings
    litellm.turn_off_message_logging = False
    print("Test passed")


@pytest.mark.parametrize(
    "duration, unit",
    [("7s", "s"), ("7m", "m"), ("7h", "h"), ("7d", "d"), ("7mo", "mo")],
)
def test_extract_from_regex(duration, unit):
    value, _unit = _extract_from_regex(duration=duration)

    assert value == 7
    assert _unit == unit


def test_duration_in_seconds():
    """
    Test if duration int is correctly calculated for different str
    """
    import time

    now = time.time()
    current_time = datetime.fromtimestamp(now)

    if current_time.month == 12:
        target_year = current_time.year + 1
        target_month = 1
    else:
        target_year = current_time.year
        target_month = current_time.month + 1

    # Determine the day to set for next month
    target_day = current_time.day
    last_day_of_target_month = get_last_day_of_month(target_year, target_month)

    if target_day > last_day_of_target_month:
        target_day = last_day_of_target_month

    next_month = datetime(
        year=target_year,
        month=target_month,
        day=target_day,
        hour=current_time.hour,
        minute=current_time.minute,
        second=current_time.second,
        microsecond=current_time.microsecond,
    )

    # Calculate the duration until the first day of the next month
    duration_until_next_month = next_month - current_time
    expected_duration = int(duration_until_next_month.total_seconds())

    value = duration_in_seconds(duration="1mo")

    assert value - expected_duration < 2


def test_duration_in_seconds_basic():
    assert duration_in_seconds(duration="3s") == 3
    assert duration_in_seconds(duration="3m") == 180
    assert duration_in_seconds(duration="3h") == 10800
    assert duration_in_seconds(duration="3d") == 259200
    assert duration_in_seconds(duration="3w") == 1814400


def test_get_llm_provider_ft_models():
    """
    All ft prefixed models should map to OpenAI
    gpt-3.5-turbo-0125 (recommended),
    gpt-3.5-turbo-1106,
    gpt-3.5-turbo,
    gpt-4-0613 (experimental)
    gpt-4o-2024-05-13.
    babbage-002, davinci-002,

    """
    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-0125")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-1106")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-4-0613")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-4o-2024-05-13")
    assert custom_llm_provider == "openai"


@pytest.mark.parametrize("langfuse_trace_id", [None, "my-unique-trace-id"])
@pytest.mark.parametrize(
    "langfuse_existing_trace_id", [None, "my-unique-existing-trace-id"]
)
def test_logging_trace_id(langfuse_trace_id, langfuse_existing_trace_id):
    """
    - Unit test for `_get_trace_id` function in Logging obj
    """
    from litellm.litellm_core_utils.litellm_logging import Logging

    litellm.success_callback = ["langfuse"]
    litellm_call_id = "my-unique-call-id"
    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id=litellm_call_id,
        start_time=datetime.now(),
        function_id="1234",
    )

    metadata = {}

    if langfuse_trace_id is not None:
        metadata["trace_id"] = langfuse_trace_id
    if langfuse_existing_trace_id is not None:
        metadata["existing_trace_id"] = langfuse_existing_trace_id

    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey how's it going?"}],
        mock_response="Hey!",
        litellm_logging_obj=litellm_logging_obj,
        metadata=metadata,
    )

    time.sleep(3)
    assert litellm_logging_obj._get_trace_id(service_name="langfuse") is not None

    ## if existing_trace_id exists
    if langfuse_existing_trace_id is not None:
        assert (
            litellm_logging_obj._get_trace_id(service_name="langfuse")
            == langfuse_existing_trace_id
        )
    ## if trace_id exists
    elif langfuse_trace_id is not None:
        assert (
            litellm_logging_obj._get_trace_id(service_name="langfuse")
            == langfuse_trace_id
        )
    ## if existing_trace_id exists
    else:
        assert (
            litellm_logging_obj._get_trace_id(service_name="langfuse")
            == litellm_call_id
        )


def test_convert_model_response_object():
    """
    Unit test to ensure model response object correctly handles openrouter errors.
    """
    args = {
        "response_object": {
            "id": None,
            "choices": None,
            "created": None,
            "model": None,
            "object": None,
            "service_tier": None,
            "system_fingerprint": None,
            "usage": None,
            "error": {
                "message": '{"type":"error","error":{"type":"invalid_request_error","message":"Output blocked by content filtering policy"}}',
                "code": 400,
            },
        },
        "model_response_object": litellm.ModelResponse(
            id="chatcmpl-b88ce43a-7bfc-437c-b8cc-e90d59372cfb",
            choices=[
                litellm.Choices(
                    finish_reason="stop",
                    index=0,
                    message=litellm.Message(content="default", role="assistant"),
                )
            ],
            created=1719376241,
            model="openrouter/anthropic/claude-3.5-sonnet",
            object="chat.completion",
            system_fingerprint=None,
            usage=litellm.Usage(),
        ),
        "response_type": "completion",
        "stream": False,
        "start_time": None,
        "end_time": None,
        "hidden_params": None,
    }

    try:
        litellm.convert_to_model_response_object(**args)
        pytest.fail("Expected this to fail")
    except Exception as e:
        assert hasattr(e, "status_code")
        assert e.status_code == 400
        assert hasattr(e, "message")
        assert (
            e.message
            == '{"type":"error","error":{"type":"invalid_request_error","message":"Output blocked by content filtering policy"}}'
        )


@pytest.mark.parametrize(
    "content, expected_reasoning, expected_content",
    [
        (None, None, None),
        (
            "<think>I am thinking here</think>The sky is a canvas of blue",
            "I am thinking here",
            "The sky is a canvas of blue",
        ),
        ("I am a regular response", None, "I am a regular response"),
    ],
)
def test_parse_content_for_reasoning(content, expected_reasoning, expected_content):
    assert litellm.utils._parse_content_for_reasoning(content) == (
        expected_reasoning,
        expected_content,
    )


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("vertex_ai/gemini-1.5-pro", True),
        ("gemini/gemini-1.5-pro", True),
        ("predibase/llama3-8b-instruct", True),
        ("gpt-3.5-turbo", False),
        ("groq/llama-3.3-70b-versatile", True),
    ],
)
def test_supports_response_schema(model, expected_bool):
    """
    Unit tests for 'supports_response_schema' helper function.

    Should be true for gemini-1.5-pro on google ai studio / vertex ai AND predibase models
    Should be false otherwise
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm.utils import supports_response_schema

    response = supports_response_schema(model=model, custom_llm_provider=None)

    assert expected_bool == response


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("gpt-3.5-turbo", True),
        ("gpt-4", True),
        ("command-nightly", False),
        ("gemini-pro", True),
    ],
)
def test_supports_function_calling_v2(model, expected_bool):
    """
    Unit test for 'supports_function_calling' helper function.
    """
    from litellm.utils import supports_function_calling

    response = supports_function_calling(model=model, custom_llm_provider=None)
    assert expected_bool == response


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("gpt-4-vision-preview", True),
        ("gpt-3.5-turbo", False),
        ("claude-3-opus-20240229", True),
        ("gemini-pro-vision", True),
        ("command-nightly", False),
    ],
)
def test_supports_vision(model, expected_bool):
    """
    Unit test for 'supports_vision' helper function.
    """
    from litellm.utils import supports_vision

    response = supports_vision(model=model, custom_llm_provider=None)
    assert expected_bool == response


def test_usage_object_null_tokens():
    """
    Unit test.

    Asserts Usage obj always returns int.

    Fixes https://github.com/BerriAI/litellm/issues/5096
    """
    usage_obj = litellm.Usage(prompt_tokens=2, completion_tokens=None, total_tokens=2)

    assert usage_obj.completion_tokens == 0


def test_is_base64_encoded():
    import base64

    import requests

    litellm.set_verbose = True
    url = "https://dummyimage.com/100/100/fff&text=Test+image"
    response = requests.get(url)
    file_data = response.content

    encoded_file = base64.b64encode(file_data).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_file}"

    from litellm.utils import is_base64_encoded

    assert is_base64_encoded(s=base64_image) is True


@mock.patch("httpx.AsyncClient")
@mock.patch.dict(
    os.environ,
    {"SSL_VERIFY": "/certificate.pem", "SSL_CERTIFICATE": "/client.pem"},
    clear=True,
)
def test_async_http_handler(mock_async_client):
    import httpx
    import ssl

    timeout = 120
    event_hooks = {"request": [lambda r: r]}
    concurrent_limit = 2

    # Mock the transport creation to return a specific transport
    with mock.patch.object(
        AsyncHTTPHandler, "_create_async_transport"
    ) as mock_create_transport:
        mock_transport = mock.MagicMock()
        mock_create_transport.return_value = mock_transport

        AsyncHTTPHandler(timeout, event_hooks, concurrent_limit)

        # Get the call arguments
        call_args = mock_async_client.call_args[1]

        # Assert SSL context is being used instead of direct cert/verify params
        assert call_args["cert"] == "/client.pem"
        assert isinstance(call_args["verify"], ssl.SSLContext)
        assert call_args["transport"] == mock_transport
        assert call_args["event_hooks"] == event_hooks
        assert call_args["headers"] == headers
        assert call_args["timeout"] == timeout
        assert call_args["follow_redirects"] is True


@mock.patch("httpx.AsyncClient")
@mock.patch.dict(os.environ, {}, clear=True)
def test_async_http_handler_force_ipv4(mock_async_client):
    """
    Test AsyncHTTPHandler when litellm.force_ipv4 is True

    This is prod test - we need to ensure that httpx always uses ipv4 when litellm.force_ipv4 is True
    """
    import httpx
    import ssl
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    # Set force_ipv4 to True
    litellm.force_ipv4 = True
    litellm.disable_aiohttp_transport = True

    try:
        timeout = 120
        event_hooks = {"request": [lambda r: r]}
        concurrent_limit = 2

        AsyncHTTPHandler(timeout, event_hooks, concurrent_limit)

        # Get the call arguments
        call_args = mock_async_client.call_args[1]

        ############# IMPORTANT ASSERTION #################
        # Assert transport exists and is configured correctly for using ipv4
        assert isinstance(call_args["transport"], httpx.AsyncHTTPTransport)
        print(call_args["transport"])
        assert call_args["transport"]._pool._local_address == "0.0.0.0"
        ####################################

        # Assert other parameters match
        assert call_args["event_hooks"] == event_hooks
        assert call_args["headers"] == headers
        assert call_args["timeout"] == timeout
        assert isinstance(call_args["verify"], ssl.SSLContext)
        assert call_args["cert"] is None
        assert call_args["follow_redirects"] is True

    finally:
        # Reset force_ipv4 to default
        litellm.force_ipv4 = False


@pytest.mark.parametrize(
    "model, expected_bool", [("gpt-3.5-turbo", False), ("gpt-4o-audio-preview", True)]
)
def test_supports_audio_input(model, expected_bool):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm.utils import supports_audio_input, supports_audio_output

    supports_pc = supports_audio_input(model=model)

    assert supports_pc == expected_bool


def test_is_base64_encoded_2():
    from litellm.utils import is_base64_encoded

    assert (
        is_base64_encoded(
            s="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/x+AAwMCAO+ip1sAAAAASUVORK5CYII="
        )
        is True
    )

    assert is_base64_encoded(s="Dog") is False


@pytest.mark.parametrize(
    "messages, expected_bool",
    [
        ([{"role": "user", "content": "hi"}], True),
        ([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], True),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "file_id": "123",
                                "file_name": "test.txt",
                                "file_size": 100,
                                "file_type": "text/plain",
                                "file_url": "https://example.com/test.txt",
                            },
                        }
                    ],
                }
            ],
            True,
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "url": "https://example.com/image.png"}
                    ],
                }
            ],
            True,
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image",
                            "source": {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "1234",
                                },
                            },
                        },
                    ],
                }
            ],
            False,
        ),
    ],
)
def test_validate_chat_completion_user_messages(messages, expected_bool):
    from litellm.utils import validate_chat_completion_user_messages

    if expected_bool:
        ## Valid message
        validate_chat_completion_user_messages(messages=messages)
    else:
        ## Invalid message
        with pytest.raises(Exception):
            validate_chat_completion_user_messages(messages=messages)


@pytest.mark.parametrize(
    "tool_choice, expected_bool",
    [
        ({"type": "function", "function": {"name": "get_current_weather"}}, True),
        ({"type": "tool", "name": "get_current_weather"}, False),
        (None, True),
        ("auto", True),
        ("required", True),
    ],
)
def test_validate_chat_completion_tool_choice(tool_choice, expected_bool):
    from litellm.utils import validate_chat_completion_tool_choice

    if expected_bool:
        validate_chat_completion_tool_choice(tool_choice=tool_choice)
    else:
        with pytest.raises(Exception):
            validate_chat_completion_tool_choice(tool_choice=tool_choice)


def test_models_by_provider():
    """
    Make sure all providers from model map are in the valid providers list
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm import models_by_provider

    providers = set()
    for k, v in litellm.model_cost.items():
        if "_" in v["litellm_provider"] and "-" in v["litellm_provider"]:
            continue
        elif k == "sample_spec":
            continue
        elif (
            v["litellm_provider"] == "sagemaker"
            or v["litellm_provider"] == "bedrock_converse"
        ):
            continue
        elif v.get("mode") == "search":
            # Skip search providers as they don't have traditional models
            continue
        else:
            providers.add(v["litellm_provider"])

    for provider in providers:
        assert provider in models_by_provider.keys()


@pytest.mark.parametrize(
    "litellm_params, disable_end_user_cost_tracking, expected_end_user_id",
    [
        ({}, False, None),
        ({"user_api_key_end_user_id": "123"}, False, "123"),
        ({"user_api_key_end_user_id": "123"}, True, None),
    ],
)
def test_get_end_user_id_for_cost_tracking(
    litellm_params, disable_end_user_cost_tracking, expected_end_user_id
):
    from litellm.utils import get_end_user_id_for_cost_tracking

    litellm.disable_end_user_cost_tracking = disable_end_user_cost_tracking
    assert (
        get_end_user_id_for_cost_tracking(litellm_params=litellm_params)
        == expected_end_user_id
    )


@pytest.mark.parametrize(
    "litellm_params, enable_end_user_cost_tracking_prometheus_only, expected_end_user_id",
    [
        ({}, True, None),
        ({"user_api_key_end_user_id": "123"}, True, "123"),
        ({"user_api_key_end_user_id": "123"}, False, None),
    ],
)
def test_get_end_user_id_for_cost_tracking_prometheus_only(
    litellm_params, enable_end_user_cost_tracking_prometheus_only, expected_end_user_id
):
    from litellm.utils import get_end_user_id_for_cost_tracking

    litellm.enable_end_user_cost_tracking_prometheus_only = (
        enable_end_user_cost_tracking_prometheus_only
    )
    assert (
        get_end_user_id_for_cost_tracking(
            litellm_params=litellm_params, service_type="prometheus"
        )
        == expected_end_user_id
    )


@pytest.mark.parametrize(
    "litellm_params, expected_end_user_id",
    [
        # Test with only metadata field (old behavior)
        ({"metadata": {"user_api_key_end_user_id": "user_from_metadata"}}, "user_from_metadata"),
        # Test with only litellm_metadata field (new behavior)
        ({"litellm_metadata": {"user_api_key_end_user_id": "user_from_litellm_metadata"}}, "user_from_litellm_metadata"),
        # Test with both fields - metadata should take precedence for user_api_key fields
        ({"metadata": {"user_api_key_end_user_id": "user_from_metadata"}, 
          "litellm_metadata": {"user_api_key_end_user_id": "user_from_litellm_metadata"}}, 
         "user_from_metadata"),
        # Test with user_api_key_end_user_id in litellm_params (should take precedence over metadata)
        ({"user_api_key_end_user_id": "user_from_params", 
          "metadata": {"user_api_key_end_user_id": "user_from_metadata"}}, 
         "user_from_params"),
        # Test with empty metadata but valid litellm_metadata
        ({"metadata": {}, "litellm_metadata": {"user_api_key_end_user_id": "user_from_litellm_metadata"}}, 
         "user_from_litellm_metadata"),
        # Test with no metadata fields
        ({}, None),
    ],
)
def test_get_end_user_id_for_cost_tracking_metadata_handling(
    litellm_params, expected_end_user_id
):
    """
    Test that get_end_user_id_for_cost_tracking correctly handles both metadata and litellm_metadata
    fields using the get_litellm_metadata_from_kwargs helper function.
    """
    from litellm.utils import get_end_user_id_for_cost_tracking
    
    # Ensure cost tracking is enabled for this test
    litellm.disable_end_user_cost_tracking = False
    
    result = get_end_user_id_for_cost_tracking(litellm_params=litellm_params)
    assert result == expected_end_user_id


def test_is_prompt_caching_enabled_error_handling():
    """
    Assert that `is_prompt_caching_valid_prompt` safely handles errors in `token_counter`.
    """
    with patch(
        "litellm.utils.token_counter",
        side_effect=Exception(
            "Mocked error, This should not raise an error. Instead is_prompt_caching_valid_prompt should return False."
        ),
    ):
        result = litellm.utils.is_prompt_caching_valid_prompt(
            messages=[{"role": "user", "content": "test"}],
            tools=None,
            custom_llm_provider="anthropic",
            model="anthropic/claude-sonnet-4-5-20250929",
        )

        assert result is False  # Should return False when an error occurs


def test_is_prompt_caching_enabled_return_default_image_dimensions():
    """
    Assert that `is_prompt_caching_valid_prompt` calls token_counter with use_default_image_token_count=True
    when processing messages containing images

    IMPORTANT: Ensures Get token counter does not make a GET request to the image url
    """
    with patch("litellm.utils.token_counter") as mock_token_counter:
        litellm.utils.is_prompt_caching_valid_prompt(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://www.gstatic.com/webp/gallery/1.webp",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            tools=None,
            custom_llm_provider="openai",
            model="gpt-4o-mini",
        )

        # Assert token_counter was called with use_default_image_token_count=True
        args_to_mock_token_counter = mock_token_counter.call_args[1]
        print("args_to_mock", args_to_mock_token_counter)
        assert args_to_mock_token_counter["use_default_image_token_count"] is True


def test_token_counter_with_image_url_with_detail_high():
    """
    Assert that token_counter does not make a GET request to the image url when `use_default_image_token_count=True`

    PROD TEST this is importat - Can impact latency very badly
    """
    from litellm.constants import DEFAULT_IMAGE_TOKEN_COUNT
    from litellm._logging import verbose_logger
    import logging

    verbose_logger.setLevel(logging.DEBUG)

    _tokens = litellm.utils.token_counter(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.gstatic.com/webp/gallery/1.webp",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        model="gpt-4o-mini",
        use_default_image_token_count=True,
    )
    print("tokens", _tokens)
    assert _tokens == DEFAULT_IMAGE_TOKEN_COUNT + 7


def test_fireworks_ai_document_inlining():
    """
    With document inlining, all fireworks ai models are now:
    - supports_pdf
    - supports_vision
    """
    from litellm.utils import supports_pdf_input, supports_vision

    litellm._turn_on_debug()

    assert supports_pdf_input("fireworks_ai/llama-3.1-8b-instruct") is True
    assert supports_vision("fireworks_ai/llama-3.1-8b-instruct") is True


def test_logprobs_type():
    from litellm.types.utils import Logprobs

    logprobs = {
        "text_offset": None,
        "token_logprobs": None,
        "tokens": None,
        "top_logprobs": None,
    }
    logprobs = Logprobs(**logprobs)
    assert logprobs.text_offset is None
    assert logprobs.token_logprobs is None
    assert logprobs.tokens is None
    assert logprobs.top_logprobs is None


def test_get_valid_models_openai_proxy(monkeypatch):
    from litellm.utils import get_valid_models
    import litellm

    litellm._turn_on_debug()

    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-1234")
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://litellm-api.up.railway.app/")
    monkeypatch.delenv("FIREWORKS_AI_ACCOUNT_ID", None)
    monkeypatch.delenv("FIREWORKS_AI_API_KEY", None)

    mock_response_data = {
        "object": "list",
        "data": [
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner",
            },
        ],
    }

    # Create a mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data

    with patch.object(
        litellm.module_level_client, "get", return_value=mock_response
    ) as mock_post:
        valid_models = get_valid_models(check_provider_endpoint=True)
        assert "litellm_proxy/gpt-4o" in valid_models


def test_get_valid_models_fireworks_ai(monkeypatch):
    from litellm.utils import get_valid_models
    import litellm

    litellm._turn_on_debug()

    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-1234")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "1234")
    monkeypatch.setattr(litellm, "provider_list", ["fireworks_ai"])

    mock_response_data = {
        "models": [
            {
                "name": "accounts/fireworks/models/llama-3.1-8b-instruct",
                "displayName": "<string>",
                "description": "<string>",
                "createTime": "2023-11-07T05:31:56Z",
                "createdBy": "<string>",
                "state": "STATE_UNSPECIFIED",
                "status": {"code": "OK", "message": "<string>"},
                "kind": "KIND_UNSPECIFIED",
                "githubUrl": "<string>",
                "huggingFaceUrl": "<string>",
                "baseModelDetails": {
                    "worldSize": 123,
                    "checkpointFormat": "CHECKPOINT_FORMAT_UNSPECIFIED",
                    "parameterCount": "<string>",
                    "moe": True,
                    "tunable": True,
                },
                "peftDetails": {
                    "baseModel": "<string>",
                    "r": 123,
                    "targetModules": ["<string>"],
                },
                "teftDetails": {},
                "public": True,
                "conversationConfig": {
                    "style": "<string>",
                    "system": "<string>",
                    "template": "<string>",
                },
                "contextLength": 123,
                "supportsImageInput": True,
                "supportsTools": True,
                "importedFrom": "<string>",
                "fineTuningJob": "<string>",
                "defaultDraftModel": "<string>",
                "defaultDraftTokenCount": 123,
                "precisions": ["PRECISION_UNSPECIFIED"],
                "deployedModelRefs": [
                    {
                        "name": "<string>",
                        "deployment": "<string>",
                        "state": "STATE_UNSPECIFIED",
                        "default": True,
                        "public": True,
                    }
                ],
                "cluster": "<string>",
                "deprecationDate": {"year": 123, "month": 123, "day": 123},
            }
        ],
        "nextPageToken": "<string>",
        "totalSize": 123,
    }

    # Create a mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data

    with patch.object(
        litellm.module_level_client, "get", return_value=mock_response
    ) as mock_post:
        valid_models = get_valid_models(check_provider_endpoint=True)
        print("valid_models", valid_models)
        mock_post.assert_called_once()
        assert (
            "fireworks_ai/accounts/fireworks/models/llama-3.1-8b-instruct"
            in valid_models
        )


def test_get_valid_models_default(monkeypatch):
    """
    Ensure that the default models is used when error retrieving from model api.

    Prevent regression for existing usage.
    """
    from litellm.utils import get_valid_models
    import litellm

    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-1234")
    valid_models = get_valid_models()
    assert len(valid_models) > 0


def test_supports_vision_gemini():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    from litellm.utils import supports_vision

    assert supports_vision("gemini-1.5-pro") is True


def test_pick_cheapest_chat_model_from_llm_provider():
    from litellm.litellm_core_utils.llm_request_utils import (
        pick_cheapest_chat_models_from_llm_provider,
    )

    assert len(pick_cheapest_chat_models_from_llm_provider("openai", n=3)) == 3

    assert len(pick_cheapest_chat_models_from_llm_provider("unknown", n=1)) == 0


@pytest.mark.parametrize("num_retries", [0, 1, 5])
def test_get_num_retries(num_retries):
    from litellm.utils import _get_wrapper_num_retries

    assert _get_wrapper_num_retries(
        kwargs={"num_retries": num_retries}, exception=Exception("test")
    ) == (
        num_retries,
        {
            "num_retries": num_retries,
        },
    )


def test_add_custom_logger_callback_to_specific_event(monkeypatch):
    from litellm.utils import _add_custom_logger_callback_to_specific_event

    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])

    _add_custom_logger_callback_to_specific_event("langfuse", "success")

    assert len(litellm.success_callback) == 1
    assert len(litellm.failure_callback) == 0


def test_add_custom_logger_callback_to_specific_event_e2e(monkeypatch):

    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])

    litellm.success_callback = ["humanloop"]

    curr_len_success_callback = len(litellm.success_callback)
    curr_len_failure_callback = len(litellm.failure_callback)

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Testing langfuse",
    )

    assert len(litellm.success_callback) == curr_len_success_callback
    assert len(litellm.failure_callback) == curr_len_failure_callback


def test_custom_logger_exists_in_callbacks_individual_functions(monkeypatch):
    """
    Test _custom_logger_class_exists_in_success_callbacks and _custom_logger_class_exists_in_failure_callbacks helper functions
    Tests if logger is found in different callback lists
    """
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.utils import (
        _custom_logger_class_exists_in_failure_callbacks,
        _custom_logger_class_exists_in_success_callbacks,
    )

    # Create a mock CustomLogger class
    class MockCustomLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            pass

    # Reset all callback lists
    for list_name in [
        "callbacks",
        "_async_success_callback",
        "_async_failure_callback",
        "success_callback",
        "failure_callback",
    ]:
        monkeypatch.setattr(litellm, list_name, [])

    mock_logger = MockCustomLogger()

    # Test 1: No logger exists in any callback list
    assert _custom_logger_class_exists_in_success_callbacks(mock_logger) == False
    assert _custom_logger_class_exists_in_failure_callbacks(mock_logger) == False

    # Test 2: Logger exists in success_callback
    litellm.success_callback.append(mock_logger)
    assert _custom_logger_class_exists_in_success_callbacks(mock_logger) == True
    assert _custom_logger_class_exists_in_failure_callbacks(mock_logger) == False

    # Reset callbacks
    litellm.success_callback = []

    # Test 3: Logger exists in _async_success_callback
    litellm._async_success_callback.append(mock_logger)
    assert _custom_logger_class_exists_in_success_callbacks(mock_logger) == True
    assert _custom_logger_class_exists_in_failure_callbacks(mock_logger) == False

    # Reset callbacks
    litellm._async_success_callback = []

    # Test 4: Logger exists in failure_callback
    litellm.failure_callback.append(mock_logger)
    assert _custom_logger_class_exists_in_success_callbacks(mock_logger) == False
    assert _custom_logger_class_exists_in_failure_callbacks(mock_logger) == True

    # Reset callbacks
    litellm.failure_callback = []

    # Test 5: Logger exists in _async_failure_callback
    litellm._async_failure_callback.append(mock_logger)
    assert _custom_logger_class_exists_in_success_callbacks(mock_logger) == False
    assert _custom_logger_class_exists_in_failure_callbacks(mock_logger) == True

    # Test 6: Logger exists in both success and failure callbacks
    litellm.success_callback.append(mock_logger)
    litellm.failure_callback.append(mock_logger)
    assert _custom_logger_class_exists_in_success_callbacks(mock_logger) == True
    assert _custom_logger_class_exists_in_failure_callbacks(mock_logger) == True

    # Test 7: Different instance of same logger class
    mock_logger_2 = MockCustomLogger()
    assert _custom_logger_class_exists_in_success_callbacks(mock_logger_2) == True
    assert _custom_logger_class_exists_in_failure_callbacks(mock_logger_2) == True


@pytest.mark.asyncio
async def test_add_custom_logger_callback_to_specific_event_with_duplicates(
    monkeypatch,
):
    """
    Test that when a callback exists in both success_callback and _async_success_callback,
    it's not added again
    """
    from litellm.integrations.langfuse.langfuse_prompt_management import (
        LangfusePromptManagement,
    )

    # Reset all callback lists
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])

    # Add logger to both success_callback and _async_success_callback
    langfuse_logger = LangfusePromptManagement()
    litellm.success_callback.append(langfuse_logger)
    litellm._async_success_callback.append(langfuse_logger)

    # Get initial lengths
    initial_success_callback_len = len(litellm.success_callback)
    initial_async_success_callback_len = len(litellm._async_success_callback)

    # Make a completion call
    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Testing duplicate callbacks",
    )

    # Assert no new callbacks were added
    assert len(litellm.success_callback) == initial_success_callback_len
    assert len(litellm._async_success_callback) == initial_async_success_callback_len


@pytest.mark.asyncio
async def test_add_custom_logger_callback_to_specific_event_with_duplicates_success_callback(
    monkeypatch,
):
    """
    Test that when a callback exists in both success_callback and _async_success_callback,
    it's not added again
    """
    from litellm.integrations.langfuse.langfuse_prompt_management import (
        LangfusePromptManagement,
    )

    # Reset all callback lists
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])

    # Add logger to both success_callback and _async_success_callback
    langfuse_logger = LangfusePromptManagement()
    litellm.success_callback.append(langfuse_logger)

    # Get initial lengths
    initial_success_callback_len = len(litellm.success_callback)
    initial_async_success_callback_len = len(litellm._async_success_callback)

    # Make a completion call
    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Testing duplicate callbacks",
    )

    # Assert no new callbacks were added
    assert len(litellm.success_callback) == initial_success_callback_len
    assert len(litellm._async_success_callback) == initial_async_success_callback_len


@pytest.mark.asyncio
async def test_add_custom_logger_callback_to_specific_event_with_duplicates_callbacks(
    monkeypatch,
):
    """
    Test that when a callback exists in both success_callback and _async_success_callback,
    it's not added again
    """
    from litellm.integrations.langfuse.langfuse_prompt_management import (
        LangfusePromptManagement,
    )

    # Reset all callback lists
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])

    # Add logger to both success_callback and _async_success_callback
    langfuse_logger = LangfusePromptManagement()
    litellm.callbacks.append(langfuse_logger)

    # Make a completion call
    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Testing duplicate callbacks",
    )

    # Assert no new callbacks were added
    initial_callbacks_len = len(litellm.callbacks)
    initial_async_success_callback_len = len(litellm._async_success_callback)
    initial_success_callback_len = len(litellm.success_callback)
    print(
        f"Num callbacks before: litellm.callbacks: {len(litellm.callbacks)}, litellm._async_success_callback: {len(litellm._async_success_callback)}, litellm.success_callback: {len(litellm.success_callback)}"
    )

    for _ in range(10):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="Testing duplicate callbacks",
        )

    assert len(litellm.callbacks) == initial_callbacks_len
    assert len(litellm._async_success_callback) == initial_async_success_callback_len
    assert len(litellm.success_callback) == initial_success_callback_len

    print(
        f"Num callbacks after 10 mock calls: litellm.callbacks: {len(litellm.callbacks)}, litellm._async_success_callback: {len(litellm._async_success_callback)}, litellm.success_callback: {len(litellm.success_callback)}"
    )


def test_add_custom_logger_callback_to_specific_event_e2e_failure(monkeypatch):
    from litellm.integrations.openmeter import OpenMeterLogger

    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setenv("OPENMETER_API_KEY", "wedlwe")
    monkeypatch.setenv("OPENMETER_API_URL", "https://openmeter.dev")

    litellm.failure_callback = ["openmeter"]

    curr_len_success_callback = len(litellm.success_callback)
    curr_len_failure_callback = len(litellm.failure_callback)

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Testing langfuse",
    )

    assert len(litellm.success_callback) == curr_len_success_callback
    assert len(litellm.failure_callback) == curr_len_failure_callback

    assert any(
        isinstance(callback, OpenMeterLogger) for callback in litellm.failure_callback
    )


@pytest.mark.asyncio
async def test_wrapper_kwargs_passthrough():
    from litellm.utils import client
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObject,
    )

    # Create mock original function
    mock_original = AsyncMock()

    # Apply decorator
    @client
    async def test_function(**kwargs):
        return await mock_original(**kwargs)

    # Test kwargs
    test_kwargs = {"base_model": "gpt-4o-mini"}

    # Call decorated function
    await test_function(**test_kwargs)

    mock_original.assert_called_once()

    # get litellm logging object
    litellm_logging_obj: LiteLLMLoggingObject = mock_original.call_args.kwargs.get(
        "litellm_logging_obj"
    )
    assert litellm_logging_obj is not None

    print(
        f"litellm_logging_obj.model_call_details: {litellm_logging_obj.model_call_details}"
    )

    # get base model
    assert (
        litellm_logging_obj.model_call_details["litellm_params"]["base_model"]
        == "gpt-4o-mini"
    )


def test_dict_to_response_format_helper():
    from litellm.llms.base_llm.base_utils import _dict_to_response_format_helper

    args = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "$defs": {
                        "CalendarEvent": {
                            "properties": {
                                "name": {"title": "Name", "type": "string"},
                                "date": {"title": "Date", "type": "string"},
                                "participants": {
                                    "items": {"type": "string"},
                                    "title": "Participants",
                                    "type": "array",
                                },
                            },
                            "required": ["name", "date", "participants"],
                            "title": "CalendarEvent",
                            "type": "object",
                            "additionalProperties": False,
                        }
                    },
                    "properties": {
                        "events": {
                            "items": {"$ref": "#/$defs/CalendarEvent"},
                            "title": "Events",
                            "type": "array",
                        }
                    },
                    "required": ["events"],
                    "title": "EventsList",
                    "type": "object",
                    "additionalProperties": False,
                },
                "name": "EventsList",
                "strict": True,
            },
        },
        "ref_template": "/$defs/{model}",
    }
    _dict_to_response_format_helper(**args)


def test_validate_user_messages_invalid_content_type():
    from litellm.utils import validate_chat_completion_user_messages

    messages = [{"content": [{"type": "invalid_type", "text": "Hello"}]}]

    with pytest.raises(Exception) as e:
        validate_chat_completion_user_messages(messages)

    assert "Invalid message" in str(e)
    print(e)


from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.utils import get_applied_guardrails
from unittest.mock import Mock


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "default_on_guardrail",
            "callbacks": [
                CustomGuardrail(guardrail_name="test_guardrail", default_on=True)
            ],
            "kwargs": {"metadata": {"requester_metadata": {"guardrails": []}}},
            "expected": ["test_guardrail"],
        },
        {
            "name": "request_specific_guardrail",
            "callbacks": [
                CustomGuardrail(guardrail_name="test_guardrail", default_on=False)
            ],
            "kwargs": {
                "metadata": {"requester_metadata": {"guardrails": ["test_guardrail"]}}
            },
            "expected": ["test_guardrail"],
        },
        {
            "name": "multiple_guardrails",
            "callbacks": [
                CustomGuardrail(guardrail_name="default_guardrail", default_on=True),
                CustomGuardrail(guardrail_name="request_guardrail", default_on=False),
            ],
            "kwargs": {
                "metadata": {
                    "requester_metadata": {"guardrails": ["request_guardrail"]}
                }
            },
            "expected": ["default_guardrail", "request_guardrail"],
        },
        {
            "name": "empty_metadata",
            "callbacks": [
                CustomGuardrail(guardrail_name="test_guardrail", default_on=False)
            ],
            "kwargs": {},
            "expected": [],
        },
        {
            "name": "none_callback",
            "callbacks": [
                None,
                CustomGuardrail(guardrail_name="test_guardrail", default_on=True),
            ],
            "kwargs": {},
            "expected": ["test_guardrail"],
        },
        {
            "name": "non_guardrail_callback",
            "callbacks": [
                Mock(),
                CustomGuardrail(guardrail_name="test_guardrail", default_on=True),
            ],
            "kwargs": {},
            "expected": ["test_guardrail"],
        },
    ],
)
def test_get_applied_guardrails(test_case):

    # Setup
    litellm.callbacks = test_case["callbacks"]

    # Execute
    result = get_applied_guardrails(test_case["kwargs"])

    # Assert
    assert sorted(result) == sorted(test_case["expected"])


@pytest.mark.parametrize(
    "endpoint, params, expected_bool",
    [
        ("localhost:4000/v1/rerank", ["max_chunks_per_doc"], True),
        ("localhost:4000/v2/rerank", ["max_chunks_per_doc"], False),
        ("localhost:4000", ["max_chunks_per_doc"], True),
        ("localhost:4000/v1/rerank", ["max_tokens_per_doc"], True),
        ("localhost:4000/v2/rerank", ["max_tokens_per_doc"], False),
        ("localhost:4000", ["max_tokens_per_doc"], False),
        (
            "localhost:4000/v1/rerank",
            ["max_chunks_per_doc", "max_tokens_per_doc"],
            True,
        ),
        (
            "localhost:4000/v2/rerank",
            ["max_chunks_per_doc", "max_tokens_per_doc"],
            False,
        ),
        ("localhost:4000", ["max_chunks_per_doc", "max_tokens_per_doc"], False),
    ],
)
def test_should_use_cohere_v1_client(endpoint, params, expected_bool):
    assert litellm.utils.should_use_cohere_v1_client(endpoint, params) == expected_bool


def test_add_openai_metadata():
    from litellm.utils import add_openai_metadata

    metadata = {
        "user_api_key_end_user_id": "123",
        "hidden_params": {"api_key": "123"},
        "litellm_parent_otel_span": MagicMock(),
        "none-val": None,
        "int-val": 1,
        "dict-val": {"a": 1, "b": 2},
    }

    result = add_openai_metadata(metadata)

    assert result == {
        "user_api_key_end_user_id": "123",
    }


def test_message_object():
    from litellm.types.utils import Message

    message = Message(content="Hello, world!", role="user")
    assert message.content == "Hello, world!"
    assert message.role == "user"
    assert not hasattr(message, "audio")
    assert not hasattr(message, "thinking_blocks")
    assert not hasattr(message, "reasoning_content")


def test_delta_object():
    from litellm.types.utils import Delta

    delta = Delta(content="Hello, world!", role="user")
    assert delta.content == "Hello, world!"
    assert delta.role == "user"
    assert not hasattr(delta, "thinking_blocks")
    assert not hasattr(delta, "reasoning_content")


def test_get_provider_audio_transcription_config():
    from litellm.utils import ProviderConfigManager
    from litellm.types.utils import LlmProviders

    for provider in LlmProviders:
        config = ProviderConfigManager.get_provider_audio_transcription_config(
            model="whisper-1", provider=provider
        )


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("anthropic.claude-3-7-sonnet-20250219-v1:0", True),
        ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", True),
    ],
)
def test_claude_3_7_sonnet_supports_pdf_input(model, expected_bool):
    from litellm.utils import supports_pdf_input

    assert supports_pdf_input(model) == expected_bool


def test_get_valid_models_from_provider():
    """
    Test that get_valid_models returns the correct models for a given provider
    """
    from litellm.utils import get_valid_models

    valid_models = get_valid_models(custom_llm_provider="openai")
    assert len(valid_models) > 0
    assert "gpt-4o-mini" in valid_models

    print("Valid models: ", valid_models)
    valid_models.remove("gpt-4o-mini")
    assert "gpt-4o-mini" not in valid_models

    valid_models = get_valid_models(custom_llm_provider="openai")
    assert len(valid_models) > 0
    assert "gpt-4o-mini" in valid_models


def test_get_valid_models_from_provider_cache_invalidation(monkeypatch):
    """
    Test that get_valid_models returns the correct models for a given provider
    """
    from litellm.utils import _model_cache

    monkeypatch.setenv("OPENAI_API_KEY", "123")

    _model_cache.set_cached_model_info(
        "openai", litellm_params=None, available_models=["gpt-4o-mini"]
    )
    monkeypatch.delenv("OPENAI_API_KEY")

    assert _model_cache.get_cached_model_info("openai") is None


def test_get_valid_models_from_dynamic_api_key():
    """
    Test that get_valid_models returns the correct models for a given provider
    """
    from litellm.utils import get_valid_models
    from litellm.types.router import CredentialLiteLLMParams

    creds = CredentialLiteLLMParams(api_key="123")

    valid_models = get_valid_models(
        custom_llm_provider="anthropic",
        litellm_params=creds,
        check_provider_endpoint=True,
    )
    assert len(valid_models) == 0

    creds = CredentialLiteLLMParams(api_key=os.getenv("ANTHROPIC_API_KEY"))
    valid_models = get_valid_models(
        custom_llm_provider="anthropic",
        litellm_params=creds,
        check_provider_endpoint=True,
    )
    assert len(valid_models) > 0
    assert "anthropic/claude-3-7-sonnet-20250219" in valid_models


def test_get_whitelisted_models():
    """
    Snapshot of all bedrock models as of 12/24/2024.

    Enforce any new bedrock chat model to be added as `bedrock_converse` unless explicitly whitelisted.

    Create whitelist to prevent naming regressions for older litellm versions.
    """
    whitelisted_models = []
    for model, info in litellm.model_cost.items():
        if info.get("litellm_provider") == "bedrock" and info.get("mode") == "chat":
            whitelisted_models.append(model)

        # Write to a local file
    with open("whitelisted_bedrock_models.txt", "w") as file:
        for model in whitelisted_models:
            file.write(f"{model}\n")

    print("whitelisted_models written to whitelisted_bedrock_models.txt")


def test_delta_tool_calls_sequential_indices():
    """
    Test that multiple tool calls without explicit indices receive sequential indices.

    When providers don't include index fields in tool calls, the Delta class
    should automatically assign sequential indices (0, 1, 2, ...) instead of
    defaulting all tool calls to index=0.
    """
    import json
    from litellm.types.utils import Delta

    # Simulate tool calls from streaming responses without explicit indices
    tool_calls_without_indices = [
        {
            "id": "call_1",
            "function": {
                "name": "get_weather_for_dallas",
                "arguments": json.dumps({})
            },
            "type": "function",
            # Note: no "index" field - simulates provider response
        },
        {
            "id": "call_2",
            "function": {
                "name": "get_weather_precise",
                "arguments": json.dumps({"location": "Dallas, TX"})
            },
            "type": "function",
            # Note: no "index" field - simulates provider response
        }
    ]

    # Create Delta object as LiteLLM would when processing streaming response
    delta = Delta(
        content=None,
        tool_calls=tool_calls_without_indices
    )

    # Verify tool calls have sequential indices
    assert delta.tool_calls is not None, "Tool calls should not be None"
    assert len(delta.tool_calls) == 2
    assert delta.tool_calls[0].index == 0, f"First tool call should have index 0, got {delta.tool_calls[0].index}"
    assert delta.tool_calls[1].index == 1, f"Second tool call should have index 1, got {delta.tool_calls[1].index}"

    # Verify tool call details are preserved
    assert delta.tool_calls[0].function.name == "get_weather_for_dallas"
    assert delta.tool_calls[1].function.name == "get_weather_precise"

def test_completion_with_no_model():
    """
    Ensure error is raised when no model is provided
    """
    # test on empty
    with pytest.raises(TypeError):
        response = litellm.completion(messages=[{"role": "user", "content": "Hello, how are you?"}])


def test_get_base_model_from_metadata():
    """
    Test _get_base_model_from_metadata function with both metadata and litellm_metadata.
    This ensures cost tracking works for both Chat Completions API and Responses API.

    Related issue: https://github.com/BerriAI/litellm/issues/16772
    """
    from litellm.utils import _get_base_model_from_metadata

    # Test 1: base_model in metadata (Chat Completions API pattern)
    model_call_details_with_metadata = {
        "litellm_params": {
            "metadata": {
                "model_info": {
                    "base_model": "azure/gpt-4"
                }
            }
        }
    }
    result = _get_base_model_from_metadata(model_call_details_with_metadata)
    assert result == "azure/gpt-4", f"Expected 'azure/gpt-4', got {result}"

    # Test 2: base_model in litellm_metadata (Responses API and generic API calls pattern)
    model_call_details_with_litellm_metadata = {
        "litellm_params": {
            "litellm_metadata": {
                "model_info": {
                    "base_model": "azure/gpt-5-mini"
                }
            }
        }
    }
    result = _get_base_model_from_metadata(model_call_details_with_litellm_metadata)
    assert result == "azure/gpt-5-mini", f"Expected 'azure/gpt-5-mini', got {result}"

    # Test 3: base_model in litellm_params (direct base_model)
    model_call_details_with_direct_base_model = {
        "litellm_params": {
            "base_model": "azure/gpt-3.5-turbo"
        }
    }
    result = _get_base_model_from_metadata(model_call_details_with_direct_base_model)
    assert result == "azure/gpt-3.5-turbo", f"Expected 'azure/gpt-3.5-turbo', got {result}"

    # Test 4: metadata takes precedence over litellm_metadata
    model_call_details_with_both = {
        "litellm_params": {
            "metadata": {
                "model_info": {
                    "base_model": "azure/gpt-4-from-metadata"
                }
            },
            "litellm_metadata": {
                "model_info": {
                    "base_model": "azure/gpt-4-from-litellm-metadata"
                }
            }
        }
    }
    result = _get_base_model_from_metadata(model_call_details_with_both)
    assert result == "azure/gpt-4-from-metadata", f"Expected metadata to take precedence, got {result}"

    # Test 5: No base_model present
    model_call_details_without_base_model = {
        "litellm_params": {
            "metadata": {}
        }
    }
    result = _get_base_model_from_metadata(model_call_details_without_base_model)
    assert result is None, f"Expected None when no base_model present, got {result}"

    # Test 6: None input
    result = _get_base_model_from_metadata(None)
    assert result is None, f"Expected None for None input, got {result}"

