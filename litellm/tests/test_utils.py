import copy
import sys
import time
from datetime import datetime
from unittest import mock

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm.proxy.utils import (
    _duration_in_seconds,
    _extract_from_regex,
    get_last_day_of_month,
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
    token_counter,
    trim_messages,
    validate_environment,
)

# Assuming your trim_messages, shorten_message_to_fit_limit, and get_token_count functions are all in a module named 'message_utils'


# Test 1: Check trimming of normal message
def test_basic_trimming():
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


def test_aget_valid_models():
    old_environ = os.environ
    os.environ = {"OPENAI_API_KEY": "temp"}  # mock set only openai key in environ

    valid_models = get_valid_models()
    print(valid_models)

    # list of openai supported llms on litellm
    expected_models = (
        litellm.open_ai_chat_completion_models + litellm.open_ai_text_completion_models
    )

    assert valid_models == expected_models

    # reset replicate env key
    os.environ = old_environ

    # GEMINI
    expected_models = litellm.gemini_models
    old_environ = os.environ
    os.environ = {"GEMINI_API_KEY": "temp"}  # mock set only openai key in environ

    valid_models = get_valid_models()

    print(valid_models)
    assert valid_models == expected_models

    # reset replicate env key
    os.environ = old_environ


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


def test_token_counter():
    try:
        messages = [{"role": "user", "content": "hi how are you what time is it"}]
        tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
        print("gpt-35-turbo")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="claude-2", messages=messages)
        print("claude-2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="palm/chat-bison", messages=messages)
        print("palm/chat-bison")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="ollama/llama2", messages=messages)
        print("ollama/llama2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="anthropic.claude-instant-v1", messages=messages)
        print("anthropic.claude-instant-v1")
        print(tokens)
        assert tokens > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_token_counter()


def test_supports_function_calling():
    try:
        assert litellm.supports_function_calling(model="gpt-3.5-turbo") == True
        assert (
            litellm.supports_function_calling(model="azure/gpt-4-1106-preview") == True
        )
        assert litellm.supports_function_calling(model="groq/gemma-7b-it") == True
        assert (
            litellm.supports_function_calling(model="anthropic.claude-instant-v1")
            == False
        )
        assert litellm.supports_function_calling(model="palm/chat-bison") == False
        assert litellm.supports_function_calling(model="ollama/llama2") == False
        assert (
            litellm.supports_function_calling(model="anthropic.claude-instant-v1")
            == False
        )
        assert litellm.supports_function_calling(model="claude-2") == False
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

    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        litellm_logging_obj=Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="acompletion",
            litellm_call_id="1234",
            start_time=datetime.now(),
            function_id="1234",
        ),
    )

    # Assert the response_obj content is NOT modified
    assert (
        response_obj.choices[0].message.content
        == "I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner."
    )

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

    value = _duration_in_seconds(duration="1mo")

    assert value - expected_duration < 2


def test_get_llm_provider_ft_models():
    """
    All ft prefixed models should map to OpenAI
    gpt-3.5-turbo-0125 (recommended),
    gpt-3.5-turbo-1106,
    gpt-3.5-turbo-0613,
    gpt-4-0613 (experimental)
    gpt-4o-2024-05-13.
    babbage-002, davinci-002,

    """
    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-0125")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-1106")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-0613")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-4-0613")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-0613")
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
