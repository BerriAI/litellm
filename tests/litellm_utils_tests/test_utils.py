import copy
import logging
import time
from datetime import datetime
from unittest import mock

from dotenv import load_dotenv

from litellm.types.utils import StandardCallbackDynamicParams

load_dotenv()
import os

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
    get_llm_provider,
    get_supported_openai_params,
    get_token_count,
    get_valid_models,
    trim_messages,
    validate_environment,
)
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from unittest.mock import AsyncMock, MagicMock, patch


# Assuming your trim_messages, shorten_message_to_fit_limit, and get_token_count functions are all in a module named 'message_utils'
@pytest.fixture(autouse=True)
def reset_mock_cache():
    from litellm.utils import _model_cache

    _model_cache.flush_cache()


# Test 1: Check trimming of normal message


# test_basic_trimming()




# test_basic_trimming_no_max_tokens_specified()




# test_multiple_messages_trimming()




# test_multiple_messages_no_trimming()




# test_large_trimming()


















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
    response = check_valid_key(model="gpt-5-mini", api_key=key)
    print(response, key)
    assert response == False


def test_good_key():
    key = os.environ["OPENAI_API_KEY"]
    response = check_valid_key(model="gpt-5-mini", api_key=key)
    assert response == True


# test validate environment














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






@pytest.mark.parametrize("langfuse_trace_id", [None, "my-unique-trace-id"])
@pytest.mark.parametrize(
    "langfuse_existing_trace_id", [None, "my-unique-existing-trace-id"]
)
def test_logging_trace_id(langfuse_trace_id, langfuse_existing_trace_id):
    """
    - Unit test for `_get_trace_id` function in Logging obj
    """
    from litellm.integrations.langfuse.langfuse_sdk import resolve_trace_id
    from litellm.litellm_core_utils.litellm_logging import Logging

    litellm.success_callback = ["langfuse"]
    litellm_call_id = "my-unique-call-id"
    litellm_logging_obj = Logging(
        model="gpt-5-mini",
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
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hey how's it going?"}],
        mock_response="Hey!",
        litellm_logging_obj=litellm_logging_obj,
        metadata=metadata,
    )

    time.sleep(3)
    assert litellm_logging_obj.get_trace_id(service_name="langfuse") is not None

    # langfuse addresses a trace by a 32-hex id, so the id litellm reports back is the
    # resolved form of whichever source won; that is what the alerting deep link needs
    if langfuse_existing_trace_id is not None:
        expected_source = langfuse_existing_trace_id
    elif langfuse_trace_id is not None:
        expected_source = langfuse_trace_id
    else:
        expected_source = litellm_logging_obj.litellm_trace_id

    assert litellm_logging_obj.get_trace_id(service_name="langfuse") == resolve_trace_id(
        expected_source
    )








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






















def test_is_prompt_caching_enabled_return_default_image_dimensions():
    """
    Assert that `is_prompt_caching_valid_prompt` counts tokens with use_default_image_token_count=True
    when processing messages containing images

    IMPORTANT: Ensures Get token counter does not make a GET request to the image url
    """
    mock_token_counter = MagicMock(return_value=False)
    with patch(
        "litellm.utils.get_messages_reach_token_count",
        return_value=mock_token_counter,
    ):
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








def test_get_valid_models_fireworks_ai(monkeypatch):
    from litellm.utils import get_valid_models
    import litellm

    litellm.turn_on_debug()

    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-9876")
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

    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-9876")
    valid_models = get_valid_models()
    assert len(valid_models) > 0






def test_add_custom_logger_callback_to_specific_event(monkeypatch):
    from litellm.utils import add_custom_logger_callback_to_specific_event

    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])

    add_custom_logger_callback_to_specific_event("langfuse", "success")

    assert len(litellm.success_callback) == 1
    assert len(litellm.failure_callback) == 0






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
        model="gpt-5-mini",
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
        model="gpt-5-mini",
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
        model="gpt-5-mini",
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
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="Testing duplicate callbacks",
        )

    assert len(litellm.callbacks) == initial_callbacks_len
    assert len(litellm._async_success_callback) == initial_async_success_callback_len
    assert len(litellm.success_callback) == initial_success_callback_len

    print(
        f"Num callbacks after 10 mock calls: litellm.callbacks: {len(litellm.callbacks)}, litellm._async_success_callback: {len(litellm._async_success_callback)}, litellm.success_callback: {len(litellm.success_callback)}"
    )










from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.utils import get_applied_guardrails
from unittest.mock import Mock












def test_get_provider_audio_transcription_config():
    from litellm.utils import ProviderConfigManager
    from litellm.types.utils import LlmProviders

    for provider in LlmProviders:
        config = ProviderConfigManager.get_provider_audio_transcription_config(
            model="whisper-1", provider=provider
        )








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
    assert "anthropic/claude-sonnet-4-6" in valid_models


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
