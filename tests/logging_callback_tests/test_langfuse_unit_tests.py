import os
import sys
import threading
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import pytest
from litellm.integrations.langfuse.langfuse import (
    LangFuseLogger,
)
from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
from unittest.mock import Mock, patch
from respx import MockRouter
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingModelInformation,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
    StandardCallbackDynamicParams,
    ModelResponse,
    Choices,
    Message,
    TextCompletionResponse,
    TextChoices,
)


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        response_cost=0.1,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-3.5-turbo", model_map_value=None
        ),
        model="gpt-3.5-turbo",
        model_id="model-123",
        model_group="openai-gpt",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_org_id=None,
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_team_alias="test_team_alias",
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
    )


@pytest.fixture
def dynamic_logging_cache():
    return DynamicLoggingCache()


global_langfuse_logger = LangFuseLogger(
    langfuse_public_key="global_public_key",
    langfuse_secret="global_secret",
    langfuse_host="https://global.langfuse.com",
)


# IMPORTANT: Test that passing both langfuse_secret_key and langfuse_secret works
standard_params_1 = StandardCallbackDynamicParams(
    langfuse_public_key="test_public_key",
    langfuse_secret="test_secret",
    langfuse_host="https://test.langfuse.com",
)

standard_params_2 = StandardCallbackDynamicParams(
    langfuse_public_key="test_public_key",
    langfuse_secret_key="test_secret",
    langfuse_host="https://test.langfuse.com",
)


@pytest.mark.parametrize("globalLangfuseLogger", [None, global_langfuse_logger])
@pytest.mark.parametrize("standard_params", [standard_params_1, standard_params_2])
def test_get_langfuse_logger_for_request_with_dynamic_params(
    dynamic_logging_cache, globalLangfuseLogger, standard_params
):
    """
    If StandardCallbackDynamicParams contain langfuse credentials the returned Langfuse logger should use the dynamic params

    the new Langfuse logger should be cached

    Even if globalLangfuseLogger is provided, it should use dynamic params if they are passed
    """

    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=standard_params,
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=globalLangfuseLogger,
    )

    assert isinstance(result, LangFuseLogger)
    assert result.public_key == "test_public_key"
    assert result.secret_key == "test_secret"
    assert result.langfuse_host == "https://test.langfuse.com"

    print("langfuse logger=", result)
    print("vars in langfuse logger=", vars(result))

    # Check if the logger is cached
    cached_logger = dynamic_logging_cache.get_cache(
        credentials={
            "langfuse_public_key": "test_public_key",
            "langfuse_secret": "test_secret",
            "langfuse_host": "https://test.langfuse.com",
        },
        service_name="langfuse",
    )
    assert cached_logger is result


@pytest.mark.parametrize("globalLangfuseLogger", [None, global_langfuse_logger])
def test_get_langfuse_logger_for_request_with_no_dynamic_params(
    dynamic_logging_cache, globalLangfuseLogger
):
    """
    If StandardCallbackDynamicParams are not provided, the globalLangfuseLogger should be returned
    """
    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=StandardCallbackDynamicParams(),
        in_memory_dynamic_logger_cache=dynamic_logging_cache,
        globalLangfuseLogger=globalLangfuseLogger,
    )

    assert result is not None
    assert isinstance(result, LangFuseLogger)

    print("langfuse logger=", result)

    if globalLangfuseLogger is not None:
        assert result.public_key == "global_public_key"
        assert result.secret_key == "global_secret"
        assert result.langfuse_host == "https://global.langfuse.com"


def test_dynamic_langfuse_credentials_are_passed():
    # Test when credentials are passed
    params_with_credentials = StandardCallbackDynamicParams(
        langfuse_public_key="test_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )
    assert (
        LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            params_with_credentials
        )
        is True
    )

    # Test when no credentials are passed
    params_without_credentials = StandardCallbackDynamicParams()
    assert (
        LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            params_without_credentials
        )
        is False
    )

    # Test when only some credentials are passed
    params_partial_credentials = StandardCallbackDynamicParams(
        langfuse_public_key="test_key"
    )
    assert (
        LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            params_partial_credentials
        )
        is True
    )


def test_get_dynamic_langfuse_logging_config():
    # Test with dynamic params
    dynamic_params = StandardCallbackDynamicParams(
        langfuse_public_key="dynamic_key",
        langfuse_secret="dynamic_secret",
        langfuse_host="https://dynamic.langfuse.com",
    )
    config = LangFuseHandler.get_dynamic_langfuse_logging_config(dynamic_params)
    assert config["langfuse_public_key"] == "dynamic_key"
    assert config["langfuse_secret"] == "dynamic_secret"
    assert config["langfuse_host"] == "https://dynamic.langfuse.com"

    # Test with no dynamic params
    empty_params = StandardCallbackDynamicParams()
    config = LangFuseHandler.get_dynamic_langfuse_logging_config(empty_params)
    assert config["langfuse_public_key"] is None
    assert config["langfuse_secret"] is None
    assert config["langfuse_host"] is None


def test_return_global_langfuse_logger():
    mock_cache = Mock()
    global_logger = LangFuseLogger(
        langfuse_public_key="global_key", langfuse_secret="global_secret"
    )

    # Test with existing global logger
    result = LangFuseHandler._return_global_langfuse_logger(global_logger, mock_cache)
    assert result == global_logger

    # Test without global logger, but with cached logger, should return cached logger
    mock_cache.get_cache.return_value = global_logger
    result = LangFuseHandler._return_global_langfuse_logger(None, mock_cache)
    assert result == global_logger

    # Test without global logger and without cached logger, should create new logger
    mock_cache.get_cache.return_value = None
    with patch.object(
        LangFuseHandler,
        "_create_langfuse_logger_from_credentials",
        return_value=global_logger,
    ):
        result = LangFuseHandler._return_global_langfuse_logger(None, mock_cache)
        assert result == global_logger


def test_get_langfuse_logger_for_request_with_cached_logger():
    """
    Test that get_langfuse_logger_for_request returns the cached logger if it exists when dynamic params are passed
    """
    mock_cache = Mock()
    cached_logger = LangFuseLogger(
        langfuse_public_key="cached_key", langfuse_secret="cached_secret"
    )
    mock_cache.get_cache.return_value = cached_logger

    dynamic_params = StandardCallbackDynamicParams(
        langfuse_public_key="test_key",
        langfuse_secret="test_secret",
        langfuse_host="https://test.langfuse.com",
    )

    result = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=dynamic_params,
        in_memory_dynamic_logger_cache=mock_cache,
        globalLangfuseLogger=None,
    )

    assert result == cached_logger
    mock_cache.get_cache.assert_called_once()


def test_get_langfuse_tags():
    """
    Test that _get_langfuse_tags correctly extracts tags from the standard logging payload
    """
    # Create a mock logging payload with tags
    mock_payload = create_standard_logging_payload()
    mock_payload["request_tags"] = ["tag1", "tag2", "test_tag"]

    # Test with payload containing tags
    result = global_langfuse_logger._get_langfuse_tags(mock_payload)
    assert result == ["tag1", "tag2", "test_tag"]

    # Test with payload without tags
    mock_payload["request_tags"] = None
    result = global_langfuse_logger._get_langfuse_tags(mock_payload)
    assert result == []

    # Test with empty tags list
    mock_payload["request_tags"] = []
    result = global_langfuse_logger._get_langfuse_tags(mock_payload)
    assert result == []


@patch.dict(os.environ, {}, clear=True)  # Start with empty environment
def test_get_langfuse_flush_interval():
    """
    Test that _get_langfuse_flush_interval correctly reads from environment variable
    or falls back to the provided flush_interval
    """
    default_interval = 60

    # Test when env var is not set
    result = LangFuseLogger._get_langfuse_flush_interval(
        flush_interval=default_interval
    )
    assert result == default_interval

    # Test when env var is set
    with patch.dict(os.environ, {"LANGFUSE_FLUSH_INTERVAL": "120"}):
        result = LangFuseLogger._get_langfuse_flush_interval(
            flush_interval=default_interval
        )
        assert result == 120


def test_langfuse_e2e_sync(monkeypatch):
    from litellm import completion
    import litellm
    import respx
    import httpx
    import time
    litellm.disable_aiohttp_transport = True # since this uses respx, we need to set use_aiohttp_transport to False

    litellm._turn_on_debug()
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])

    with respx.mock:
        # Mock Langfuse
        # Mock any Langfuse endpoint
        langfuse_mock = respx.post(
            "https://*.cloud.langfuse.com/api/public/ingestion"
        ).mock(return_value=httpx.Response(200))
        completion(
            model="openai/my-fake-endpoint",
            messages=[{"role": "user", "content": "hello from litellm"}],
            stream=False,
            mock_response="Hello from litellm 2",
        )

        time.sleep(3)

        assert langfuse_mock.called


def test_get_chat_content_for_langfuse():
    """
    Test that _get_chat_content_for_langfuse correctly extracts content from chat completion responses
    """
    # Test with valid response
    mock_response = ModelResponse(
        choices=[Choices(message=Message(role="assistant", content="Hello world"))]
    )

    result = LangFuseLogger._get_chat_content_for_langfuse(mock_response)
    assert result["content"] == "Hello world"
    assert result["role"] == "assistant"

    # Test with empty choices
    mock_response = ModelResponse(choices=[])
    result = LangFuseLogger._get_chat_content_for_langfuse(mock_response)
    assert result is None


def test_get_text_completion_content_for_langfuse():
    """
    Test that _get_text_completion_content_for_langfuse correctly extracts content from text completion responses
    """
    # Test with valid response
    mock_response = TextCompletionResponse(choices=[TextChoices(text="Hello world")])
    result = LangFuseLogger._get_text_completion_content_for_langfuse(mock_response)
    assert result == "Hello world"

    # Test with empty choices
    mock_response = TextCompletionResponse(choices=[])
    result = LangFuseLogger._get_text_completion_content_for_langfuse(mock_response)
    assert result is None

    # Test with no choices field
    mock_response = TextCompletionResponse()
    result = LangFuseLogger._get_text_completion_content_for_langfuse(mock_response)
    assert result is None


def test_apply_masking_function_with_string():
    """
    Test that _apply_masking_function correctly applies masking to strings
    """
    import re

    def mask_credit_cards(data):
        if isinstance(data, str):
            return re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', data)
        return data

    # Test with string containing credit card
    input_str = "My card is 4532-1234-5678-9012"
    result = LangFuseLogger._apply_masking_function(input_str, mask_credit_cards)
    assert result == "My card is [CARD]"
    assert "4532" not in result

    # Test with string without sensitive data
    input_str = "Hello world"
    result = LangFuseLogger._apply_masking_function(input_str, mask_credit_cards)
    assert result == "Hello world"


def test_apply_masking_function_with_dict():
    """
    Test that _apply_masking_function correctly applies masking to nested dicts
    """
    import re

    def mask_emails(data):
        if isinstance(data, str):
            return re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', data)
        return data

    # Test with dict containing messages
    input_dict = {
        "messages": [
            {"role": "user", "content": "My email is test@example.com"}
        ]
    }
    result = LangFuseLogger._apply_masking_function(input_dict, mask_emails)
    assert result["messages"][0]["content"] == "My email is [EMAIL]"
    assert "test@example.com" not in str(result)


def test_apply_masking_function_with_none():
    """
    Test that _apply_masking_function handles None correctly
    """
    def dummy_mask(data):
        return data

    result = LangFuseLogger._apply_masking_function(None, dummy_mask)
    assert result is None


def test_apply_masking_function_with_list():
    """
    Test that _apply_masking_function correctly applies masking to lists
    """
    import re

    def mask_ssn(data):
        if isinstance(data, str):
            return re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', data)
        return data

    input_list = ["SSN: 123-45-6789", "No sensitive data here"]
    result = LangFuseLogger._apply_masking_function(input_list, mask_ssn)
    assert result[0] == "SSN: [SSN]"
    assert result[1] == "No sensitive data here"


def test_masking_function_isolated_from_other_loggers():
    """
    Test that langfuse_masking_function is extracted from metadata and stored separately.
    This ensures the callable doesn't leak to other logging integrations.
    """
    from litellm.litellm_core_utils.litellm_logging import scrub_sensitive_keys_in_metadata

    def my_masking_fn(data):
        return data

    # Simulate litellm_params with masking function in metadata
    litellm_params = {
        "metadata": {
            "langfuse_masking_function": my_masking_fn,
            "other_key": "other_value",
        }
    }

    # Scrub should extract the function
    result = scrub_sensitive_keys_in_metadata(litellm_params)

    # Function should be removed from metadata (won't leak to other loggers)
    assert "langfuse_masking_function" not in result["metadata"]

    # Function should be stored in dedicated key for Langfuse to access
    assert result.get("_langfuse_masking_function") == my_masking_fn

    # Other metadata should remain intact
    assert result["metadata"]["other_key"] == "other_value"


def test_masking_function_not_in_metadata_when_not_provided():
    """
    Test that scrub_sensitive_keys_in_metadata works normally when no masking function is provided.
    """
    from litellm.litellm_core_utils.litellm_logging import scrub_sensitive_keys_in_metadata

    litellm_params = {
        "metadata": {
            "some_key": "some_value",
        }
    }

    result = scrub_sensitive_keys_in_metadata(litellm_params)

    # No _langfuse_masking_function should be added
    assert "_langfuse_masking_function" not in result

    # Original metadata should be unchanged
    assert result["metadata"]["some_key"] == "some_value"
