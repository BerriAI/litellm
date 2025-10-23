"""
Unit tests for StandardLoggingPayloadSetup
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from datetime import datetime as dt_object
import time
import pytest
import litellm
from litellm.types.utils import (
    StandardLoggingPayload,
    Usage,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingHiddenParams,
)
from create_mock_standard_logging_payload import (
    create_standard_logging_payload,
    create_standard_logging_payload_with_long_content,
)
from litellm.litellm_core_utils.litellm_logging import (
    StandardLoggingPayloadSetup,
)

from litellm.integrations.custom_logger import CustomLogger


@pytest.mark.parametrize(
    "response_obj,expected_values",
    [
        # Test None input
        (None, (0, 0, 0)),
        # Test empty dict
        ({}, (0, 0, 0)),
        # Test valid usage dict
        (
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            },
            (10, 20, 30),
        ),
        # Test with litellm.Usage object
        (
            {"usage": Usage(prompt_tokens=15, completion_tokens=25, total_tokens=40)},
            (15, 25, 40),
        ),
        # Test invalid usage type
        ({"usage": "invalid"}, (0, 0, 0)),
        # Test None usage
        ({"usage": None}, (0, 0, 0)),
    ],
)
def test_get_usage(response_obj, expected_values):
    """
    Make sure values returned from get_usage are always integers
    """

    usage = StandardLoggingPayloadSetup.get_usage_from_response_obj(response_obj)

    # Check types
    assert isinstance(usage.prompt_tokens, int)
    assert isinstance(usage.completion_tokens, int)
    assert isinstance(usage.total_tokens, int)

    # Check values
    assert usage.prompt_tokens == expected_values[0]
    assert usage.completion_tokens == expected_values[1]
    assert usage.total_tokens == expected_values[2]


def test_get_additional_headers():
    additional_headers = {
        "x-ratelimit-limit-requests": "2000",
        "x-ratelimit-remaining-requests": "1999",
        "x-ratelimit-limit-tokens": "160000",
        "x-ratelimit-remaining-tokens": "160000",
        "llm_provider-date": "Tue, 29 Oct 2024 23:57:37 GMT",
        "llm_provider-content-type": "application/json",
        "llm_provider-transfer-encoding": "chunked",
        "llm_provider-connection": "keep-alive",
        "llm_provider-anthropic-ratelimit-requests-limit": "2000",
        "llm_provider-anthropic-ratelimit-requests-remaining": "1999",
        "llm_provider-anthropic-ratelimit-requests-reset": "2024-10-29T23:57:40Z",
        "llm_provider-anthropic-ratelimit-tokens-limit": "160000",
        "llm_provider-anthropic-ratelimit-tokens-remaining": "160000",
        "llm_provider-anthropic-ratelimit-tokens-reset": "2024-10-29T23:57:36Z",
        "llm_provider-request-id": "req_01F6CycZZPSHKRCCctcS1Vto",
        "llm_provider-via": "1.1 google",
        "llm_provider-cf-cache-status": "DYNAMIC",
        "llm_provider-x-robots-tag": "none",
        "llm_provider-server": "cloudflare",
        "llm_provider-cf-ray": "8da71bdbc9b57abb-SJC",
        "llm_provider-content-encoding": "gzip",
        "llm_provider-x-ratelimit-limit-requests": "2000",
        "llm_provider-x-ratelimit-remaining-requests": "1999",
        "llm_provider-x-ratelimit-limit-tokens": "160000",
        "llm_provider-x-ratelimit-remaining-tokens": "160000",
    }
    additional_logging_headers = StandardLoggingPayloadSetup.get_additional_headers(
        additional_headers
    )
    assert additional_logging_headers == {
        "x_ratelimit_limit_requests": 2000,
        "x_ratelimit_remaining_requests": 1999,
        "x_ratelimit_limit_tokens": 160000,
        "x_ratelimit_remaining_tokens": 160000,
    }


def all_fields_present(standard_logging_metadata: StandardLoggingMetadata):
    for field in StandardLoggingMetadata.__annotations__.keys():
        assert field in standard_logging_metadata


@pytest.mark.parametrize(
    "metadata_key, metadata_value",
    [
        ("user_api_key_alias", "test_alias"),
        ("user_api_key_hash", "test_hash"),
        ("user_api_key_team_id", "test_team_id"),
        ("user_api_key_user_id", "test_user_id"),
        ("user_api_key_team_alias", "test_team_alias"),
        ("user_api_key_spend", 10.50),
        ("spend_logs_metadata", {"key": "value"}),
        ("requester_ip_address", "127.0.0.1"),
        ("requester_metadata", {"user_agent": "test_agent"}),
    ],
)
def test_get_standard_logging_metadata(metadata_key, metadata_value):
    """
    Test that the get_standard_logging_metadata function correctly sets the metadata fields.
    All fields in StandardLoggingMetadata should ALWAYS be present.
    """
    metadata = {metadata_key: metadata_value}
    standard_logging_metadata = (
        StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)
    )

    print("standard_logging_metadata", standard_logging_metadata)

    # Assert that all fields in StandardLoggingMetadata are present
    all_fields_present(standard_logging_metadata)

    # Assert that the specific metadata field is set correctly
    assert standard_logging_metadata[metadata_key] == metadata_value


def test_get_standard_logging_metadata_user_api_key_hash():
    valid_hash = "a" * 64  # 64 character string
    metadata = {"user_api_key": valid_hash}
    result = StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)
    assert result["user_api_key_hash"] == valid_hash


def test_get_standard_logging_metadata_invalid_user_api_key():
    invalid_hash = "not_a_valid_hash"
    metadata = {"user_api_key": invalid_hash}
    result = StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)
    all_fields_present(result)
    assert result["user_api_key_hash"] is None


def test_get_standard_logging_metadata_invalid_keys():
    metadata = {
        "user_api_key_alias": "test_alias",
        "invalid_key": "should_be_ignored",
        "another_invalid_key": 123,
    }
    result = StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)
    all_fields_present(result)
    assert result["user_api_key_alias"] == "test_alias"
    assert "invalid_key" not in result
    assert "another_invalid_key" not in result


def test_cleanup_timestamps():
    """Test cleanup_timestamps with different input types"""
    # Test with datetime objects
    now = dt_object.now()
    start = now
    end = now
    completion = now

    result = StandardLoggingPayloadSetup.cleanup_timestamps(start, end, completion)

    assert all(isinstance(x, float) for x in result)
    assert len(result) == 3

    # Test with float timestamps
    start_float = time.time()
    end_float = start_float + 1
    completion_float = end_float

    result = StandardLoggingPayloadSetup.cleanup_timestamps(
        start_float, end_float, completion_float
    )

    assert all(isinstance(x, float) for x in result)
    assert result[0] == start_float
    assert result[1] == end_float
    assert result[2] == completion_float

    # Test with mixed types
    result = StandardLoggingPayloadSetup.cleanup_timestamps(
        start_float, end, completion_float
    )
    assert all(isinstance(x, float) for x in result)

    # Test invalid input
    with pytest.raises(ValueError):
        StandardLoggingPayloadSetup.cleanup_timestamps(
            "invalid", end_float, completion_float
        )


def test_get_model_cost_information():
    """Test get_model_cost_information with different inputs"""
    # Test with None values
    result = StandardLoggingPayloadSetup.get_model_cost_information(
        base_model=None,
        custom_pricing=None,
        custom_llm_provider=None,
        init_response_obj={},
    )
    assert result["model_map_key"] == ""
    assert result["model_map_value"] is None  # this was not found in model cost map
    # assert all fields in StandardLoggingModelInformation are present
    assert all(
        field in result for field in StandardLoggingModelInformation.__annotations__
    )

    # Test with valid model
    result = StandardLoggingPayloadSetup.get_model_cost_information(
        base_model="gpt-3.5-turbo",
        custom_pricing=False,
        custom_llm_provider="openai",
        init_response_obj={},
    )
    litellm_info_gpt_3_5_turbo_model_map_value = litellm.get_model_info(
        model="gpt-3.5-turbo", custom_llm_provider="openai"
    )
    print("result", result)
    assert result["model_map_key"] == "gpt-3.5-turbo"
    assert result["model_map_value"] is not None
    assert result["model_map_value"] == litellm_info_gpt_3_5_turbo_model_map_value
    # assert all fields in StandardLoggingModelInformation are present
    assert all(
        field in result for field in StandardLoggingModelInformation.__annotations__
    )


def test_get_hidden_params():
    """Test get_hidden_params with different inputs"""
    # Test with None
    result = StandardLoggingPayloadSetup.get_hidden_params(None)
    assert result["model_id"] is None
    assert result["cache_key"] is None
    assert result["api_base"] is None
    assert result["response_cost"] is None
    assert result["additional_headers"] is None

    # assert all fields in StandardLoggingHiddenParams are present
    assert all(field in result for field in StandardLoggingHiddenParams.__annotations__)

    # Test with valid params
    hidden_params = {
        "model_id": "test-model",
        "cache_key": "test-cache",
        "api_base": "https://api.test.com",
        "response_cost": 0.001,
        "additional_headers": {
            "x-ratelimit-limit-requests": "2000",
            "x-ratelimit-remaining-requests": "1999",
        },
    }
    result = StandardLoggingPayloadSetup.get_hidden_params(hidden_params)
    assert result["model_id"] == "test-model"
    assert result["cache_key"] == "test-cache"
    assert result["api_base"] == "https://api.test.com"
    assert result["response_cost"] == 0.001
    assert result["additional_headers"] is not None
    assert result["additional_headers"]["x_ratelimit_limit_requests"] == 2000
    # assert all fields in StandardLoggingHiddenParams are present
    assert all(field in result for field in StandardLoggingHiddenParams.__annotations__)


def test_get_final_response_obj():
    """Test get_final_response_obj with different input types and redaction scenarios"""
    # Test with direct response_obj
    response_obj = {"choices": [{"message": {"content": "test content"}}]}
    result = StandardLoggingPayloadSetup.get_final_response_obj(
        response_obj=response_obj, init_response_obj=None, kwargs={}
    )
    assert result == response_obj

    # Test redaction when litellm.turn_off_message_logging is True
    litellm.turn_off_message_logging = True
    try:
        model_response = litellm.ModelResponse(
            choices=[
                litellm.Choices(message=litellm.Message(content="sensitive content"))
            ]
        )
        kwargs = {"messages": [{"role": "user", "content": "original message"}]}
        result = StandardLoggingPayloadSetup.get_final_response_obj(
            response_obj=model_response, init_response_obj=model_response, kwargs=kwargs
        )

        print("result", result)
        print("type(result)", type(result))
        # Verify response message content was redacted
        assert result["choices"][0]["message"]["content"] == "redacted-by-litellm"
        # Verify that redaction occurred in kwargs
        assert kwargs["messages"][0]["content"] == "redacted-by-litellm"
    finally:
        # Reset litellm.turn_off_message_logging to its original value
        litellm.turn_off_message_logging = False


def test_get_standard_logging_payload_trace_id():
    """Test _get_standard_logging_payload_trace_id with different input scenarios"""
    # Test case 1: When litellm_trace_id is provided in litellm_params
    from unittest.mock import MagicMock
    
    # Create a mock Logging object
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_trace_id = "default-trace-id"
    
    # Test when litellm_trace_id is in litellm_params
    litellm_params = {"litellm_trace_id": "dynamic-trace-id"}
    result = StandardLoggingPayloadSetup._get_standard_logging_payload_trace_id(
        logging_obj=mock_logging_obj,
        litellm_params=litellm_params
    )
    assert result == "dynamic-trace-id"
    
    # Test case 2: When litellm_trace_id is not provided in litellm_params
    litellm_params = {}
    result = StandardLoggingPayloadSetup._get_standard_logging_payload_trace_id(
        logging_obj=mock_logging_obj,
        litellm_params=litellm_params
    )
    assert result == "default-trace-id"
    
    # Test case 3: When litellm_params is None
    result = StandardLoggingPayloadSetup._get_standard_logging_payload_trace_id(
        logging_obj=mock_logging_obj,
        litellm_params={}
    )
    assert result == "default-trace-id"
    
    # Test case 4: When litellm_trace_id in params is not a string
    litellm_params = {"litellm_trace_id": 12345}
    result = StandardLoggingPayloadSetup._get_standard_logging_payload_trace_id(
        logging_obj=mock_logging_obj,
        litellm_params=litellm_params
    )
    assert result == "12345"
    assert isinstance(result, str)


def test_truncate_standard_logging_payload():
    """
    1. original messages, response, and error_str should NOT BE MODIFIED, since these are from kwargs
    2. the `messages`, `response`, and `error_str` in new standard_logging_payload should be truncated
    """
    _custom_logger = CustomLogger()
    standard_logging_payload: StandardLoggingPayload = (
        create_standard_logging_payload_with_long_content()
    )
    original_messages = standard_logging_payload["messages"]
    len_original_messages = len(str(original_messages))
    original_response = standard_logging_payload["response"]
    len_original_response = len(str(original_response))
    original_error_str = standard_logging_payload["error_str"]
    len_original_error_str = len(str(original_error_str))

    _custom_logger.truncate_standard_logging_payload_content(standard_logging_payload)

    # Original messages, response, and error_str should NOT BE MODIFIED
    assert standard_logging_payload["messages"] != original_messages
    assert standard_logging_payload["response"] != original_response
    assert standard_logging_payload["error_str"] != original_error_str
    assert len_original_messages == len(str(original_messages))
    assert len_original_response == len(str(original_response))
    assert len_original_error_str == len(str(original_error_str))

    print(
        "logged standard_logging_payload",
        json.dumps(standard_logging_payload, indent=2),
    )

    # Logged messages, response, and error_str should be truncated
    # assert len of messages is less than 10_500
    assert len(str(standard_logging_payload["messages"])) < 10_500
    # assert len of response is less than 10_500
    assert len(str(standard_logging_payload["response"])) < 10_500
    # assert len of error_str is less than 10_500
    assert len(str(standard_logging_payload["error_str"])) < 10_500


def test_strip_trailing_slash():
    common_api_base = "https://api.test.com"
    assert (
        StandardLoggingPayloadSetup.strip_trailing_slash(common_api_base + "/")
        == common_api_base
    )
    assert (
        StandardLoggingPayloadSetup.strip_trailing_slash(common_api_base)
        == common_api_base
    )


def test_get_error_information():
    """Test get_error_information with different types of exceptions"""

    # Test with None
    result = StandardLoggingPayloadSetup.get_error_information(None)
    print("error_information", json.dumps(result, indent=2))
    assert result["error_code"] == ""
    assert result["error_class"] == ""
    assert result["llm_provider"] == ""

    # Test with a basic Exception
    basic_exception = Exception("Test error")
    result = StandardLoggingPayloadSetup.get_error_information(basic_exception)
    print("error_information", json.dumps(result, indent=2))
    assert result["error_code"] == ""
    assert result["error_class"] == "Exception"
    assert result["llm_provider"] == ""

    # Test with litellm exception from provider
    litellm_exception = litellm.exceptions.RateLimitError(
        message="Test error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
        response=None,
        litellm_debug_info=None,
        max_retries=None,
        num_retries=None,
    )
    result = StandardLoggingPayloadSetup.get_error_information(litellm_exception)
    print("error_information", json.dumps(result, indent=2))
    assert result["error_code"] == "429"
    assert result["error_class"] == "RateLimitError"
    assert result["llm_provider"] == "openai"
    assert result["error_message"] == "litellm.RateLimitError: Test error"


def test_get_response_time():
    """Test get_response_time with different streaming scenarios"""
    # Test case 1: Non-streaming response
    start_time = 1000.0
    end_time = 1005.0
    completion_start_time = 1003.0
    stream = False

    response_time = StandardLoggingPayloadSetup.get_response_time(
        start_time_float=start_time,
        end_time_float=end_time,
        completion_start_time_float=completion_start_time,
        stream=stream,
    )

    # For non-streaming, should return end_time - start_time
    assert response_time == 5.0

    # Test case 2: Streaming response
    start_time = 1000.0
    end_time = 1010.0
    completion_start_time = 1002.0
    stream = True

    response_time = StandardLoggingPayloadSetup.get_response_time(
        start_time_float=start_time,
        end_time_float=end_time,
        completion_start_time_float=completion_start_time,
        stream=stream,
    )

    # For streaming, should return completion_start_time - start_time
    assert response_time == 2.0


@pytest.mark.parametrize(
    "metadata, expected_requester_metadata",
    [
        ({"metadata": {"test": "test2"}}, {"test": "test2"}),
        ({"metadata": {"test": "test2"}, "model_id": "test-model"}, {"test": "test2"}),
        (
            {
                "metadata": {
                    "test": "test2",
                },
                "model_id": "test-model",
                "requester_metadata": {"test": "test2"},
            },
            {"test": "test2"},
        ),
    ],
)
def test_standard_logging_metadata_requester_metadata(
    metadata, expected_requester_metadata
):
    result = StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)
    assert result["requester_metadata"] == expected_requester_metadata


def test_cost_breakdown_in_standard_logging_payload():
    """
    Test that cost breakdown fields are properly included in StandardLoggingPayload.
    Tests input_cost, output_cost, tool_usage_cost, and total_cost fields.
    """
    from litellm.litellm_core_utils.litellm_logging import get_standard_logging_object_payload, Logging
    from litellm.types.utils import Usage
    from datetime import datetime
    import time
    
    # Create a mock logging object with cost breakdown
    logging_obj = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="test-123",
        function_id="test-function"
    )
    
    # Simulate cost breakdown being stored during cost calculation
    logging_obj.set_cost_breakdown(
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.0035,
        cost_for_built_in_tools_cost_usd_dollar=0.0005
    )
    
    # Mock response object
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ]
    }
    
    # Create kwargs
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "response_cost": 0.0035,
        "custom_llm_provider": "openai",
    }
    
    start_time = datetime.now()
    end_time = datetime.now()
    
    # Get the standard logging payload
    payload = get_standard_logging_object_payload(
        kwargs=kwargs,
        init_response_obj=mock_response,
        start_time=start_time,
        end_time=end_time,
        logging_obj=logging_obj,
        status="success"
    )
    
    # Verify the cost breakdown field is present
    assert payload is not None
    assert payload["cost_breakdown"] is not None
    assert payload["cost_breakdown"]["input_cost"] == 0.001
    assert payload["cost_breakdown"]["output_cost"] == 0.002
    assert payload["cost_breakdown"]["tool_usage_cost"] == 0.0005
    assert payload["cost_breakdown"]["total_cost"] == 0.0035
    assert payload["response_cost"] == 0.0035
    
    print("✅ Cost breakdown test passed!")


def test_cost_breakdown_missing_in_standard_logging_payload():
    """
    Test that cost breakdown field is None when not available (e.g., for embedding calls)
    """
    from litellm.litellm_core_utils.litellm_logging import get_standard_logging_object_payload, Logging
    from datetime import datetime
    
    # Create a mock logging object without cost breakdown
    logging_obj = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="embedding",  # Non-completion call type
        start_time=datetime.now(),
        litellm_call_id="test-123",
        function_id="test-function"
    )
    
    # No cost breakdown stored
    
    # Mock response object
    mock_response = {
        "object": "list",
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 10, "total_tokens": 10}
    }
    
    kwargs = {
        "model": "text-embedding-ada-002",
        "input": ["Hello"],
        "response_cost": 0.0001,
        "custom_llm_provider": "openai",
    }
    
    start_time = datetime.now()
    end_time = datetime.now()
    
    # Get the standard logging payload
    payload = get_standard_logging_object_payload(
        kwargs=kwargs,
        init_response_obj=mock_response,
        start_time=start_time,
        end_time=end_time,
        logging_obj=logging_obj,
        status="success"
    )
    
    # Verify the cost breakdown field is None for non-completion calls
    assert payload is not None
    assert payload["cost_breakdown"] is None
    assert payload["response_cost"] == 0.0001
    
    print("✅ Cost breakdown missing test passed!")
