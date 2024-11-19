"""
Unit tests for StandardLoggingPayloadSetup
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

from pydantic.main import Model

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from datetime import datetime as dt_object
import time
import pytest
import litellm
from litellm.types.utils import (
    Usage,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingHiddenParams,
)
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup


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
