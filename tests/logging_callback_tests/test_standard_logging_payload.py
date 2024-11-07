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

import pytest
import litellm
from litellm.types.utils import Usage, StandardLoggingMetadata
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
