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
from litellm.types.utils import Usage
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
