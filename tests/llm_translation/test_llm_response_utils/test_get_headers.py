import json
import os
import sys
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path

import litellm
import pytest

from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
    _get_llm_provider_headers,
)


def test_get_response_headers_empty():
    result = get_response_headers()
    assert result == {}, "Expected empty dictionary for no input"


def test_get_response_headers_with_openai_headers():
    """
    OpenAI headers are forwarded as is
    Other headers are prefixed with llm_provider-
    """
    input_headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-limit-tokens": "1000",
        "x-ratelimit-remaining-tokens": "500",
        "other-header": "value",
    }
    expected_output = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-limit-tokens": "1000",
        "x-ratelimit-remaining-tokens": "500",
        "llm_provider-x-ratelimit-limit-requests": "100",
        "llm_provider-x-ratelimit-remaining-requests": "50",
        "llm_provider-x-ratelimit-limit-tokens": "1000",
        "llm_provider-x-ratelimit-remaining-tokens": "500",
        "llm_provider-other-header": "value",
    }
    result = get_response_headers(input_headers)
    assert result == expected_output, "Unexpected output for OpenAI headers"


def test_get_response_headers_without_openai_headers():
    """
    Non-OpenAI headers are prefixed with llm_provider-
    """
    input_headers = {"custom-header-1": "value1", "custom-header-2": "value2"}
    expected_output = {
        "llm_provider-custom-header-1": "value1",
        "llm_provider-custom-header-2": "value2",
    }
    result = get_response_headers(input_headers)
    assert result == expected_output, "Unexpected output for non-OpenAI headers"


def test_get_llm_provider_headers():
    """
    If non OpenAI headers are already prefixed with llm_provider- they are not prefixed with llm_provider- again
    """
    input_headers = {
        "header1": "value1",
        "header2": "value2",
        "llm_provider-existing": "existing_value",
    }
    expected_output = {
        "llm_provider-header1": "value1",
        "llm_provider-header2": "value2",
        "llm_provider-existing": "existing_value",
    }
    result = _get_llm_provider_headers(input_headers)
    assert result == expected_output, "Unexpected output for _get_llm_provider_headers"

def test_get_response_headers_with_retry_after():
    """
    When upstream provider returns retry-after header (e.g. on 429),
    it should be forwarded as both 'retry-after' and 'llm_provider-retry-after'.

    Ref: https://github.com/BerriAI/litellm/issues/21553
    """
    input_headers = {
        "retry-after": "57",
        "x-ratelimit-limit-tokens": "100000",
        "content-type": "application/json",
    }
    result = get_response_headers(input_headers)

    # retry-after should be preserved as a bare header
    assert result.get("retry-after") == "57"
    # retry-after should also be present as llm_provider-retry-after
    assert result.get("llm_provider-retry-after") == "57"
    # OpenAI headers should still work
    assert result.get("x-ratelimit-limit-tokens") == "100000"
    # Other headers should only be prefixed
    assert result.get("llm_provider-content-type") == "application/json"
    assert "content-type" not in result


def test_get_response_headers_without_retry_after():
    """
    When upstream provider does NOT return retry-after header,
    the output should not contain a retry-after key.
    """
    input_headers = {
        "x-ratelimit-limit-tokens": "100000",
        "content-type": "application/json",
    }
    result = get_response_headers(input_headers)

    assert "retry-after" not in result
    assert result.get("x-ratelimit-limit-tokens") == "100000"
