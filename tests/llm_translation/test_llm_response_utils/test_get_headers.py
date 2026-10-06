import json
from datetime import datetime


import litellm
import pytest

from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
    _get_llm_provider_headers,
    get_provider_request_id,
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


@pytest.mark.parametrize("header", ("request-id", "Request-Id", "x-request-id", "llm_provider-request-id"))
def test_native_clients_receive_the_provider_request_id(header: str) -> None:
    result = get_response_headers({header: "req_test", "unrelated": "value"})
    assert result["request-id"] == "req_test"
    assert get_provider_request_id(result) == "req_test"
    assert result["llm_provider-unrelated"] == "value"
    assert "unrelated" not in result


@pytest.mark.parametrize("headers", (None, {}, {"request-id": ""}, {"request-id": 42}))
def test_invalid_provider_request_ids_remain_absent(headers: object) -> None:
    assert get_provider_request_id(headers) is None
