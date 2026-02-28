"""
Unit tests for get_additional_headers() in StandardLoggingPayloadSetup.

Verifies that all provider response headers are preserved in the
StandardLoggingPayload, while maintaining backward compatibility for
the 4 standard rate-limit headers (underscore aliases with int conversion).
"""

import pytest

from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup


def test_get_additional_headers_preserves_provider_headers():
    """Verify that llm_provider-* headers are not dropped."""
    headers = {
        "x-ratelimit-limit-requests": "100",
        "llm_provider-request-id": "req_abc123",
        "llm_provider-cf-ray": "8da71bdbc9b57abb-SJC",
        "llm_provider-openai-organization": "org-test",
        "llm_provider-openai-processing-ms": "250",
    }
    result = StandardLoggingPayloadSetup.get_additional_headers(headers)

    assert result is not None
    assert result["llm_provider-request-id"] == "req_abc123"
    assert result["llm_provider-cf-ray"] == "8da71bdbc9b57abb-SJC"
    assert result["llm_provider-openai-organization"] == "org-test"
    assert result["llm_provider-openai-processing-ms"] == "250"


def test_get_additional_headers_rate_limit_backward_compat():
    """Verify underscore aliases and int conversion still work for rate-limit headers."""
    headers = {
        "x-ratelimit-limit-requests": "2000",
        "x-ratelimit-remaining-requests": "1999",
        "x-ratelimit-limit-tokens": "160000",
        "x-ratelimit-remaining-tokens": "159000",
    }
    result = StandardLoggingPayloadSetup.get_additional_headers(headers)

    assert result is not None
    # Underscore aliases with int values
    assert result["x_ratelimit_limit_requests"] == 2000
    assert result["x_ratelimit_remaining_requests"] == 1999
    assert result["x_ratelimit_limit_tokens"] == 160000
    assert result["x_ratelimit_remaining_tokens"] == 159000

    # Original dashed keys also present (as strings)
    assert result["x-ratelimit-limit-requests"] == "2000"
    assert result["x-ratelimit-remaining-requests"] == "1999"


def test_get_additional_headers_none_input():
    """Returns None when input is None."""
    result = StandardLoggingPayloadSetup.get_additional_headers(None)
    assert result is None


def test_get_additional_headers_empty_input():
    """Returns empty dict when input is empty dict."""
    result = StandardLoggingPayloadSetup.get_additional_headers({})
    assert result == {}


def test_get_additional_headers_non_integer_values():
    """Non-integer rate-limit values are stored as-is instead of being dropped."""
    headers = {
        "x-ratelimit-limit-requests": "not-a-number",
        "x-ratelimit-remaining-tokens": "unlimited",
    }
    result = StandardLoggingPayloadSetup.get_additional_headers(headers)

    assert result is not None
    # Non-integer values stored as-is under underscore keys (not dropped)
    assert result["x_ratelimit_limit_requests"] == "not-a-number"
    assert result["x_ratelimit_remaining_tokens"] == "unlimited"
