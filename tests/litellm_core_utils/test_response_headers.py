"""
Tests for vendor response header preservation in litellm.

Verifies that custom vendor headers (e.g., TRAFFIC-TYPE) are preserved
with their original names alongside the llm_provider-prefixed versions.
"""

import pytest

from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    _get_llm_provider_headers,
    get_response_headers,
)
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.llms.azure.common_utils import process_azure_headers


class TestGetLlmProviderHeaders:
    """Tests for _get_llm_provider_headers in get_headers.py"""

    def test_custom_header_preserved_with_original_name(self):
        """TRAFFIC-TYPE should be accessible by its original key."""
        headers = {"TRAFFIC-TYPE": "synthetic", "content-type": "application/json"}
        result = _get_llm_provider_headers(headers)
        assert "TRAFFIC-TYPE" in result
        assert result["TRAFFIC-TYPE"] == "synthetic"

    def test_custom_header_also_has_llm_provider_prefix(self):
        """TRAFFIC-TYPE should also exist with the llm_provider- prefix."""
        headers = {"TRAFFIC-TYPE": "synthetic"}
        result = _get_llm_provider_headers(headers)
        assert "llm_provider-TRAFFIC-TYPE" in result
        assert result["llm_provider-TRAFFIC-TYPE"] == "synthetic"

    def test_already_prefixed_header_not_double_prefixed(self):
        """Headers already prefixed with llm_provider should not be re-prefixed."""
        headers = {"llm_provider-some-header": "value"}
        result = _get_llm_provider_headers(headers)
        assert "llm_provider-some-header" in result
        assert "llm_provider-llm_provider-some-header" not in result

    def test_empty_headers_returns_empty_dict(self):
        result = _get_llm_provider_headers({})
        assert result == {}

    def test_multiple_custom_headers_all_preserved(self):
        """All vendor-specific headers should be preserved with original names."""
        headers = {
            "TRAFFIC-TYPE": "synthetic",
            "x-custom-vendor": "some-value",
            "x-request-id": "abc-123",
        }
        result = _get_llm_provider_headers(headers)
        for original_key in headers:
            assert original_key in result, f"Original header '{original_key}' missing"
            assert f"llm_provider-{original_key}" in result


class TestGetResponseHeaders:
    """Tests for get_response_headers in get_headers.py"""

    def test_none_input_returns_empty_dict(self):
        assert get_response_headers(None) == {}

    def test_traffic_type_preserved_in_final_output(self):
        """TRAFFIC-TYPE should appear by original name in get_response_headers output."""
        headers = {
            "TRAFFIC-TYPE": "live",
            "x-ratelimit-remaining-requests": "100",
        }
        result = get_response_headers(headers)
        assert "TRAFFIC-TYPE" in result
        assert result["TRAFFIC-TYPE"] == "live"
        assert "llm_provider-TRAFFIC-TYPE" in result
        assert result["x-ratelimit-remaining-requests"] == "100"

    def test_openai_ratelimit_headers_preserved_without_prefix(self):
        """Standard OpenAI rate-limit headers should keep their original name."""
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-tokens": "5000",
        }
        result = get_response_headers(headers)
        assert result["x-ratelimit-limit-requests"] == "60"
        assert result["x-ratelimit-remaining-tokens"] == "5000"


class TestProcessResponseHeaders:
    """Tests for process_response_headers in core_helpers.py (streaming path)."""

    def test_custom_header_preserved_with_original_name(self):
        """Streaming path should also preserve TRAFFIC-TYPE by original name."""
        headers = {"TRAFFIC-TYPE": "synthetic", "content-type": "application/json"}
        result = process_response_headers(headers)
        assert "TRAFFIC-TYPE" in result
        assert result["TRAFFIC-TYPE"] == "synthetic"

    def test_custom_header_also_has_llm_provider_prefix(self):
        headers = {"TRAFFIC-TYPE": "synthetic"}
        result = process_response_headers(headers)
        assert "llm_provider-TRAFFIC-TYPE" in result
        assert result["llm_provider-TRAFFIC-TYPE"] == "synthetic"

    def test_already_prefixed_header_not_double_prefixed(self):
        headers = {"llm_provider-some-header": "value"}
        result = process_response_headers(headers)
        assert "llm_provider-some-header" in result
        assert "llm_provider-llm_provider-some-header" not in result

    def test_openai_headers_preserved(self):
        headers = {"x-ratelimit-limit-requests": "60", "TRAFFIC-TYPE": "live"}
        result = process_response_headers(headers)
        assert result["x-ratelimit-limit-requests"] == "60"
        assert "TRAFFIC-TYPE" in result


class TestProcessAzureHeaders:
    """Tests for process_azure_headers in azure/common_utils.py."""

    def test_traffic_type_preserved_with_original_name(self):
        """Azure-specific header processing should preserve TRAFFIC-TYPE."""
        headers = {
            "TRAFFIC-TYPE": "synthetic",
            "x-ratelimit-remaining-requests": "50",
        }
        result = process_azure_headers(headers)
        assert "TRAFFIC-TYPE" in result
        assert result["TRAFFIC-TYPE"] == "synthetic"
        assert "llm_provider-TRAFFIC-TYPE" in result

    def test_ratelimit_headers_accessible_by_original_name(self):
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-remaining-tokens": "5000",
        }
        result = process_azure_headers(headers)
        assert result["x-ratelimit-limit-requests"] == "60"
        assert result["x-ratelimit-remaining-requests"] == "50"
        assert result["x-ratelimit-limit-tokens"] == "10000"
        assert result["x-ratelimit-remaining-tokens"] == "5000"

    def test_empty_headers(self):
        result = process_azure_headers({})
        assert result == {}

    def test_multiple_custom_headers_preserved(self):
        headers = {
            "TRAFFIC-TYPE": "live",
            "x-custom-header": "val",
            "x-ratelimit-limit-requests": "60",
        }
        result = process_azure_headers(headers)
        assert "TRAFFIC-TYPE" in result
        assert "x-custom-header" in result
        assert "llm_provider-TRAFFIC-TYPE" in result
        assert "llm_provider-x-custom-header" in result
