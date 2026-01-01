"""
Test Vertex AI Passthrough Anthropic Headers Forwarding

This module tests the forwarding of Anthropic-specific headers
(anthropic-version, anthropic-beta) in Vertex AI passthrough requests.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.types.llms.anthropic import ANTHROPIC_API_HEADERS


class TestAnthropicApiHeadersConstant:
    """Test the ANTHROPIC_API_HEADERS constant"""

    def test_anthropic_api_headers_contains_expected_headers(self):
        """Test that ANTHROPIC_API_HEADERS contains the expected headers"""
        assert "anthropic-version" in ANTHROPIC_API_HEADERS
        assert "anthropic-beta" in ANTHROPIC_API_HEADERS

    def test_anthropic_api_headers_is_set(self):
        """Test that ANTHROPIC_API_HEADERS is a set type"""
        assert isinstance(ANTHROPIC_API_HEADERS, set)


class TestAnthropicHeadersForwardingLogic:
    """Test the header forwarding logic without importing the full module"""

    def test_forward_anthropic_headers_when_present(self):
        """
        Test that the header forwarding logic correctly adds Anthropic headers
        to the output headers dictionary when they are present in the request.
        """
        # Simulate request headers
        request_headers = {
            "authorization": "Bearer test-token",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15",
        }

        # Simulate the headers dict that would be built
        headers = {
            "Authorization": "Bearer test-auth-header",
        }

        # Apply the same logic as in _prepare_vertex_auth_headers
        for header_name in ANTHROPIC_API_HEADERS:
            if header_name in request_headers:
                headers[header_name] = request_headers[header_name]

        # Verify Anthropic headers are in the result
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == "2023-06-01"
        assert "anthropic-beta" in headers
        assert headers["anthropic-beta"] == "max-tokens-3-5-sonnet-2024-07-15"
        # Also verify Authorization header is still present
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-auth-header"

    def test_no_anthropic_headers_when_not_present(self):
        """
        Test that no Anthropic headers are added when they are not in the request.
        """
        # Simulate request headers without Anthropic headers
        request_headers = {
            "authorization": "Bearer test-token",
            "content-type": "application/json",
        }

        # Simulate the headers dict that would be built
        headers = {
            "Authorization": "Bearer test-auth-header",
        }

        # Apply the same logic as in _prepare_vertex_auth_headers
        for header_name in ANTHROPIC_API_HEADERS:
            if header_name in request_headers:
                headers[header_name] = request_headers[header_name]

        # Verify Anthropic headers are NOT in the result
        assert "anthropic-version" not in headers
        assert "anthropic-beta" not in headers
        # Authorization header should still be present
        assert "Authorization" in headers

    def test_partial_anthropic_headers_forwarded(self):
        """
        Test that only present Anthropic headers are forwarded.
        """
        # Simulate request headers with only anthropic-version
        request_headers = {
            "authorization": "Bearer test-token",
            "anthropic-version": "2023-06-01",
        }

        # Simulate the headers dict that would be built
        headers = {
            "Authorization": "Bearer test-auth-header",
        }

        # Apply the same logic as in _prepare_vertex_auth_headers
        for header_name in ANTHROPIC_API_HEADERS:
            if header_name in request_headers:
                headers[header_name] = request_headers[header_name]

        # Verify only anthropic-version is in the result
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == "2023-06-01"
        assert "anthropic-beta" not in headers

    def test_header_values_preserved_exactly(self):
        """
        Test that header values are preserved exactly as provided.
        """
        # Test with complex beta header value
        request_headers = {
            "anthropic-beta": "extended-thinking-2025-01-24,output-128k-2025-01-24",
        }

        headers = {}

        for header_name in ANTHROPIC_API_HEADERS:
            if header_name in request_headers:
                headers[header_name] = request_headers[header_name]

        assert headers["anthropic-beta"] == "extended-thinking-2025-01-24,output-128k-2025-01-24"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
