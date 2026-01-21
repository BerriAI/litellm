"""
Tests for exception header preservation.

These tests verify that when LLM providers return error responses with headers,
those headers are preserved in the exception and can be returned to clients.

This is important for debugging and observability - headers like x-request-id,
x-ms-region, rate limit headers, etc. should be available even when errors occur.
"""

import httpx
import pytest

from litellm.exceptions import (
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    ImageFetchError,
)


class TestExceptionHeaderPreservation:
    """Test that exception classes preserve headers from provider responses."""

    @pytest.fixture
    def mock_response_with_headers(self) -> httpx.Response:
        """Create a mock response with typical provider headers."""
        return httpx.Response(
            status_code=400,
            headers={
                "x-request-id": "req-abc123",
                "x-ms-region": "eastus",
                "x-ratelimit-remaining-requests": "99",
                "x-ratelimit-remaining-tokens": "9999",
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

    def test_bad_request_error_preserves_headers(
        self, mock_response_with_headers: httpx.Response
    ):
        """BadRequestError should preserve headers from the provider response."""
        error = BadRequestError(
            message="Invalid request",
            model="gpt-4",
            llm_provider="azure",
            response=mock_response_with_headers,
        )

        assert error.response is not None
        assert error.response.headers.get("x-request-id") == "req-abc123"
        assert error.response.headers.get("x-ms-region") == "eastus"
        assert error.response.headers.get("x-ratelimit-remaining-requests") == "99"

    def test_content_policy_violation_error_preserves_headers(
        self, mock_response_with_headers: httpx.Response
    ):
        """ContentPolicyViolationError should preserve headers from the provider response."""
        error = ContentPolicyViolationError(
            message="Content policy violation",
            model="gpt-4",
            llm_provider="azure",
            response=mock_response_with_headers,
        )

        assert error.response is not None
        assert error.response.headers.get("x-request-id") == "req-abc123"
        assert error.response.headers.get("x-ms-region") == "eastus"

    def test_context_window_exceeded_error_preserves_headers(
        self, mock_response_with_headers: httpx.Response
    ):
        """ContextWindowExceededError should preserve headers from the provider response."""
        error = ContextWindowExceededError(
            message="Context window exceeded",
            model="gpt-4",
            llm_provider="azure",
            response=mock_response_with_headers,
        )

        assert error.response is not None
        assert error.response.headers.get("x-request-id") == "req-abc123"
        assert error.response.headers.get("x-ms-region") == "eastus"

    def test_image_fetch_error_preserves_headers(
        self, mock_response_with_headers: httpx.Response
    ):
        """ImageFetchError should preserve headers from the provider response."""
        error = ImageFetchError(
            message="Failed to fetch image",
            model="gpt-4",
            llm_provider="azure",
            response=mock_response_with_headers,
        )

        assert error.response is not None
        assert error.response.headers.get("x-request-id") == "req-abc123"
        assert error.response.headers.get("x-ms-region") == "eastus"

    def test_bad_request_error_handles_none_response(self):
        """BadRequestError should handle None response gracefully."""
        error = BadRequestError(
            message="Invalid request",
            model="gpt-4",
            llm_provider="azure",
            response=None,
        )

        assert error.response is not None
        # Headers should be empty but not cause an error
        assert error.response.headers.get("x-request-id") is None

    def test_content_policy_violation_error_handles_none_response(self):
        """ContentPolicyViolationError should handle None response gracefully."""
        error = ContentPolicyViolationError(
            message="Content policy violation",
            model="gpt-4",
            llm_provider="azure",
            response=None,
        )

        assert error.response is not None
        assert error.response.headers.get("x-request-id") is None

    def test_context_window_exceeded_error_handles_none_response(self):
        """ContextWindowExceededError should handle None response gracefully."""
        error = ContextWindowExceededError(
            message="Context window exceeded",
            model="gpt-4",
            llm_provider="azure",
            response=None,
        )

        assert error.response is not None
        assert error.response.headers.get("x-request-id") is None


class TestExceptionMessageFormatting:
    """Test that exception messages are formatted correctly after refactoring."""

    def test_bad_request_error_message_format(self):
        """BadRequestError should format message with litellm prefix."""
        error = BadRequestError(
            message="test error",
            model="gpt-4",
            llm_provider="azure",
        )

        assert "litellm.BadRequestError" in error.message
        assert "test error" in error.message

    def test_content_policy_violation_error_message_format(self):
        """ContentPolicyViolationError should format message with specific prefix."""
        error = ContentPolicyViolationError(
            message="test error",
            model="gpt-4",
            llm_provider="azure",
        )

        assert "litellm.ContentPolicyViolationError" in error.message
        assert "test error" in error.message

    def test_context_window_exceeded_error_message_format(self):
        """ContextWindowExceededError should format message with specific prefix."""
        error = ContextWindowExceededError(
            message="test error",
            model="gpt-4",
            llm_provider="azure",
        )

        assert "litellm.ContextWindowExceededError" in error.message
        assert "test error" in error.message


class TestExceptionAttributes:
    """Test that exception attributes are set correctly."""

    def test_content_policy_violation_error_provider_specific_fields(self):
        """ContentPolicyViolationError should preserve provider_specific_fields."""
        provider_fields = {"innererror": {"code": "ResponsibleAIPolicyViolation"}}

        error = ContentPolicyViolationError(
            message="test error",
            model="gpt-4",
            llm_provider="azure",
            provider_specific_fields=provider_fields,
        )

        assert error.provider_specific_fields == provider_fields
        assert (
            error.provider_specific_fields["innererror"]["code"]
            == "ResponsibleAIPolicyViolation"
        )

    def test_bad_request_error_attributes(self):
        """BadRequestError should set all expected attributes."""
        error = BadRequestError(
            message="test error",
            model="gpt-4",
            llm_provider="azure",
            litellm_debug_info="debug info",
            max_retries=3,
            num_retries=1,
        )

        assert error.model == "gpt-4"
        assert error.llm_provider == "azure"
        assert error.litellm_debug_info == "debug info"
        assert error.max_retries == 3
        assert error.num_retries == 1
        assert error.status_code == 400


class TestProxyHeaderExtraction:
    """Test that proxy correctly extracts headers from exceptions."""

    def test_get_response_headers_adds_llm_provider_prefix(self):
        """get_response_headers should prefix non-OpenAI headers with llm_provider-."""
        from litellm.litellm_core_utils.llm_response_utils.get_headers import (
            get_response_headers,
        )

        response_headers = {
            "x-request-id": "req-abc123",
            "x-ms-region": "eastus",
            "x-ratelimit-remaining-requests": "99",  # OpenAI header - should not be prefixed
        }

        result = get_response_headers(response_headers)

        # OpenAI ratelimit headers should be preserved as-is
        assert result.get("x-ratelimit-remaining-requests") == "99"
        # Other headers should be prefixed with llm_provider-
        assert result.get("llm_provider-x-request-id") == "req-abc123"
        assert result.get("llm_provider-x-ms-region") == "eastus"

    def test_proxy_can_extract_headers_from_exception_response(self):
        """Simulate how proxy extracts headers from exception.response.headers."""
        from litellm.litellm_core_utils.llm_response_utils.get_headers import (
            get_response_headers,
        )

        # Create exception with headers in response
        mock_response = httpx.Response(
            status_code=400,
            headers={
                "x-request-id": "req-abc123",
                "x-ms-region": "eastus",
            },
            request=httpx.Request("POST", "https://test.com"),
        )
        error = ContentPolicyViolationError(
            message="test",
            model="gpt-4",
            llm_provider="azure",
            response=mock_response,
        )

        # Simulate proxy header extraction logic
        headers = getattr(error, "headers", None) or {}
        if not headers:
            _response = getattr(error, "response", None)
            if _response is not None:
                _response_headers = getattr(_response, "headers", None)
                if _response_headers:
                    headers = get_response_headers(dict(_response_headers))

        # Verify headers are extracted and prefixed correctly
        assert headers.get("llm_provider-x-request-id") == "req-abc123"
        assert headers.get("llm_provider-x-ms-region") == "eastus"
