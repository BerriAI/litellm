"""Test that _handle_error maps exceptions to typed exceptions when model/provider are known.

Covers fix for https://github.com/BerriAI/litellm/issues/20507:
anthropic_messages pass-through was raising BaseLLMException instead of typed
exceptions (RateLimitError, ContextWindowExceededError, etc.), breaking Router
retry/fallback logic.
"""

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler


class TestHandleErrorTypedExceptions:
    """Verify _handle_error produces typed exceptions when model+provider are supplied."""

    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()

    def test_rate_limit_error_typed(self):
        """429 with model+provider should raise RateLimitError, not BaseLLMException."""
        mock_response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            text="rate_limit_error: too many requests",
        )
        exc = httpx.HTTPStatusError(
            message="429 Too Many Requests",
            request=mock_response.request,
            response=mock_response,
        )

        with pytest.raises(litellm.RateLimitError):
            self.handler._handle_error(
                e=exc,
                provider_config=None,
                model="claude-sonnet-4-6",
                custom_llm_provider="anthropic",
            )

    def test_context_window_error_typed(self):
        """400 with context window message should raise ContextWindowExceededError."""
        mock_response = httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            text='{"error": {"type": "invalid_request_error", "message": "prompt is too long: 200000 tokens > 100000 maximum"}}',
        )
        exc = httpx.HTTPStatusError(
            message="400 Bad Request",
            request=mock_response.request,
            response=mock_response,
        )

        with pytest.raises(litellm.ContextWindowExceededError):
            self.handler._handle_error(
                e=exc,
                provider_config=None,
                model="claude-sonnet-4-6",
                custom_llm_provider="anthropic",
            )

    def test_auth_error_typed(self):
        """401 should raise AuthenticationError."""
        mock_response = httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            text='{"error": {"type": "authentication_error", "message": "invalid x-api-key"}}',
        )
        exc = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=mock_response.request,
            response=mock_response,
        )

        with pytest.raises(litellm.AuthenticationError):
            self.handler._handle_error(
                e=exc,
                provider_config=None,
                model="claude-sonnet-4-6",
                custom_llm_provider="anthropic",
            )

    def test_server_error_typed(self):
        """500 should raise a litellm exception, not BaseLLMException."""
        mock_response = httpx.Response(
            status_code=500,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            text="Internal Server Error",
        )
        exc = httpx.HTTPStatusError(
            message="500 Internal Server Error",
            request=mock_response.request,
            response=mock_response,
        )

        with pytest.raises(litellm.InternalServerError):
            self.handler._handle_error(
                e=exc,
                provider_config=None,
                model="claude-sonnet-4-6",
                custom_llm_provider="anthropic",
            )

    def test_without_model_raises_base_exception(self):
        """Without model/provider, should still raise BaseLLMException (backward compat)."""
        mock_response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            text="rate limit exceeded",
        )
        exc = httpx.HTTPStatusError(
            message="429 Too Many Requests",
            request=mock_response.request,
            response=mock_response,
        )

        with pytest.raises(BaseLLMException):
            self.handler._handle_error(
                e=exc,
                provider_config=None,
            )

    def test_already_typed_exception_passes_through(self):
        """If the exception is already a litellm type, it should pass through."""
        typed_exc = litellm.RateLimitError(
            message="rate limited",
            model="claude-sonnet-4-6",
            llm_provider="anthropic",
        )

        with pytest.raises(litellm.RateLimitError):
            self.handler._handle_error(
                e=typed_exc,
                provider_config=None,
                model="claude-sonnet-4-6",
                custom_llm_provider="anthropic",
            )
