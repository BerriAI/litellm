"""Test that _handle_error maps exceptions to typed exceptions when model/provider are known.

Covers fix for https://github.com/BerriAI/litellm/issues/20507:
anthropic_messages pass-through was raising BaseLLMException instead of typed
exceptions (RateLimitError, ContextWindowExceededError, etc.), breaking Router
retry/fallback logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

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


class TestHandleErrorWithProviderConfig:
    """Exercise the provider_config (non-None) branch of _handle_error."""

    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()

    def _make_status_error(self, status_code=429, text="rate limit exceeded"):
        mock_response = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            text=text,
        )
        return httpx.HTTPStatusError(
            message=f"{status_code}",
            request=mock_response.request,
            response=mock_response,
        )

    def test_provider_config_without_model_raises_raw_exception(self):
        """provider_config set, no model/provider: raise the raw exception from
        get_error_class unchanged (covers the else branch and the final raise)."""
        raw = BaseLLMException(
            status_code=429,
            message="rate limit exceeded",
            headers={},
        )
        provider_config = MagicMock()
        provider_config.get_error_class.return_value = raw

        with pytest.raises(BaseLLMException) as exc_info:
            self.handler._handle_error(
                e=self._make_status_error(),
                provider_config=provider_config,
            )

        # The exact object from get_error_class is re-raised (no typed mapping).
        assert exc_info.value is raw
        provider_config.get_error_class.assert_called_once()
        call_kwargs = provider_config.get_error_class.call_args.kwargs
        assert call_kwargs["status_code"] == 429
        assert "error_message" in call_kwargs
        assert "headers" in call_kwargs

    def test_provider_config_with_model_maps_to_typed_exception(self):
        """provider_config set plus model/provider: the raw exception is mapped to
        a typed litellm exception via exception_type."""
        raw = BaseLLMException(
            status_code=429,
            message="rate_limit_error: too many requests",
            headers={},
        )
        provider_config = MagicMock()
        provider_config.get_error_class.return_value = raw

        with pytest.raises(litellm.RateLimitError):
            self.handler._handle_error(
                e=self._make_status_error(),
                provider_config=provider_config,
                model="claude-sonnet-4-6",
                custom_llm_provider="anthropic",
            )


class TestAnthropicMessagesRetryErrorMapping:
    """Drive both except branches of the anthropic messages HTTP retry helper."""

    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()

    def _make_provider_config(self):
        provider_config = MagicMock()
        # max_attempts = max(1, 1) -> first attempt is the last, so the retry
        # short-circuit is False and execution falls through to the raise.
        provider_config.max_retry_on_anthropic_messages_http_error = 1
        provider_config.should_retry_anthropic_messages_on_http_error.return_value = (
            False
        )
        provider_config.get_error_class.side_effect = (
            lambda error_message, status_code, headers: BaseLLMException(
                status_code=status_code,
                message=error_message,
                headers=headers,
            )
        )
        return provider_config

    async def _call_retry_helper(self, async_httpx_client, provider_config):
        return await self.handler._async_post_anthropic_messages_with_http_error_retry(
            async_httpx_client=async_httpx_client,
            request_url="https://api.anthropic.com/v1/messages",
            headers={},
            signed_json_body=None,
            request_body={"model": "claude-sonnet-4-6"},
            stream=False,
            logging_obj=MagicMock(),
            provider_config=provider_config,
            litellm_params=litellm.types.router.GenericLiteLLMParams(),
            api_key="sk-test",
            model="claude-sonnet-4-6",
            custom_llm_provider="anthropic",
        )

    async def test_http_status_error_maps_to_typed_exception(self):
        """The except httpx.HTTPStatusError branch re-raises through _handle_error
        with model/custom_llm_provider, producing a typed exception."""
        mock_response = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            text="rate_limit_error: too many requests",
        )
        status_error = httpx.HTTPStatusError(
            message="429",
            request=mock_response.request,
            response=mock_response,
        )
        async_httpx_client = MagicMock()
        async_httpx_client.post = AsyncMock(side_effect=status_error)

        with pytest.raises(litellm.RateLimitError):
            await self._call_retry_helper(
                async_httpx_client, self._make_provider_config()
            )

    async def test_generic_exception_maps_to_typed_exception(self):
        """The bare except Exception branch re-raises through _handle_error with
        model/custom_llm_provider, producing a mapped litellm exception."""
        async_httpx_client = MagicMock()
        async_httpx_client.post = AsyncMock(side_effect=ValueError("boom"))

        provider_config = self._make_provider_config()

        with patch.object(
            self.handler, "_handle_error", wraps=self.handler._handle_error
        ) as spy:
            # A generic exception defaults to status 500, mapping to InternalServerError.
            with pytest.raises(litellm.InternalServerError):
                await self._call_retry_helper(async_httpx_client, provider_config)

        # model and custom_llm_provider were threaded through to _handle_error.
        spy.assert_called_once()
        kwargs = spy.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["custom_llm_provider"] == "anthropic"
        assert isinstance(kwargs["e"], ValueError)


class TestAnthropicMessagesHandlerThreadsProvider:
    """Confirm async_anthropic_messages_handler threads custom_llm_provider into
    the retry helper (the call-site kwarg)."""

    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()

    async def test_custom_llm_provider_passed_to_retry_helper(self):
        provider_config = MagicMock()
        provider_config.validate_anthropic_messages_environment.return_value = (
            {},
            "https://api.anthropic.com",
        )
        provider_config.sign_request.return_value = ({}, None)
        provider_config.transform_anthropic_messages_request.return_value = {
            "model": "claude-sonnet-4-6"
        }
        provider_config.get_complete_url.return_value = (
            "https://api.anthropic.com/v1/messages"
        )

        logging_obj = MagicMock()

        with patch.object(
            self.handler,
            "_async_post_anthropic_messages_with_http_error_retry",
            new=AsyncMock(return_value=MagicMock()),
        ) as retry_mock:
            with patch(
                "litellm.llms.custom_httpx.llm_http_handler.update_headers_with_filtered_beta",
                return_value={},
            ):
                try:
                    await self.handler.async_anthropic_messages_handler(
                        model="claude-sonnet-4-6",
                        messages=[{"role": "user", "content": "hi"}],
                        anthropic_messages_provider_config=provider_config,
                        anthropic_messages_optional_request_params={},
                        custom_llm_provider="anthropic",
                        litellm_params=litellm.types.router.GenericLiteLLMParams(),
                        logging_obj=logging_obj,
                        client=None,
                        api_key="sk-test",
                        api_base="https://api.anthropic.com",
                        stream=False,
                    )
                except Exception:
                    # Post-response transform/logging may fail against mocks; we only
                    # assert the retry helper was reached with the threaded kwarg.
                    pass

        retry_mock.assert_called_once()
        assert retry_mock.call_args.kwargs["custom_llm_provider"] == "anthropic"
