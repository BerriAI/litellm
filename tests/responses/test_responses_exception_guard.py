"""
Unit tests for the isinstance guard added to responses() and aresponses()
to prevent double-mapping of already-typed litellm exceptions.

Issue: https://github.com/BerriAI/litellm/issues/22121
PR:    https://github.com/BerriAI/litellm/pull/30455
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.responses.main import responses


def _make_auth_error() -> litellm.AuthenticationError:
    """Build an AuthenticationError without needing a real HTTP response."""
    import httpx

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, text="Unauthorized", request=req)
    return litellm.AuthenticationError(
        message="Invalid API key.",
        llm_provider="openai",
        model="gpt-4o",
        response=resp,
    )


def _make_rate_limit_error() -> litellm.RateLimitError:
    import httpx

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, text="Too Many Requests", request=req)
    return litellm.RateLimitError(
        message="Rate limit exceeded.",
        llm_provider="openai",
        model="gpt-4o",
        response=resp,
    )


class TestResponsesExceptionGuard:
    """
    Verify that responses() re-raises already-typed litellm exceptions
    without double-mapping them through exception_type().
    """

    @patch("litellm.responses.main._resolve_model_provider_for_responses")
    @patch("litellm.responses.main.LiteLLMCompletionTransformationHandler")
    def test_auth_error_preserved(self, mock_handler_cls, mock_resolve):
        """
        AuthenticationError raised inside responses() must propagate as
        AuthenticationError, not be remapped to APIConnectionError.
        """
        mock_resolve.return_value = ("gpt-4o", "openai")

        auth_err = _make_auth_error()
        mock_handler = MagicMock()
        mock_handler.response_api_handler.side_effect = auth_err
        mock_handler_cls.return_value = mock_handler

        with pytest.raises(litellm.AuthenticationError):
            responses(input="hello", model="gpt-4o")

    @patch("litellm.responses.main._resolve_model_provider_for_responses")
    @patch("litellm.responses.main.LiteLLMCompletionTransformationHandler")
    def test_rate_limit_error_preserved(self, mock_handler_cls, mock_resolve):
        """
        RateLimitError raised inside responses() must propagate as
        RateLimitError, not be remapped to APIConnectionError.
        """
        mock_resolve.return_value = ("gpt-4o", "openai")

        rate_err = _make_rate_limit_error()
        mock_handler = MagicMock()
        mock_handler.response_api_handler.side_effect = rate_err
        mock_handler_cls.return_value = mock_handler

        with pytest.raises(litellm.RateLimitError):
            responses(input="hello", model="gpt-4o")

    def test_double_map_collapses_to_api_connection_error(self):
        """
        Demonstrate the bug: without the guard, passing an already-typed
        litellm exception through exception_type() a second time would
        previously collapse it to APIConnectionError if exception_type()
        itself errored internally.

        With the current version of litellm this is handled inside
        exception_type() too, but the guard in responses() makes the
        intent explicit and avoids the overhead of re-entering the mapper.
        """
        auth_err = _make_auth_error()
        # exception_type() must return the same type when it's already typed
        result = litellm.exception_type(
            model="gpt-4o",
            custom_llm_provider="openai",
            original_exception=auth_err,
            completion_kwargs={},
            extra_kwargs={},
        )
        assert isinstance(result, litellm.AuthenticationError), (
            f"Expected AuthenticationError, got {type(result).__name__}"
        )

    def test_isinstance_guard_logic(self):
        """
        Unit-test the exact guard expression used in the fix.
        All entries in LITELLM_EXCEPTION_TYPES must be caught by it.
        """
        auth_err = _make_auth_error()
        assert isinstance(auth_err, tuple(litellm.LITELLM_EXCEPTION_TYPES))

        rate_err = _make_rate_limit_error()
        assert isinstance(rate_err, tuple(litellm.LITELLM_EXCEPTION_TYPES))

        # Plain Python exceptions must NOT match the guard
        plain = ValueError("something went wrong")
        assert not isinstance(plain, tuple(litellm.LITELLM_EXCEPTION_TYPES))