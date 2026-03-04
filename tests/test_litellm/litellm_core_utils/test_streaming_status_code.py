"""
Tests for _handle_stream_fallback_error preserving correct HTTP status codes.

Regression tests for https://github.com/BerriAI/litellm/issues/18689
"""

from unittest.mock import MagicMock

import httpx
import pytest

from litellm.exceptions import (
    ContextWindowExceededError,
    MidStreamFallbackError,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper


def _make_stream_wrapper(**overrides) -> CustomStreamWrapper:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    defaults = dict(
        completion_stream=None,
        model="test-model",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    defaults.update(overrides)
    return CustomStreamWrapper(**defaults)


# ---------- ContextWindowExceededError must keep 400 ----------


def test_context_window_error_preserves_400():
    """ContextWindowExceededError should be re-raised directly with status 400,
    not wrapped in MidStreamFallbackError (503)."""
    wrapper = _make_stream_wrapper()
    exc = ContextWindowExceededError(
        message="max_tokens exceeds model context window",
        model="test-model",
        llm_provider="openai",
    )

    with pytest.raises(ContextWindowExceededError) as exc_info:
        wrapper._handle_stream_fallback_error(exc)

    assert exc_info.value.status_code == 400


# ---------- Retriable errors get MidStreamFallbackError ----------


def test_retriable_429_gets_mid_stream_fallback():
    """429 rate-limit errors should be wrapped in MidStreamFallbackError
    so the Router can fall back to another model."""
    wrapper = _make_stream_wrapper()

    from litellm.exceptions import RateLimitError

    exc = RateLimitError(
        message="rate limited",
        model="test-model",
        llm_provider="openai",
    )

    with pytest.raises(MidStreamFallbackError):
        wrapper._handle_stream_fallback_error(exc)


# ---------- Non-retriable 4xx errors raised directly ----------


def test_auth_error_raised_directly():
    """401 AuthenticationError should be raised directly, not wrapped."""
    wrapper = _make_stream_wrapper()

    from litellm.exceptions import AuthenticationError

    exc = AuthenticationError(
        message="invalid api key",
        model="test-model",
        llm_provider="openai",
    )

    with pytest.raises(AuthenticationError):
        wrapper._handle_stream_fallback_error(exc)


def test_permission_error_raised_directly():
    """403 PermissionDeniedError should be raised directly, not wrapped."""
    wrapper = _make_stream_wrapper()

    from litellm.exceptions import PermissionDeniedError

    exc = PermissionDeniedError(
        message="forbidden",
        model="test-model",
        llm_provider="openai",
        response=httpx.Response(403, request=httpx.Request("POST", "https://api.openai.com")),
    )

    with pytest.raises(PermissionDeniedError):
        wrapper._handle_stream_fallback_error(exc)
