"""
Tests for get_num_retries_from_retry_policy in litellm/router_utils/get_retry_from_policy.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.exceptions import BadRequestError, ContextWindowExceededError
from litellm.router_utils.get_retry_from_policy import get_num_retries_from_retry_policy
from litellm.types.router import RetryPolicy


def _make_context_window_error() -> ContextWindowExceededError:
    return ContextWindowExceededError(
        message="context window exceeded",
        llm_provider="azure",
        model="gpt-4o",
    )


class TestContextWindowExceededRetryPolicy:
    """ContextWindowExceededError should not be retried by default."""

    def test_context_window_error_returns_zero_with_default_policy(self):
        """Even when BadRequestErrorRetries is set, ContextWindowExceededError should return 0."""
        policy = RetryPolicy(BadRequestErrorRetries=5)
        retries = get_num_retries_from_retry_policy(
            exception=_make_context_window_error(),
            retry_policy=policy,
        )
        assert retries == 0

    def test_context_window_error_returns_zero_with_empty_policy(self):
        """With an empty retry policy, ContextWindowExceededError should return 0."""
        policy = RetryPolicy()
        retries = get_num_retries_from_retry_policy(
            exception=_make_context_window_error(),
            retry_policy=policy,
        )
        assert retries == 0

    def test_context_window_error_respects_explicit_override(self):
        """If user explicitly sets ContextWindowExceededErrorRetries, honor it."""
        policy = RetryPolicy(ContextWindowExceededErrorRetries=3)
        retries = get_num_retries_from_retry_policy(
            exception=_make_context_window_error(),
            retry_policy=policy,
        )
        assert retries == 3

    def test_bad_request_error_still_retried(self):
        """Regular BadRequestError should still be retried when policy is set."""
        policy = RetryPolicy(BadRequestErrorRetries=5)
        retries = get_num_retries_from_retry_policy(
            exception=BadRequestError(
                message="bad request",
                llm_provider="azure",
                model="gpt-4o",
            ),
            retry_policy=policy,
        )
        assert retries == 5
