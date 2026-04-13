"""Unit tests for litellm.router_utils.get_retry_from_policy."""

import litellm
from litellm.router_utils.get_retry_from_policy import (
    get_num_retries_from_retry_policy,
)
from litellm.types.router import RetryPolicy


def test_internal_server_error_retries_is_honored():
    """Regression: `InternalServerErrorRetries` must be returned for
    `InternalServerError` exceptions. Previously the dispatcher had no
    branch for this field and silently returned `None`, causing the
    caller to fall back to `num_retries`."""
    retry_policy = RetryPolicy(InternalServerErrorRetries=5)
    exc = litellm.exceptions.InternalServerError(
        message="test", llm_provider="openai", model="gpt-3.5-turbo"
    )

    num_retries = get_num_retries_from_retry_policy(
        exception=exc, retry_policy=retry_policy
    )

    assert num_retries == 5


def test_internal_server_error_retries_unset_returns_none():
    """When the field is not set, the dispatcher should return `None`
    so the caller falls back to `num_retries`."""
    retry_policy = RetryPolicy()
    exc = litellm.exceptions.InternalServerError(
        message="test", llm_provider="openai", model="gpt-3.5-turbo"
    )

    num_retries = get_num_retries_from_retry_policy(
        exception=exc, retry_policy=retry_policy
    )

    assert num_retries is None
