"""Unit tests for get_retry_from_policy module."""

import pytest

import litellm
from litellm.router_utils.get_retry_from_policy import get_num_retries_from_retry_policy
from litellm.types.router import RetryPolicy


def _make_exc(exc_class):
    return exc_class(message="test", llm_provider="openai", model="gpt-4o")


@pytest.mark.parametrize(
    "exc_class, field, retries",
    [
        (litellm.AuthenticationError, "AuthenticationErrorRetries", 1),
        (litellm.Timeout, "TimeoutErrorRetries", 2),
        (litellm.RateLimitError, "RateLimitErrorRetries", 3),
        (litellm.ContentPolicyViolationError, "ContentPolicyViolationErrorRetries", 4),
        (litellm.BadRequestError, "BadRequestErrorRetries", 5),
        (litellm.InternalServerError, "InternalServerErrorRetries", 6),
    ],
)
def test_get_num_retries_returns_configured_value(exc_class, field, retries):
    policy = RetryPolicy(**{field: retries})
    result = get_num_retries_from_retry_policy(
        exception=_make_exc(exc_class), retry_policy=policy
    )
    assert result == retries


def test_get_num_retries_returns_none_when_policy_is_none():
    assert get_num_retries_from_retry_policy(
        exception=_make_exc(litellm.InternalServerError), retry_policy=None
    ) is None


def test_get_num_retries_returns_none_when_field_not_set():
    policy = RetryPolicy()
    assert get_num_retries_from_retry_policy(
        exception=_make_exc(litellm.InternalServerError), retry_policy=policy
    ) is None
