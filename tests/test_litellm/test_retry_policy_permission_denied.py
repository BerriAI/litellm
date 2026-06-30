"""
Unit tests for PermissionDeniedErrorRetries in RetryPolicy.
"""

import httpx
from unittest.mock import MagicMock

import litellm
from litellm.router_utils.get_retry_from_policy import get_num_retries_from_retry_policy
from litellm.types.router import RetryPolicy


def _make_permission_denied() -> litellm.exceptions.PermissionDeniedError:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 403
    response.headers = httpx.Headers({})
    return litellm.exceptions.PermissionDeniedError(
        message="403 Permission Denied",
        llm_provider="openai",
        model="gpt-4o",
        response=response,
    )


def test_permission_denied_retries_returns_configured_value():
    """PermissionDeniedErrorRetries is returned when a PermissionDeniedError is raised."""
    policy = RetryPolicy(PermissionDeniedErrorRetries=0)
    result = get_num_retries_from_retry_policy(
        exception=_make_permission_denied(),
        retry_policy=policy,
    )
    assert result == 0


def test_permission_denied_retries_none_when_not_configured():
    """Returns None (falls through to default) when PermissionDeniedErrorRetries is not set."""
    policy = RetryPolicy(RateLimitErrorRetries=3)
    result = get_num_retries_from_retry_policy(
        exception=_make_permission_denied(),
        retry_policy=policy,
    )
    assert result is None
