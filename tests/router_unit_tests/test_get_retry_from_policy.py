"""Direct coverage for litellm/router_utils/get_retry_from_policy.py"""

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import pytest
import litellm
from litellm.types.router import RetryPolicy
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    NotFoundError,
    RateLimitError,
    Timeout,
)
from litellm.router_utils.get_retry_from_policy import get_num_retries_from_retry_policy


@pytest.mark.parametrize(
    "exception_type, retry_policy, expected",
    [
        (BadRequestError, RetryPolicy(BadRequestErrorRetries=1), 1),
        (AuthenticationError, RetryPolicy(AuthenticationErrorRetries=2), 2),
        (Timeout, RetryPolicy(TimeoutErrorRetries=3), 3),
        (RateLimitError, RetryPolicy(RateLimitErrorRetries=4), 4),
        (
            ContentPolicyViolationError,
            RetryPolicy(ContentPolicyViolationErrorRetries=5),
            5,
        ),
        (NotFoundError, RetryPolicy(NotFoundErrorRetries=0), 0),
        (NotFoundError, RetryPolicy(NotFoundErrorRetries=2), 2),
    ],
)
def test_get_num_retries_from_retry_policy_direct(
    exception_type, retry_policy, expected
):
    exception = exception_type(message="test", llm_provider="openai", model="gpt-5-mini")
    assert get_num_retries_from_retry_policy(exception, retry_policy) == expected


def test_get_num_retries_from_retry_policy_notfound_none_direct():
    retry_policy = RetryPolicy()
    exception = NotFoundError(message="test", llm_provider="openai", model="gpt-5-mini")
    assert get_num_retries_from_retry_policy(exception, retry_policy) is None


def test_get_num_retries_from_retry_policy_unknown_exception_direct():
    retry_policy = RetryPolicy(BadRequestErrorRetries=3)
    exception = Exception("unknown")
    assert get_num_retries_from_retry_policy(exception, retry_policy) is None
