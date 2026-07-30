"""
Tests for router retry backoff behavior.
"""

from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm import Router


@pytest.mark.asyncio
async def test_retry_backoff_uses_current_exception_headers():
    """
    Ensure retry backoff uses the current retry exception, not the initial one.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
            }
        ],
        num_retries=2,
    )

    first_error = litellm.RateLimitError(
        message="Rate limited on first attempt",
        model="gpt-3.5-turbo",
        llm_provider="openai",
    )
    first_error.litellm_response_headers = httpx.Headers({"retry-after": "1"})

    second_error = litellm.RateLimitError(
        message="Rate limited on second attempt",
        model="gpt-3.5-turbo",
        llm_provider="openai",
    )
    second_error.litellm_response_headers = httpx.Headers({"retry-after": "15"})

    third_error = litellm.RateLimitError(
        message="Rate limited on third attempt",
        model="gpt-3.5-turbo",
        llm_provider="openai",
    )
    third_error.litellm_response_headers = httpx.Headers({"retry-after": "30"})

    raised_errors = [first_error, second_error, third_error]
    captured_backoff_errors = []

    async def mock_make_call(*args, **kwargs):
        raise raised_errors.pop(0)

    def mock_time_to_sleep_before_retry(*args, **kwargs):
        captured_backoff_errors.append(kwargs["e"])
        return 0.01

    with patch.object(router, "make_call", side_effect=mock_make_call):
        with patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(
                [{"model_info": {"id": "test-id"}}],
                [{"model_info": {"id": "test-id"}}],
            ),
        ):
            with patch.object(
                router,
                "_time_to_sleep_before_retry",
                side_effect=mock_time_to_sleep_before_retry,
            ):
                with pytest.raises(litellm.RateLimitError):
                    await router.acompletion(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

    # Router computes backoff once after the initial failure, then once per failed retry.
    # With num_retries=2 and all attempts failing, that's 1 + 2 = 3 invocations.
    assert len(captured_backoff_errors) == router.num_retries + 1
    assert captured_backoff_errors[0] is first_error
    assert captured_backoff_errors[1] is second_error
    assert captured_backoff_errors[2] is third_error


def test_time_to_sleep_before_retry_default_ignores_retry_after_with_multiple_deployments():
    """
    Default behavior (respect_retry_after_with_multiple_deployments=False,
    the default) is unchanged: instant retry (timeout == 0) when there are
    healthy sibling deployments, even with an explicit Retry-After header.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-a"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-b"},
            },
        ],
        num_retries=3,
    )

    error = litellm.RateLimitError(
        message="Rate limited",
        model="gpt-3.5-turbo",
        llm_provider="openai",
    )
    error.litellm_response_headers = httpx.Headers({"retry-after": "15"})

    healthy_deployments = [
        {"model_info": {"id": "dep-a"}},
        {"model_info": {"id": "dep-b"}},
    ]

    timeout = router._time_to_sleep_before_retry(
        e=error,
        remaining_retries=2,
        num_retries=3,
        healthy_deployments=healthy_deployments,
        all_deployments=healthy_deployments,
    )

    assert timeout == 0


def test_time_to_sleep_before_retry_opt_in_honors_retry_after_with_multiple_deployments():
    """
    Regression test for: https://github.com/BerriAI/litellm/issues/34399

    With respect_retry_after_with_multiple_deployments=True, an explicit
    Retry-After header must be honored even when there are multiple healthy
    deployments in the model group -- sibling deployments frequently share
    the same throttled upstream, so instant failover just hammers it.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-a"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-b"},
            },
        ],
        num_retries=3,
        respect_retry_after_with_multiple_deployments=True,
    )

    error = litellm.RateLimitError(
        message="Rate limited",
        model="gpt-3.5-turbo",
        llm_provider="openai",
    )
    error.litellm_response_headers = httpx.Headers({"retry-after": "15"})

    healthy_deployments = [
        {"model_info": {"id": "dep-a"}},
        {"model_info": {"id": "dep-b"}},
    ]

    timeout = router._time_to_sleep_before_retry(
        e=error,
        remaining_retries=2,
        num_retries=3,
        healthy_deployments=healthy_deployments,
        all_deployments=healthy_deployments,
    )

    assert timeout != 0
    assert timeout > 0


def test_time_to_sleep_before_retry_opt_in_still_instant_without_retry_after():
    """
    Sanity check: even with respect_retry_after_with_multiple_deployments=True,
    multi-deployment groups still get instant failover (timeout == 0) when
    there is no explicit Retry-After header -- the opt-in only changes
    behavior when the provider actually sent one.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-a"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-b"},
            },
        ],
        num_retries=3,
        respect_retry_after_with_multiple_deployments=True,
    )

    error = litellm.RateLimitError(
        message="Rate limited, no retry-after header",
        model="gpt-3.5-turbo",
        llm_provider="openai",
    )

    healthy_deployments = [
        {"model_info": {"id": "dep-a"}},
        {"model_info": {"id": "dep-b"}},
    ]

    timeout = router._time_to_sleep_before_retry(
        e=error,
        remaining_retries=2,
        num_retries=3,
        healthy_deployments=healthy_deployments,
        all_deployments=healthy_deployments,
    )

    assert timeout == 0
