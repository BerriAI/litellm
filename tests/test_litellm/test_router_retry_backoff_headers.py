"""
Tests for router retry backoff behavior.
"""

import asyncio
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.constants import MAX_RETRY_DELAY

_BACKOFF_DETECTION_TIMEOUT: Final = MAX_RETRY_DELAY / 4


def _router_with_single_failing_deployment(fallbacks: list[dict[str, list[str]]]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "sk-test",
                    "mock_response": "litellm.InternalServerError",
                },
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "sk-test",
                    "mock_response": "answered by backup",
                },
            },
        ],
        num_retries=2,
        retry_after=int(MAX_RETRY_DELAY),
        fallbacks=fallbacks,
    )


@pytest.mark.asyncio
async def test_single_deployment_group_with_fallback_does_not_back_off_before_falling_back():
    router: Final = _router_with_single_failing_deployment(fallbacks=[{"primary": ["backup"]}])

    response: Final = await asyncio.wait_for(
        router.acompletion(model="primary", messages=[{"role": "user", "content": "Hello"}]),
        timeout=_BACKOFF_DETECTION_TIMEOUT,
    )

    assert response.choices[0].message.content == "answered by backup"


@pytest.mark.asyncio
async def test_fallback_configured_for_another_group_keeps_retry_backoff():
    router: Final = _router_with_single_failing_deployment(fallbacks=[{"backup": ["primary"]}])

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            router.acompletion(model="primary", messages=[{"role": "user", "content": "Hello"}]),
            timeout=_BACKOFF_DETECTION_TIMEOUT,
        )


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
