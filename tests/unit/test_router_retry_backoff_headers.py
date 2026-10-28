"""
Tests for router retry backoff behavior.
"""

import asyncio
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.constants import MAX_RETRY_DELAY
from litellm.router import _untried_fallback_target_exists
from litellm.router_utils.fallback_event_handlers import AttemptedFallbackTargets, record_disable_fallbacks
from litellm.types.router import RetryPolicy

_SERVER_ERROR: Final = litellm.InternalServerError(message="provider down", model="gpt-5.4-mini", llm_provider="openai")
_CONTENT_POLICY_ERROR: Final = litellm.ContentPolicyViolationError(
    message="flagged", model="gpt-5.4-mini", llm_provider="openai"
)
_CONTEXT_WINDOW_ERROR: Final = litellm.ContextWindowExceededError(
    message="too long", model="gpt-5.4-mini", llm_provider="openai"
)


def _router_with_single_failing_deployment(
    fallbacks: list[dict[str, list[str]]],
    primary_error: str | None = "litellm.InternalServerError",
    content_policy_fallbacks: list[dict[str, list[str]]] | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-test"}
                | ({"mock_response": primary_error} if primary_error is not None else {}),
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
        content_policy_fallbacks=content_policy_fallbacks,
        retry_policy=retry_policy,
    )


def _backoff_delays(sleeper: AsyncMock) -> tuple[float, ...]:
    """
    The delays the router asked to wait between retries. Reading its decision keeps these
    tests off the clock, which tests/unit rules out, and off the CI scheduler's timing.
    """
    return tuple(call.args[0] for call in sleeper.await_args_list if call.args)


@pytest.mark.asyncio
async def test_single_deployment_group_with_fallback_does_not_back_off_before_falling_back():
    router: Final = _router_with_single_failing_deployment(fallbacks=[{"primary": ["backup"]}])
    sleeper: Final = AsyncMock()

    with patch.object(asyncio, "sleep", sleeper):
        response: Final = await router.acompletion(model="primary", messages=[{"role": "user", "content": "Hello"}])

    assert response.choices[0].message.content == "answered by backup"
    assert not any(delay > 0 for delay in _backoff_delays(sleeper))


@pytest.mark.asyncio
async def test_fallback_configured_for_another_group_keeps_retry_backoff():
    router: Final = _router_with_single_failing_deployment(fallbacks=[{"backup": ["primary"]}])
    sleeper: Final = AsyncMock()

    with patch.object(asyncio, "sleep", sleeper), pytest.raises(litellm.InternalServerError):
        await router.acompletion(model="primary", messages=[{"role": "user", "content": "Hello"}])

    assert any(delay > 0 for delay in _backoff_delays(sleeper))


@pytest.mark.asyncio
async def test_empty_fallback_chain_keeps_retry_backoff():
    """A chain configured as an empty list names no target, so the request has nothing to
    fall back to and the retries must still space themselves out."""
    router: Final = _router_with_single_failing_deployment(fallbacks=[{"primary": []}])
    sleeper: Final = AsyncMock()

    with patch.object(asyncio, "sleep", sleeper), pytest.raises(litellm.InternalServerError):
        await router.acompletion(model="primary", messages=[{"role": "user", "content": "Hello"}])

    assert any(delay > 0 for delay in _backoff_delays(sleeper))


def test_untried_fallback_target_exists_is_false_for_a_chain_with_no_entries():
    assert _untried_fallback_target_exists(["backup"], {}) is True
    assert _untried_fallback_target_exists([], {}) is False
    assert _untried_fallback_target_exists(None, {}) is False


@pytest.mark.asyncio
async def test_client_side_fallback_list_does_not_back_off_before_falling_back():
    router: Final = _router_with_single_failing_deployment(fallbacks=[])

    sleeper: Final = AsyncMock()

    with patch.object(asyncio, "sleep", sleeper):
        response: Final = await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "Hello"}],
            fallbacks=[{"model": "backup"}],
        )

    assert response.choices[0].message.content == "answered by backup"
    assert not any(delay > 0 for delay in _backoff_delays(sleeper))


@pytest.mark.asyncio
async def test_content_policy_error_keeps_backoff_when_its_dedicated_fallbacks_skip_the_group():
    router: Final = _router_with_single_failing_deployment(
        fallbacks=[{"primary": ["backup"]}],
        primary_error=None,
        content_policy_fallbacks=[{"backup": ["primary"]}],
        retry_policy=RetryPolicy(ContentPolicyViolationErrorRetries=2),
    )

    sleeper: Final = AsyncMock()

    with patch.object(asyncio, "sleep", sleeper), pytest.raises(litellm.ContentPolicyViolationError):
        await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response=_CONTENT_POLICY_ERROR,
        )

    assert any(delay > 0 for delay in _backoff_delays(sleeper))


@pytest.mark.parametrize(
    ("error", "fallbacks", "context_window_fallbacks", "content_policy_fallbacks", "expected"),
    [
        pytest.param(_SERVER_ERROR, [{"primary": ["backup"]}], None, None, True, id="own-chain"),
        pytest.param(_SERVER_ERROR, [{"backup": ["primary"]}], None, None, False, id="chain-for-another-group"),
        pytest.param(_SERVER_ERROR, [{"*": ["backup"]}], None, None, True, id="generic-chain"),
        pytest.param(_SERVER_ERROR, [{"model": "backup"}], None, None, True, id="client-side-list"),
        pytest.param(
            _CONTENT_POLICY_ERROR,
            [{"primary": ["backup"]}],
            None,
            [{"backup": ["primary"]}],
            False,
            id="content-policy-list-skips-group",
        ),
        pytest.param(
            _CONTENT_POLICY_ERROR,
            [{"backup": ["primary"]}],
            None,
            [{"primary": ["backup"]}],
            True,
            id="content-policy-list-covers-group",
        ),
        pytest.param(
            _CONTEXT_WINDOW_ERROR,
            [{"primary": ["backup"]}],
            [{"backup": ["primary"]}],
            None,
            False,
            id="context-window-list-skips-group",
        ),
        pytest.param(_CONTENT_POLICY_ERROR, [{"primary": ["backup"]}], None, None, True, id="no-dedicated-list"),
    ],
)
def test_fallback_available_for_error_follows_the_dispatch_order(
    error: Exception,
    fallbacks: list[dict[str, object]],
    context_window_fallbacks: list[dict[str, list[str]]] | None,
    content_policy_fallbacks: list[dict[str, list[str]]] | None,
    expected: bool,
):
    router: Final = _router_with_single_failing_deployment(fallbacks=[])

    available: Final = router._fallback_available_for_error(
        error=error,
        fallbacks=fallbacks,
        context_window_fallbacks=context_window_fallbacks,
        content_policy_fallbacks=content_policy_fallbacks,
        model_group="primary",
        kwargs={"model": "primary"},
    )

    assert available is expected


def test_fallback_available_for_error_is_false_without_a_model_group_or_when_disabled():
    router: Final = _router_with_single_failing_deployment(fallbacks=[])
    disabled_kwargs: Final = {"model": "primary", "metadata": {}}
    record_disable_fallbacks(disabled_kwargs, True)

    def available(model_group: str | None, kwargs: dict[str, object]) -> bool:
        return router._fallback_available_for_error(
            error=_SERVER_ERROR,
            fallbacks=[{"model": "backup"}],
            context_window_fallbacks=None,
            content_policy_fallbacks=None,
            model_group=model_group,
            kwargs=kwargs,
        )

    assert (available("primary", {"model": "primary"}), available(None, {}), available("primary", disabled_kwargs)) == (
        True,
        False,
        False,
    )


def test_regular_fallback_available_is_false_once_the_chain_is_used_up_or_disabled():
    router: Final = _router_with_single_failing_deployment(fallbacks=[])
    chain: Final = [{"primary": ["backup"]}]
    disabled_kwargs: Final = {"model": "primary", "metadata": {}}
    record_disable_fallbacks(disabled_kwargs, True)

    fresh: Final = router._regular_fallback_available(
        fallbacks=chain, model_group="primary", kwargs={"model": "primary"}
    )
    used_up: Final = router._regular_fallback_available(
        fallbacks=chain,
        model_group="primary",
        kwargs={"model": "primary", "attempted_targets": AttemptedFallbackTargets(keys=frozenset({"backup"}))},
    )
    disabled: Final = router._regular_fallback_available(fallbacks=chain, model_group="primary", kwargs=disabled_kwargs)

    assert (fresh, used_up, disabled) == (True, False, False)


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
