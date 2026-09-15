"""
Tests for retry back-off when a pre-call check pins a request to one deployment.

``Router._time_to_sleep_before_retry`` retries instantly whenever the model group
still reports a healthy deployment, on the assumption that the retry can fail
over to it. That assumption does not hold when a pre-call check has pinned the
request to a specific deployment: the remaining healthy deployments have already
been rejected for this request, so an immediate retry can only raise the same
error again. Errors that know this set ``no_compatible_deployment_available``.
"""

import asyncio
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
    EncryptedContentAffinityCheck,
)

MODEL_GROUP = "openai.gpt-5.1-codex"
ORIGIN_ID = "deployment-origin"
INCOMPATIBLE_PEER_ID = "deployment-incompatible-peer"
COOLDOWN_TIME = 5
NUM_RETRIES = 2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rate_limit_error(retry_after: int) -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="rate limited",
        llm_provider="",
        model=MODEL_GROUP,
        response=httpx.Response(
            status_code=429,
            headers={"retry-after": str(retry_after)},
            request=httpx.Request("POST", "https://litellm.ai/"),
        ),
    )


def _pinned_router() -> Router:
    """
    Two deployments in one model group on different encryption boundaries, so the
    peer can never serve a request pinned to the origin.
    """
    # Each Router appends its own EncryptedContentAffinityCheck to the process
    # global litellm.callbacks, so drop checks left behind by earlier tests.
    litellm.callbacks[:] = [
        callback
        for callback in litellm.callbacks
        if not isinstance(callback, EncryptedContentAffinityCheck)
    ]
    return Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_base": "https://account-a.openai.azure.com/",
                    "api_key": "key-a",
                },
                "model_info": {"id": ORIGIN_ID},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_base": "https://account-b.openai.azure.com/",
                    "api_key": "key-b",
                },
                "model_info": {"id": INCOMPATIBLE_PEER_ID},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        enable_pre_call_checks=True,
        allowed_fails=3,
        cooldown_time=COOLDOWN_TIME,
        num_retries=NUM_RETRIES,
    )


def _pinned_input() -> list:
    """A follow-up request carrying encrypted content minted by the origin."""
    return [
        {
            "type": "reasoning",
            "encrypted_content": ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(
                encrypted_content="rsn_test-reasoning-state",
                model_id=ORIGIN_ID,
            ),
        }
    ]


def _cool_down_origin(router: Router, status_code: str = "429") -> None:
    router.cooldown_cache.add_deployment_to_cooldown(
        model_id=ORIGIN_ID,
        original_exception=Exception("rate limited"),
        exception_status=status_code,
        cooldown_time=float(COOLDOWN_TIME),
    )


# ---------------------------------------------------------------------------
# Unit tests for _time_to_sleep_before_retry
# ---------------------------------------------------------------------------


def test_pinned_error_honors_retry_after_despite_healthy_deployment():
    """A pinned request cannot use the healthy peer, so it must wait."""
    router = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/gpt-5.1-codex", "api_key": "key-a"},
                "model_info": {"id": ORIGIN_ID},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/gpt-5.1-codex", "api_key": "key-b"},
                "model_info": {"id": INCOMPATIBLE_PEER_ID},
            },
        ]
    )
    error = _rate_limit_error(retry_after=4)
    error.no_compatible_deployment_available = True

    timeout = router._time_to_sleep_before_retry(
        e=error,
        remaining_retries=2,
        num_retries=2,
        healthy_deployments=[{"model_info": {"id": INCOMPATIBLE_PEER_ID}}],
        all_deployments=router.model_list,
    )

    assert timeout >= 4


def test_unpinned_error_still_fails_over_instantly():
    """A stateless 429 can use the healthy peer, so behavior is unchanged."""
    router = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/gpt-5.1-codex", "api_key": "key-a"},
                "model_info": {"id": ORIGIN_ID},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/gpt-5.1-codex", "api_key": "key-b"},
                "model_info": {"id": INCOMPATIBLE_PEER_ID},
            },
        ]
    )

    timeout = router._time_to_sleep_before_retry(
        e=_rate_limit_error(retry_after=4),
        remaining_retries=2,
        num_retries=2,
        healthy_deployments=[{"model_info": {"id": INCOMPATIBLE_PEER_ID}}],
        all_deployments=router.model_list,
    )

    assert timeout == 0


# ---------------------------------------------------------------------------
# The affinity check publishes the signal the router reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_affinity_rate_limit_error_is_marked_and_waited_out():
    router = _pinned_router()
    _cool_down_origin(router)

    with pytest.raises(litellm.RateLimitError) as excinfo:
        await router.async_get_available_deployment(
            model=MODEL_GROUP,
            request_kwargs={"input": _pinned_input()},
            input=_pinned_input(),
        )

    error = excinfo.value
    retry_after = int(error.response.headers["retry-after"])
    assert getattr(error, "no_compatible_deployment_available", False) is True
    assert 1 <= retry_after <= COOLDOWN_TIME

    healthy, all_deployments = await router._async_get_healthy_deployments(
        model=MODEL_GROUP,
        parent_otel_span=None,
    )
    # The peer is healthy, yet the retry must still wait for the origin.
    assert [d["model_info"]["id"] for d in healthy] == [INCOMPATIBLE_PEER_ID]
    assert (
        router._time_to_sleep_before_retry(
            e=error,
            remaining_retries=NUM_RETRIES,
            num_retries=NUM_RETRIES,
            healthy_deployments=healthy,
            all_deployments=all_deployments,
        )
        >= retry_after
    )


@pytest.mark.asyncio
async def test_affinity_service_unavailable_error_publishes_its_cooldown():
    """A non-429 cooldown must advertise the same remaining window."""
    router = _pinned_router()
    _cool_down_origin(router, status_code="500")

    with pytest.raises(litellm.ServiceUnavailableError) as excinfo:
        await router.async_get_available_deployment(
            model=MODEL_GROUP,
            request_kwargs={"input": _pinned_input()},
            input=_pinned_input(),
        )

    error = excinfo.value
    retry_after = int(error.response.headers["retry-after"])
    assert getattr(error, "no_compatible_deployment_available", False) is True
    assert 1 <= retry_after <= COOLDOWN_TIME

    healthy, all_deployments = await router._async_get_healthy_deployments(
        model=MODEL_GROUP,
        parent_otel_span=None,
    )
    assert (
        router._time_to_sleep_before_retry(
            e=error,
            remaining_retries=NUM_RETRIES,
            num_retries=NUM_RETRIES,
            healthy_deployments=healthy,
            all_deployments=all_deployments,
        )
        >= retry_after
    )


# ---------------------------------------------------------------------------
# End-to-end: retries are not spent inside the origin's cooldown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_request_does_not_exhaust_retries_inside_the_cooldown():
    """
    With a five-second cooldown on the originating deployment and one healthy but
    incompatible fallback, the retries must be spread across the cooldown window
    and the incompatible fallback must never be selected.
    """
    router = _pinned_router()
    _cool_down_origin(router)
    request_input = _pinned_input()
    attempts: list[str] = []
    selected: list[str] = []

    async def attempt(**kwargs: Any) -> Any:
        attempts.append(kwargs["model"])
        deployment = await router.async_get_available_deployment(
            model=kwargs["model"],
            request_kwargs=kwargs,
            input=kwargs["input"],
        )
        selected.append(deployment["model_info"]["id"])
        return deployment

    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        # Record what the router decided to wait, then yield without spending it,
        # so the test measures the decision rather than real wall-clock time.
        slept.append(delay)
        return await real_sleep(0)

    with patch("asyncio.sleep", new=recording_sleep):
        with pytest.raises(litellm.RateLimitError):
            await router.async_function_with_retries(
                original_function=attempt,
                model=MODEL_GROUP,
                input=request_input,
                # A Responses request always carries this; log_retry writes its
                # retry breadcrumb into it.
                litellm_metadata={},
                num_retries=NUM_RETRIES,
            )

    assert selected == []
    assert len(attempts) == NUM_RETRIES + 1
    # Upstream also waits once after the final attempt before giving up, so the
    # count is a lower bound. What matters is that no wait is an instant retry.
    assert len(slept) >= NUM_RETRIES
    assert all(delay >= 1 for delay in slept)
    assert sum(slept) >= COOLDOWN_TIME - 1
