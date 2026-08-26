"""Tests for the router capability cache and the order/exclusion filter ordering.

Covers three behaviours introduced together:

1. A provider answering 404 for a model no longer aborts the request. The retry
   loop is allowed to advance to the next-priority deployment.
2. `async_get_healthy_deployments` applies weighted-failover exclusion BEFORE the
   `model_info.order` filter, so excluded order-1 deployments fall through to
   order-2 instead of emptying the candidate list.
3. A DualCache entry per (deployment id, model group) records which providers
   answered 404, so later requests skip them until the TTL expires.
"""

import pytest

import litellm
from litellm import Router

MODEL_GROUP = "universal-group"


def _two_tier_router() -> Router:
    """Two deployments in one group: order 1 and order 2."""
    return Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/tier-one", "api_key": "sk-tier-one"},
                "model_info": {"id": "dep-order-1", "order": 1},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/tier-two", "api_key": "sk-tier-two"},
                "model_info": {"id": "dep-order-2", "order": 2},
            },
        ]
    )


def _flat_router() -> Router:
    """Two deployments in one group, no order set."""
    return Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/alpha", "api_key": "sk-alpha"},
                "model_info": {"id": "dep-alpha"},
            },
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": "openai/beta", "api_key": "sk-beta"},
                "model_info": {"id": "dep-beta"},
            },
        ]
    )


def _not_found() -> litellm.NotFoundError:
    return litellm.NotFoundError(
        message="The model 'some-model' does not exist",
        model=MODEL_GROUP,
        llm_provider="openai",
    )


def _ids(deployments) -> set:
    return {d["model_info"]["id"] for d in deployments}


# ---------------------------------------------------------------- retry gating


def test_404_retries_while_other_deployments_remain():
    """A 404 returns True so the retry loop moves to the next deployment."""
    router = _two_tier_router()

    assert (
        router.should_retry_this_error(
            error=_not_found(),
            healthy_deployments=[{"model_info": {"id": "dep-order-2"}}],
            all_deployments=router.model_list,
        )
        is True
    )


def test_404_raises_once_nothing_healthy_is_left():
    """The last 404 still surfaces to the caller instead of retrying forever."""
    router = _two_tier_router()

    with pytest.raises(litellm.NotFoundError):
        router.should_retry_this_error(
            error=_not_found(),
            healthy_deployments=[],
            all_deployments=router.model_list,
        )


@pytest.mark.parametrize(
    "error",
    [
        litellm.BadRequestError(message="bad", model=MODEL_GROUP, llm_provider="openai"),
        litellm.ContextWindowExceededError(
            message="too long", model=MODEL_GROUP, llm_provider="openai"
        ),
    ],
    ids=["400", "context-window"],
)
def test_other_4xx_errors_still_raise(error):
    """Only 404 was exempted; other non-retryable 4xx keep aborting the request."""
    router = _two_tier_router()

    with pytest.raises(type(error)):
        router.should_retry_this_error(
            error=error,
            healthy_deployments=[{"model_info": {"id": "dep-order-2"}}],
            all_deployments=router.model_list,
        )


# ------------------------------------------------------- exclusion before order


@pytest.mark.asyncio
async def test_excluded_order_1_falls_through_to_order_2():
    """Regression: excluding every order-1 deployment used to empty the list.

    The order filter narrowed to order-1 first, exclusion then removed all of
    them, and the caller saw "No deployments available" while a healthy order-2
    deployment was sitting right there.
    """
    router = _two_tier_router()

    deployments = await router.async_get_healthy_deployments(
        model=MODEL_GROUP,
        request_kwargs={"_excluded_deployment_ids": ["dep-order-1"]},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert _ids(deployments) == {"dep-order-2"}


@pytest.mark.asyncio
async def test_order_1_preferred_when_nothing_is_excluded():
    """The order filter still wins when exclusion removes nothing."""
    router = _two_tier_router()

    deployments = await router.async_get_healthy_deployments(
        model=MODEL_GROUP,
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert _ids(deployments) == {"dep-order-1"}


# --------------------------------------------------------------- capability cache


@pytest.mark.asyncio
async def test_incapable_deployment_is_skipped():
    """A deployment cached as incapable drops out of the candidate list."""
    router = _flat_router()
    await router.cache.async_set_cache(
        key=f"litellm:cap:dep-alpha:{MODEL_GROUP}", value=False, ttl=3600
    )

    deployments = await router.async_get_healthy_deployments(
        model=MODEL_GROUP,
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert _ids(deployments) == {"dep-beta"}


@pytest.mark.asyncio
async def test_capable_and_unknown_deployments_are_kept():
    """True means keep; a missing entry means untested, so also keep."""
    router = _flat_router()
    await router.cache.async_set_cache(
        key=f"litellm:cap:dep-alpha:{MODEL_GROUP}", value=True, ttl=86400
    )

    deployments = await router.async_get_healthy_deployments(
        model=MODEL_GROUP,
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert _ids(deployments) == {"dep-alpha", "dep-beta"}


@pytest.mark.asyncio
async def test_all_incapable_falls_back_to_the_full_list():
    """With every candidate cached incapable the filter is skipped.

    Better to re-probe a stale cache than to refuse a request outright.
    """
    router = _flat_router()
    for dep_id in ("dep-alpha", "dep-beta"):
        await router.cache.async_set_cache(
            key=f"litellm:cap:{dep_id}:{MODEL_GROUP}", value=False, ttl=3600
        )

    deployments = await router.async_get_healthy_deployments(
        model=MODEL_GROUP,
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert _ids(deployments) == {"dep-alpha", "dep-beta"}
