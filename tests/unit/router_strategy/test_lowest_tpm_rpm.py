import asyncio
from datetime import datetime, timedelta
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm import Router, token_counter
from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict, RouterErrors

MODEL_GROUP: Final = "lowest-tpm-router"
HIGH_USAGE_DEPLOYMENT_ID: Final = "highest-usage"
LOW_USAGE_DEPLOYMENT_ID: Final = "lowest-usage"


def _deployment(deployment_id: str) -> DeploymentTypedDict:
    params: LiteLLMParamsTypedDict = {
        "model": "gpt-4o",
        "api_key": "key",
        "mock_response": f"from {deployment_id}",
    }
    return {
        "model_name": MODEL_GROUP,
        "litellm_params": params,
        "model_info": {"id": deployment_id},
    }


def test_usage_based_routing_v1_selects_the_lowest_recorded_tpm() -> None:
    router: Final = Router(
        model_list=[
            _deployment(HIGH_USAGE_DEPLOYMENT_ID),
            _deployment(LOW_USAGE_DEPLOYMENT_ID),
        ],
        routing_strategy="usage-based-routing",
        num_retries=0,
    )
    usage_by_deployment: Final = {
        HIGH_USAGE_DEPLOYMENT_ID: 100,
        LOW_USAGE_DEPLOYMENT_ID: 1,
    }
    now: Final = datetime.now()
    cache_keys: Final = tuple(
        f"{MODEL_GROUP}:tpm:{(now + timedelta(minutes=offset)).strftime('%H-%M')}"
        for offset in range(60)
    )

    for cache_key in cache_keys:
        router.cache.set_cache(
            key=cache_key, value=usage_by_deployment, ttl=float("inf")
        )

    deployment: Final = router.get_available_deployment(
        model=MODEL_GROUP,
        messages=[{"role": "user", "content": "test"}],
    )

    assert deployment["model_info"]["id"] == LOW_USAGE_DEPLOYMENT_ID


@pytest.mark.asyncio
async def test_v2_async_selection_uses_prefetched_counters_only_when_they_cover_its_keys():
    router_cache = DualCache()
    router_cache.async_batch_get_cache = AsyncMock(return_value=[100, 10, None, None])  # type: ignore[method-assign]
    strategy = LowestTPMLoggingHandler_v2(router_cache=router_cache)
    deployments = [
        {"model_name": "g", "litellm_params": {"model": "m"}, "model_info": {"id": "a"}},
        {"model_name": "g", "litellm_params": {"model": "m"}, "model_info": {"id": "b"}},
    ]
    tpm_keys, rpm_keys = strategy.usage_counter_keys(deployments)
    keys = tpm_keys + rpm_keys

    covering = PrefetchedUsage(keys=frozenset(keys), values=dict(zip(keys, [10, 100, None, None])))
    with PrefetchedUsage.scoped(covering):
        chosen: Final = await strategy.async_get_available_deployments(model_group="g", healthy_deployments=deployments)
    assert chosen["model_info"]["id"] == "a", "the prefetched counters say a is the lowest"
    router_cache.async_batch_get_cache.assert_not_awaited()

    stale = PrefetchedUsage(keys=frozenset(keys[:1]), values={keys[0]: 10})
    with PrefetchedUsage.scoped(stale):
        chosen_stale: Final = await strategy.async_get_available_deployments(
            model_group="g", healthy_deployments=deployments
        )
    assert chosen_stale["model_info"]["id"] == "b", "counters that do not cover this minute's keys are read again"
    router_cache.async_batch_get_cache.assert_awaited_once_with(keys=keys)


@pytest.mark.asyncio
async def test_v2_subclass_overriding_async_get_available_deployments_with_the_old_signature_still_routes() -> None:
    class OldSignatureV2(LowestTPMLoggingHandler_v2):
        async def async_get_available_deployments(
            self,
            model_group: str,
            healthy_deployments: list,
            messages: list[dict[str, str]] | None = None,
            input: str | list | None = None,
        ):
            return await super().async_get_available_deployments(
                model_group=model_group,
                healthy_deployments=healthy_deployments,
                messages=messages,
                input=input,
            )

    router: Final = Router(
        model_list=[_deployment(HIGH_USAGE_DEPLOYMENT_ID), _deployment(LOW_USAGE_DEPLOYMENT_ID)],
        routing_strategy="usage-based-routing-v2",
    )
    router.lowesttpm_logger_v2 = OldSignatureV2(router_cache=router.cache, routing_args={})

    response: Final = await router.acompletion(
        model=MODEL_GROUP, messages=[{"role": "user", "content": "x"}]
    )

    assert response.choices[0].message.content in {
        f"from {HIGH_USAGE_DEPLOYMENT_ID}",
        f"from {LOW_USAGE_DEPLOYMENT_ID}",
    }


def _rate_limited_router(num_allowed_send: int) -> tuple[Router, tuple[list[dict[str, str]], ...]]:
    conversations: Final = tuple(
        [{"role": "user", "content": f"{index}. Hey, how's it going?"}] for index in range(num_allowed_send)
    )
    tpm: Final = sum(token_counter(model="gpt-4o", messages=messages) + 5 for messages in conversations)
    deployment: Final = _deployment(LOW_USAGE_DEPLOYMENT_ID)
    router: Final = Router(
        model_list=[{**deployment, "rpm": num_allowed_send, "tpm": tpm}],
        routing_strategy="usage-based-routing",
        enable_pre_call_checks=True,
        num_retries=0,
    )
    return router, conversations


def test_usage_based_routing_v1_serves_sync_calls_within_rpm_and_tpm() -> None:
    router, conversations = _rate_limited_router(num_allowed_send=3)
    responses: Final = [router.completion(model=MODEL_GROUP, messages=messages) for messages in conversations[:2]]
    assert [response.choices[0].message.content for response in responses] == [f"from {LOW_USAGE_DEPLOYMENT_ID}"] * 2


@pytest.mark.asyncio
async def test_usage_based_routing_v1_serves_async_calls_within_rpm_and_tpm() -> None:
    router, conversations = _rate_limited_router(num_allowed_send=3)
    responses: Final = await asyncio.gather(
        *(router.acompletion(model=MODEL_GROUP, messages=messages) for messages in conversations[:2])
    )
    assert [response.choices[0].message.content for response in responses] == [f"from {LOW_USAGE_DEPLOYMENT_ID}"] * 2


RPM_LIMIT: Final = 3


def _router_with_recorded_rpm(recorded: int, enable_pre_call_checks: bool) -> Router:
    router: Final = Router(
        model_list=[{**_deployment(LOW_USAGE_DEPLOYMENT_ID), "rpm": RPM_LIMIT}],
        routing_strategy="usage-based-routing",
        enable_pre_call_checks=enable_pre_call_checks,
        num_retries=0,
    )
    now: Final = datetime.now()
    for offset in range(-1, 2):
        router.cache.set_cache(
            key=f"{MODEL_GROUP}:rpm:{(now + timedelta(minutes=offset)).strftime('%H-%M')}",
            value={LOW_USAGE_DEPLOYMENT_ID: recorded},
            ttl=float("inf"),
        )
    return router


@pytest.mark.parametrize("enable_pre_call_checks", [True, False])
def test_usage_based_routing_v1_serves_a_deployment_below_its_recorded_rpm_limit(enable_pre_call_checks: bool) -> None:
    router: Final = _router_with_recorded_rpm(recorded=1, enable_pre_call_checks=enable_pre_call_checks)
    response: Final = router.completion(model=MODEL_GROUP, messages=[{"role": "user", "content": "hello"}])
    assert response.choices[0].message.content == f"from {LOW_USAGE_DEPLOYMENT_ID}"


@pytest.mark.parametrize("enable_pre_call_checks", [True, False])
def test_usage_based_routing_v1_rejects_a_deployment_that_reached_its_recorded_rpm_limit(
    enable_pre_call_checks: bool,
) -> None:
    router: Final = _router_with_recorded_rpm(recorded=RPM_LIMIT, enable_pre_call_checks=enable_pre_call_checks)
    with pytest.raises(ValueError, match=RouterErrors.no_deployments_available.value):
        router.completion(model=MODEL_GROUP, messages=[{"role": "user", "content": "hello"}])
