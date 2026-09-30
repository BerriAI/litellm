from datetime import datetime, timedelta
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict

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
    chosen = await strategy.async_get_available_deployments(model_group="g", healthy_deployments=deployments, prefetched_usage=covering)
    assert chosen["model_info"]["id"] == "a", "the prefetched counters say a is the lowest"
    router_cache.async_batch_get_cache.assert_not_awaited()

    stale = PrefetchedUsage(keys=frozenset(keys[:1]), values={keys[0]: 10})
    chosen = await strategy.async_get_available_deployments(model_group="g", healthy_deployments=deployments, prefetched_usage=stale)
    assert chosen["model_info"]["id"] == "b", "counters that do not cover this minute's keys are read again"
    router_cache.async_batch_get_cache.assert_awaited_once_with(keys=keys)
