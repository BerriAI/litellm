#### What this tests ####
#    The request-count entry is written without a ttl, so InMemoryCache expires it
#    on default_ttl (600s). A completion landing after that expiry used to re-create
#    the counter as {id: -1} while its peers were absent, and since absent
#    deployments are seeded to 0, that deployment then won every selection and the
#    model group collapsed onto it. Issue #39322.

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler

MODEL_GROUP = "gpt-3.5-turbo"
REQUEST_COUNT_KEY = f"{MODEL_GROUP}_request_count"
DEPLOYMENT_IDS = ("1", "2", "3", "4")
HEALTHY_DEPLOYMENTS = tuple(
    {"model_info": {"id": deployment_id}, "litellm_params": {"model": "openai/gpt-4.1-mini"}}
    for deployment_id in DEPLOYMENT_IDS
)


def _kwargs(deployment_id: str) -> dict:
    return {
        "litellm_params": {
            "metadata": {"model_group": MODEL_GROUP, "deployment": "azure/gpt-4.1-mini"},
            "model_info": {"id": deployment_id},
        }
    }


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_completion_on_expired_counter_does_not_go_negative(is_async):
    cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)
    assert cache.get_cache(key=REQUEST_COUNT_KEY) is None

    if is_async:
        await handler.async_log_success_event(kwargs=_kwargs("1"), response_obj=None, start_time=0, end_time=0)
    else:
        handler.log_success_event(kwargs=_kwargs("1"), response_obj=None, start_time=0, end_time=0)

    counts = cache.get_cache(key=REQUEST_COUNT_KEY) or {}
    assert counts.get("1", 0) >= 0


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_failure_on_expired_counter_does_not_go_negative(is_async):
    cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)

    if is_async:
        await handler.async_log_failure_event(kwargs=_kwargs("1"), response_obj=None, start_time=0, end_time=0)
    else:
        handler.log_failure_event(kwargs=_kwargs("1"), response_obj=None, start_time=0, end_time=0)

    counts = cache.get_cache(key=REQUEST_COUNT_KEY) or {}
    assert counts.get("1", 0) >= 0


@pytest.mark.asyncio
async def test_expired_counter_does_not_pin_the_model_group():
    cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)
    await handler.async_log_success_event(kwargs=_kwargs("1"), response_obj=None, start_time=0, end_time=0)

    selected = {
        handler._get_available_deployments(
            healthy_deployments=list(HEALTHY_DEPLOYMENTS),
            all_deployments=dict(cache.get_cache(key=REQUEST_COUNT_KEY) or {}),
        )["model_info"]["id"]
        for _ in range(200)
    }
    assert selected == set(DEPLOYMENT_IDS)


def test_busiest_deployment_is_never_selected():
    cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)
    for deployment_id in ("1", "1", "1", "2", "2", "3"):
        handler.log_pre_api_call(model="test", messages=[], kwargs=_kwargs(deployment_id))

    selected = {
        handler._get_available_deployments(
            healthy_deployments=list(HEALTHY_DEPLOYMENTS),
            all_deployments=dict(cache.get_cache(key=REQUEST_COUNT_KEY) or {}),
        )["model_info"]["id"]
        for _ in range(200)
    }
    assert selected == {"4"}
