from datetime import datetime, timezone

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.deployment_usage_counter import DeploymentUsageCounter, reads_deployment_usage
from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck

NOON = datetime(2026, 10, 28, 12, 0, 30, tzinfo=timezone.utc)
DEPLOYMENT = {"litellm_params": {"model": "openai/gpt-4o"}, "model_info": {"id": "deploy-1"}}
RPM_KEY = "deploy-1:openai/gpt-4o:rpm:12-00"
TPM_KEY = "deploy-1:openai/gpt-4o:tpm:12-00"
SUCCESS = {
    "standard_logging_object": {
        "model_id": "deploy-1",
        "total_tokens": 40,
        "hidden_params": {"litellm_model_name": "openai/gpt-4o"},
    }
}


def _counter(cache: DualCache, charges: bool = True) -> DeploymentUsageCounter:
    return DeploymentUsageCounter(cache, charges=lambda: charges, clock=lambda: NOON)


@pytest.mark.asyncio
async def test_each_charged_request_adds_one_to_the_minute():
    cache = DualCache()
    counter = _counter(cache)

    counts = [await counter.charge_request(DEPLOYMENT) for _ in range(3)]

    assert counts == [1, 2, 3]
    assert await cache.async_get_cache(RPM_KEY) == 3


@pytest.mark.asyncio
async def test_a_refunded_request_leaves_the_minute_as_it_was():
    cache = DualCache()
    counter = _counter(cache)
    await counter.charge_request(DEPLOYMENT)
    await counter.charge_request(DEPLOYMENT)

    await counter.refund_request(DEPLOYMENT)

    assert await cache.async_get_cache(RPM_KEY) == 1


def test_sync_charges_and_refunds_count_like_the_async_ones():
    cache = DualCache()
    counter = _counter(cache)

    assert [counter.charge_request_sync(DEPLOYMENT) for _ in range(2)] == [1, 2]
    counter.refund_request_sync(DEPLOYMENT)

    assert cache.get_cache(RPM_KEY) == 1


@pytest.mark.asyncio
async def test_a_logged_success_adds_its_tokens_to_the_minute_once():
    cache = DualCache()
    counter = _counter(cache)

    await counter.async_log_success_event(SUCCESS, None, None, None)
    counter.log_success_event(SUCCESS, None, None, None)

    assert await cache.async_get_cache(TPM_KEY) == 80


@pytest.mark.asyncio
async def test_a_logged_success_counts_nothing_when_no_callback_reads_the_counters():
    cache = DualCache()
    counter = _counter(cache, charges=False)

    await counter.async_log_success_event(SUCCESS, None, None, None)
    counter.log_success_event(SUCCESS, None, None, None)

    assert await cache.async_get_cache(TPM_KEY) is None


@pytest.mark.asyncio
async def test_a_batch_retrieve_is_not_usage():
    cache = DualCache()

    await _counter(cache).async_log_success_event({**SUCCESS, "call_type": "aretrieve_batch"}, None, None, None)

    assert await cache.async_get_cache(TPM_KEY) is None


def test_only_the_rate_limit_check_and_usage_based_v2_read_the_counters():
    assert reads_deployment_usage((object(), ModelRateLimitingCheck(dual_cache=DualCache())))
    assert not reads_deployment_usage((object(), DeploymentUsageCounter(DualCache())))
