import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.router_utils.deployment_usage_counter import (
    DeploymentUsageCounter,
    charged_rpm,
    end_attempt,
    enforcement_tpm_operation,
    reads_deployment_usage,
    usage_twins,
)
from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck

NOON = datetime(2026, 10, 28, 12, 0, 30, tzinfo=timezone.utc)
DEPLOYMENT = {"litellm_params": {"model": "openai/gpt-4o"}, "model_info": {"id": "deploy-1"}}
RPM_KEY = "deploy-1:openai/gpt-4o:rpm:12-00"


class _Clock:
    def __init__(self) -> None:
        self.now = NOON

    def __call__(self) -> datetime:
        return self.now


def _counter(cache: DualCache, clock: _Clock | None = None) -> DeploymentUsageCounter:
    return DeploymentUsageCounter(cache, clock=clock or _Clock())


@pytest.mark.asyncio
async def test_each_charged_request_adds_one_to_the_minute():
    cache = DualCache()
    counter = _counter(cache)

    charges = [await counter.charge_request(DEPLOYMENT) for _ in range(3)]

    assert [charge.count for charge in charges] == [1, 2, 3]
    assert {charge.key for charge in charges} == {RPM_KEY}
    assert await cache.async_get_cache(RPM_KEY) == 3


@pytest.mark.asyncio
async def test_a_refunded_request_leaves_the_minute_as_it_was():
    cache = DualCache()
    counter = _counter(cache)
    await counter.charge_request(DEPLOYMENT)
    second = await counter.charge_request(DEPLOYMENT)

    await counter.refund_request(second)

    assert await cache.async_get_cache(RPM_KEY) == 1


@pytest.mark.asyncio
async def test_a_refund_after_the_minute_rolled_over_goes_back_to_the_charged_minute():
    cache = DualCache()
    clock = _Clock()
    counter = _counter(cache, clock)
    charge = await counter.charge_request(DEPLOYMENT)
    clock.now = NOON + timedelta(minutes=1)

    await counter.refund_request(charge)

    assert await cache.async_get_cache(RPM_KEY) == 0
    assert await cache.async_get_cache("deploy-1:openai/gpt-4o:rpm:12-01") is None


def test_sync_charges_and_refunds_count_like_the_async_ones():
    cache = DualCache()
    counter = _counter(cache)

    charges = [counter.charge_request_sync(DEPLOYMENT) for _ in range(2)]
    counter.refund_request_sync(charges[1])

    assert [charge.count for charge in charges] == [1, 2]
    assert cache.get_cache(RPM_KEY) == 1


@pytest.mark.asyncio
async def test_a_check_reads_the_count_its_own_charge_was_admitted_with():
    cache = DualCache()
    counter = _counter(cache)

    async def attempt() -> int | None:
        charge = await counter.charge_request(DEPLOYMENT)
        await asyncio.sleep(0)
        seen = charged_rpm(RPM_KEY)
        end_attempt(charge)
        return seen

    assert sorted(await asyncio.gather(attempt(), attempt(), attempt())) == [1, 2, 3]
    assert charged_rpm(RPM_KEY) is None


@pytest.mark.asyncio
async def test_a_charge_answers_only_for_its_own_key():
    counter = _counter(DualCache())

    charge = await counter.charge_request(DEPLOYMENT)

    assert charged_rpm(RPM_KEY) == 1
    assert charged_rpm("deploy-2:openai/gpt-4o:rpm:12-00") is None
    end_attempt(charge)


def test_the_router_usage_tpm_write_has_an_enforcement_twin_and_the_rpm_write_has_none():
    tpm = RedisPipelineIncrementOperation(key="global_router:deploy-1:openai/gpt-4o:tpm:12-00", increment_value=40, ttl=60)
    rpm = RedisPipelineIncrementOperation(key="global_router:deploy-1:openai/gpt-4o:rpm:12-00", increment_value=1, ttl=60)

    assert enforcement_tpm_operation(tpm) == RedisPipelineIncrementOperation(
        key="deploy-1:openai/gpt-4o:tpm:12-00", increment_value=40, ttl=60
    )
    assert enforcement_tpm_operation(rpm) is None
    assert usage_twins((tpm, rpm), counted=True) == (enforcement_tpm_operation(tpm),)
    assert usage_twins((tpm, rpm), counted=False) == ()


def test_only_a_callback_that_says_so_reads_the_counters():
    assert reads_deployment_usage((object(), ModelRateLimitingCheck(dual_cache=DualCache())))
    assert not reads_deployment_usage((object(), DeploymentUsageCounter(DualCache())))
