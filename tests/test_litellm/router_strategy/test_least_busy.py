from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.router_strategy.least_busy import IN_FLIGHT_COUNT_TTL_SECONDS, LeastBusyLoggingHandler

GROUP: Final = "least-busy-group"
DEPLOYMENT_A: Final[dict[str, object]] = {"model_info": {"id": "dep-a"}}
DEPLOYMENT_B: Final[dict[str, object]] = {"model_info": {"id": "dep-b"}}
HEALTHY: Final = [DEPLOYMENT_A, DEPLOYMENT_B]


def _call_kwargs(deployment_id: str) -> dict[str, object]:
    return {"litellm_params": {"metadata": {"model_group": GROUP}, "model_info": {"id": deployment_id}}}


class SharedRedisCounters:
    """Mirrors what Redis gives the handler: increments clamped at zero, a TTL set once when
    the key is created, and ordered reads that raise rather than invent a value."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    def count(self, key: str) -> int | None:
        return self.counts.get(key)

    def expire(self, key: str) -> None:
        self.counts.pop(key, None)
        self.ttls.pop(key, None)

    def increment_with_floor(self, key: str, value: int, ttl: int) -> int:
        incremented: Final = max(0, self.counts.get(key, 0) + value)
        self.counts[key] = incremented
        self.ttls.setdefault(key, ttl)
        return incremented

    async def async_increment_with_floor(self, key: str, value: int, ttl: int) -> int:
        return self.increment_with_floor(key, value, ttl)

    def batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        return tuple(self.counts.get(key) for key in key_list)

    async def async_batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        return self.batch_get_counts(key_list)


def _worker(shared: SharedRedisCounters | None) -> LeastBusyLoggingHandler:
    cache: Final = DualCache(in_memory_cache=InMemoryCache(), redis_cache=shared)  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    return LeastBusyLoggingHandler(router_cache=cache)


@pytest.mark.asyncio
async def test_worker_routes_around_a_request_another_worker_started() -> None:
    shared: Final = SharedRedisCounters()
    streaming_worker: Final = _worker(shared)
    picking_worker: Final = _worker(shared)

    picking_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    await picking_worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    streaming_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await picking_worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await streaming_worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert await picking_worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_sync_pick_reads_the_shared_counts() -> None:
    shared: Final = SharedRedisCounters()
    streaming_worker: Final = _worker(shared)
    picking_worker: Final = _worker(shared)

    picking_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    picking_worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    streaming_worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    streaming_worker.log_failure_event(_call_kwargs("dep-a"), None, None, None)

    assert picking_worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_the_handler_never_pushes_a_counters_ttl_forward() -> None:
    """A worker that dies mid-request leaves a +1 nobody will ever decrement. Redis expires that
    stuck count an hour after the key was created, which only works while nothing writes the TTL
    again: a handler that refreshed it on every touch would keep the count alive for as long as
    the group takes traffic, and the deployment would read busier than it is forever."""
    shared: Final = SharedRedisCounters()
    worker: Final = _worker(shared)
    key: Final = f"{GROUP}_request_count:dep-a"

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert shared.ttls == {key: IN_FLIGHT_COUNT_TTL_SECONDS}

    shared.ttls[key] = 5
    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert shared.count(key) == 1
    assert shared.ttls == {key: 5}


@pytest.mark.asyncio
async def test_counts_stay_in_memory_without_redis() -> None:
    worker: Final = _worker(None)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B
    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A
    assert worker.router_cache.get_cache(f"{GROUP}_request_count:dep-a") == 0


class UnavailableRedis(SharedRedisCounters):
    def increment_with_floor(self, key: str, value: int, ttl: int) -> int:
        raise ConnectionError("redis is down")

    def batch_get_counts(self, key_list: list[str]) -> tuple[int | None, ...]:
        raise ConnectionError("redis is down")


@pytest.mark.asyncio
async def test_a_redis_outage_falls_back_to_this_workers_own_counts() -> None:
    worker: Final = _worker(UnavailableRedis())

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B
    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B

    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_A


def test_a_shared_counter_that_expired_mid_request_cannot_go_negative() -> None:
    shared: Final = SharedRedisCounters()
    worker: Final = _worker(shared)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    shared.expire(f"{GROUP}_request_count:dep-a")
    worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert shared.count(f"{GROUP}_request_count:dep-a") == 0

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert shared.count(f"{GROUP}_request_count:dep-a") == 1
    assert worker.get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


@pytest.mark.asyncio
async def test_a_local_counter_that_expired_mid_request_cannot_go_negative() -> None:
    worker: Final = _worker(None)
    in_memory: Final = worker.router_cache.in_memory_cache

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    in_memory.delete_cache(f"{GROUP}_request_count:dep-a")
    await worker.async_log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert worker.router_cache.get_cache(f"{GROUP}_request_count:dep-a") == 0

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

    assert await worker.async_get_available_deployments(GROUP, HEALTHY) is DEPLOYMENT_B


def test_calls_without_a_deployment_are_ignored() -> None:
    shared: Final = SharedRedisCounters()
    worker: Final = _worker(shared)

    worker.log_pre_api_call(model="m", messages=[], kwargs={"litellm_params": {"metadata": None}})
    worker.log_pre_api_call(model="m", messages=[], kwargs={})

    assert shared.counts == {}
