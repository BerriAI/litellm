import json
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
    """Stores JSON strings and hands back a fresh object per read, the way a real Redis client does."""

    def __init__(self) -> None:
        self.encoded: dict[str, str] = {}
        self.ttls: dict[str, float] = {}

    def count(self, key: str) -> object:
        raw: Final = self.encoded.get(key)
        return None if raw is None else json.loads(raw)

    def get_cache(self, key: str, **kwargs: object) -> object:
        return self.count(key)

    def set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.encoded[key] = json.dumps(value)

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        return self.count(key)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.set_cache(key, value)

    def increment_cache(self, key: str, value: int, ttl: float | None = None, refresh_ttl: bool = False) -> int:
        current: Final = self.count(key) or 0
        assert isinstance(current, int)
        incremented: Final = current + value
        self.encoded[key] = json.dumps(incremented)
        if ttl is not None and (refresh_ttl or key not in self.ttls):
            self.ttls[key] = ttl
        return incremented

    async def async_increment(self, key: str, value: float, ttl: int | None = None, refresh_ttl: bool = False) -> float:
        return self.increment_cache(key, int(value), ttl, refresh_ttl)

    def batch_get_cache(self, key_list: list[str], **kwargs: object) -> dict[str, object]:
        return {key: self.count(key) for key in key_list}

    async def async_batch_get_cache(self, key_list: list[str], **kwargs: object) -> dict[str, object]:
        return self.batch_get_cache(key_list)


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


def test_redis_counts_keep_a_refreshed_ttl() -> None:
    shared: Final = SharedRedisCounters()
    worker: Final = _worker(shared)

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))
    worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert shared.count(f"{GROUP}_request_count:dep-a") == 0
    assert shared.ttls == {f"{GROUP}_request_count:dep-a": IN_FLIGHT_COUNT_TTL_SECONDS}


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
    def batch_get_cache(self, key_list: list[str], **kwargs: object) -> dict[str, object]:
        raise ConnectionError("redis is down")

    def increment_cache(self, key: str, value: int, ttl: float | None = None, refresh_ttl: bool = False) -> int:
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
    shared.encoded.clear()
    worker.log_success_event(_call_kwargs("dep-a"), None, None, None)

    assert shared.count(f"{GROUP}_request_count:dep-a") == 0

    worker.log_pre_api_call(model="m", messages=[], kwargs=_call_kwargs("dep-a"))

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

    assert shared.encoded == {}
