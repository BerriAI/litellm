import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, TypeVar

import pytest

from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.router_usage import NativeCounter, RouterUsageStore, UsageCounter

T = TypeVar("T")


def _done(value: T) -> "asyncio.Future[T]":
    future: Final[asyncio.Future[T]] = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


@dataclass
class FakeUsage:
    connected: bool
    counts: dict[NativeCounter, float] = field(default_factory=dict)
    calls: list[tuple[str, object, str | None]] = field(default_factory=list)

    def increment(self, counter: NativeCounter, value: float, ttl_seconds: int) -> "asyncio.Future[float]":
        self.calls.append(("increment", (counter, value, ttl_seconds), current_service_target()))
        self.counts[counter] = self.counts.get(counter, 0) + value
        return _done(self.counts[counter])

    def increment_deployment(
        self, deployment_id: str, model: str | None, minute: str, tokens: float, requests: int
    ) -> "asyncio.Future[None]":
        self.calls.append(("deployment", (deployment_id, model, minute, tokens, requests), current_service_target()))
        for kind, value in (("tpm", tokens), ("rpm", requests)):
            counter: NativeCounter = (deployment_id, model, minute, kind)  # pyright: ignore[reportAssignmentType]  # literal kinds
            self.counts[counter] = self.counts.get(counter, 0) + value
        return _done(None)

    def get_many(self, counters: Sequence[NativeCounter]) -> "asyncio.Future[list[float | None]]":
        return _done([self.counts.get(counter) for counter in counters])

    def flush_local_counters(self) -> None:
        self.counts.clear()

    def snapshot(self) -> dict[str, float]:
        return {}


@dataclass
class Handles:
    built: list[FakeUsage] = field(default_factory=list)

    def __call__(self, connection: RedisCache | None) -> FakeUsage:
        self.built.append(FakeUsage(connection is not None))
        return self.built[-1]


@pytest.mark.asyncio
async def test_counters_cross_the_bridge_as_domain_tuples_under_the_router_usage_target() -> None:
    handles: Final = Handles()
    store: Final = RouterUsageStore(None, handle=handles)
    counter: Final = UsageCounter("dep", None, "12-34", "rpm")

    await store.async_increment(counter, 1, 60)
    await store.increment_deployment("dep", "model", "12-34", 7.5, 1, None)

    assert handles.built[0].calls == [
        ("increment", (("dep", None, "12-34", "rpm"), 1, 60), "router_usage"),
        ("deployment", ("dep", "model", "12-34", 7.5, 1), "router_usage"),
    ]
    assert await store.async_batch_get((counter, UsageCounter("dep", "model", "12-34", "tpm"))) == [1, 7.5]


def test_late_attach_rebuilds_the_handle_once() -> None:
    handles: Final = Handles()
    store: Final = RouterUsageStore(None, handle=handles)
    store.attach_redis(RedisCache(host="redis.internal"))
    store.attach_redis(RedisCache(host="other"))
    assert [handle.connected for handle in handles.built] == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("flush_cache", "reset"))
async def test_router_reset_clears_usage_and_failure_counters(operation: str) -> None:
    router: Final = Router(model_list=[])
    usage: Final = router._core_usage_store
    cooldowns: Final = router._core_cooldown_cache.cooldown_store
    counters: Final = (UsageCounter("dep", "model", "12-34", "tpm"), UsageCounter("dep", "model", "12-34", "rpm"))
    await usage.increment_deployment("dep", "model", "12-34", 7, 1, None)
    assert await usage.async_batch_get(counters) == [7, 1]
    assert cooldowns.increment_allowed_fails("dep", None, 30.0) == 1
    getattr(router, operation)()
    assert await usage.async_batch_get(counters) == [None, None]
    assert cooldowns.increment_allowed_fails("dep", None, 30.0) == 1
