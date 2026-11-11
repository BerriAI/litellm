import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, TypeVar

import pytest

from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.redis_batch import drain_post_call_redis_batches, request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.connection import connect
from litellm.redis.stores.router_usage import RouterUsageStore, UsageCounter
from litellm.rust_bridge._native import NativeRedisClient
from litellm.types.utils import ModelResponse
from tests.unit.caching.test_redis_batch import FakeNatives, fake_send

T = TypeVar("T")


def _done(value: T) -> "asyncio.Future[T]":
    future: Final[asyncio.Future[T]] = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


@dataclass
class FakeUsage:
    has_redis: bool = False
    counts: dict[UsageCounter, float] = field(default_factory=dict)
    calls: list[tuple[str, object, str | None]] = field(default_factory=list)

    def attach(self, client: NativeRedisClient, /) -> bool:
        attached: Final = not self.has_redis
        self.has_redis = True
        return attached

    def increment_sync(self, counter: UsageCounter, value: float, ttl_seconds: int) -> float:
        self.calls.append(("increment", (counter, value, ttl_seconds), current_service_target()))
        self.counts[counter] = self.counts.get(counter, 0) + value
        return self.counts[counter]

    def increment(self, counter: UsageCounter, value: float, ttl_seconds: int) -> "asyncio.Future[float]":
        return _done(self.increment_sync(counter, value, ttl_seconds))

    def increment_deployment_sync(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None:
        self.calls.append(("deployment", (deployment_id, model, minute, tokens, requests), current_service_target()))
        tpm: Final = UsageCounter(deployment_id, model, minute, "tpm")
        rpm: Final = UsageCounter(deployment_id, model, minute, "rpm")
        self.counts[tpm] = self.counts.get(tpm, 0) + tokens
        self.counts[rpm] = self.counts.get(rpm, 0) + requests

    def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> "asyncio.Future[None]":
        self.increment_deployment_sync(deployment_id, model, minute, tokens, requests)
        return _done(None)

    def get_many_sync(self, counters: Sequence[UsageCounter]) -> list[float | None]:
        return [self.counts.get(counter) for counter in counters]

    def get_many(self, counters: Sequence[UsageCounter]) -> "asyncio.Future[list[float | None]]":
        return _done(self.get_many_sync(counters))

    def flush_local_counters(self) -> None:
        self.counts.clear()

    def snapshot(self) -> dict[str, float]:
        return {}


@pytest.mark.asyncio
async def test_counters_cross_the_bridge_as_domain_tuples_under_the_router_usage_target() -> None:
    handle: Final = FakeUsage()
    store: Final = RouterUsageStore(handle)
    counter: Final = UsageCounter("dep", None, "12-34", "rpm")

    await store.increment(counter, 1, 60)
    await store.increment_deployment("dep", "model", "12-34", 7.5, 1)

    assert handle.calls == [
        ("increment", (counter, 1, 60), "router_usage"),
        ("deployment", ("dep", "model", "12-34", 7.5, 1), "router_usage"),
    ]
    assert await store.get_many((counter, UsageCounter("dep", "model", "12-34", "tpm"))) == [1, 7.5]


@pytest.mark.asyncio
async def test_late_attach_keeps_the_counts_this_router_already_holds() -> None:
    store: Final = RouterUsageStore(connect(None).router_usage())
    counters: Final = (UsageCounter("dep", "model", "12-34", "tpm"), UsageCounter("dep", "model", "12-34", "rpm"))
    await store.increment_deployment("dep", "model", "12-34", 7.5, 1)
    store.attach_redis(connect(RedisCache(host="redis.internal")))
    assert await store.get_many(counters) == [7.5, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("flush_cache", "reset"))
async def test_router_reset_clears_usage_and_failure_counters(operation: str) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "g",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test", "rpm": 10},
                "model_info": {"id": "dep"},
            }
        ]
    )
    cooldowns: Final = router.health().cooldowns.cooldown_store
    served: Final = ModelResponse(usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
    metadata: Final = {"model_info": {"id": "dep"}, "deployment": "openai/gpt-4o", "model_group": "g"}
    await router.increment_deployment_usage_for_response(served, {"model": "g", "metadata": metadata})
    assert await router.health().usage("g") == (7, 1)
    assert cooldowns.increment_allowed_fails("dep", None, 30.0) == 1
    getattr(router, operation)()
    assert await router.health().usage("g") == (None, None)
    assert cooldowns.increment_allowed_fails("dep", None, 30.0) == 1


@pytest.mark.asyncio
async def test_inside_a_request_deployment_increments_go_out_on_their_own_trip_like_main():
    handle: Final = FakeUsage()
    store: Final = RouterUsageStore(handle)
    natives: Final = FakeNatives()

    await store.increment_deployment("dep", "model", "12-34", 7.5, 1)
    with request_redis_batch_scope(post_call_deadline=60, native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        await store.increment_deployment("dep", "model", "12-34", 2, 1)
    await drain_post_call_redis_batches()

    assert [call[0] for call in handle.calls] == ["deployment", "deployment"]
    assert all(not native.events for native in natives.built)
