"""
The router's pre-call reads. Cooldowns go through the native store, and the tpm/rpm selector
counters (`LowestTPMLoggingHandler_v2`) are one read of the native usage store.
"""

import time
from collections.abc import Sequence
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm import Router
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore
from litellm.redis.stores.router_usage import UsageCounter
from litellm.router import router_arguments
from litellm.router_config import build_router_config
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.router_utils import routing_prefetch
from litellm.utils import get_utc_datetime
from tests.unit.caching.redis_batch_double import FakeNativePending, FakeNatives, ReadyPending, fake_pending, fake_send
from tests.unit.conftest import redis_cache_double
from tests.unit.redis.stores.router_usage_double import FakeUsageHandle, UsageHost
from tests.unit.redis.stores.test_router_cooldowns import FakeCooldowns

_MODEL_GROUP = "claude"
_MESSAGES = [{"role": "user", "content": "ping"}]


def _deployment(deployment_id: str) -> dict:
    return {
        "model_name": _MODEL_GROUP,
        "litellm_params": {"model": "anthropic/claude-x", "api_key": "test", "mock_response": "pong"},
        "model_info": {"id": deployment_id},
    }


def _router(routing_strategy: str, handle: FakeUsageHandle | None = None) -> Router:
    arguments: Final = router_arguments(
        model_list=[_deployment("dep-a"), _deployment("dep-b")], routing_strategy=routing_strategy
    )
    return Router.from_config(build_router_config(arguments), UsageHost(handle or FakeUsageHandle(redis={})))


def _counter(deployment_id: str, kind: str) -> UsageCounter:
    return UsageCounter.for_deployment(
        deployment_id,
        "anthropic/claude-x",
        kind,  # pyright: ignore[reportArgumentType]  # tpm or rpm
        get_utc_datetime().strftime("%H-%M"),
    )


def _tpm(**tokens: float | None) -> dict[UsageCounter, float]:
    return {_counter(deployment_id, "tpm"): count for deployment_id, count in tokens.items() if count is not None}


def _usage_reads(handle: FakeUsageHandle) -> list[tuple[str, ...]]:
    return [
        tuple(f"{c.deployment_id}:{c.kind}" for c in counters)  # pyright: ignore[reportAttributeAccessIssue]  # usage() args are counters
        for name, counters in handle.calls
        if name == "usage"
    ]


def _cool_down(router: Router, model_id: str, started: float) -> None:
    router.cooldown_cache.cooldown_store.put(
        model_id, {"exception_received": "429", "status_code": "429", "timestamp": started, "cooldown_time": 60}, 60
    )


@pytest.mark.asyncio
async def test_usage_based_routing_reads_the_counters_in_one_store_read():
    handle: Final = FakeUsageHandle(redis={})
    router: Final = _router("usage-based-routing-v2", handle)

    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert _usage_reads(handle) == [("dep-a:tpm", "dep-b:tpm", "dep-a:rpm", "dep-b:rpm")]


@pytest.mark.asyncio
async def test_usage_based_routing_still_batches_when_the_strategy_is_a_fixed_signature_subclass():
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

    handle: Final = FakeUsageHandle(redis={})
    router: Final = _router("usage-based-routing-v2", handle)
    router.lowesttpm_logger_v2.cancel_sync_task()
    router.lowesttpm_logger_v2 = OldSignatureV2(usage=router.lowesttpm_logger_v2.usage)

    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert _usage_reads(handle) == [("dep-a:tpm", "dep-b:tpm", "dep-a:rpm", "dep-b:rpm")], (
        "the subclassed strategy must still get the batched read, not a second one"
    )
    assert [name for name, _ in handle.calls] == ["usage"]


@pytest.mark.asyncio
async def test_simple_shuffle_reads_no_usage():
    handle: Final = FakeUsageHandle(redis={})
    router: Final = _router("simple-shuffle", handle)

    await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert handle.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tpm_a", "tpm_b", "expected"),
    [(100, 10, "dep-b"), (10, 100, "dep-a"), (None, 10, "dep-a"), (10, None, "dep-b")],
)
async def test_batched_counters_pick_the_deployment_the_strategy_picks_reading_alone(tpm_a, tpm_b, expected):
    routed: Final = _router("usage-based-routing-v2", FakeUsageHandle(redis=_tpm(**{"dep-a": tpm_a, "dep-b": tpm_b})))
    alone: Final = _router("usage-based-routing-v2", FakeUsageHandle(redis=_tpm(**{"dep-a": tpm_a, "dep-b": tpm_b})))

    routed_choice: Final = await routed.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )
    alone_choice: Final = await alone.lowesttpm_logger_v2.async_get_available_deployments(
        model_group=_MODEL_GROUP, healthy_deployments=alone.model_list, messages=_MESSAGES
    )

    assert routed_choice["model_info"]["id"] == alone_choice["model_info"]["id"] == expected


@pytest.mark.asyncio
async def test_batched_read_still_excludes_a_cooled_down_deployment():
    handle: Final = FakeUsageHandle(redis=_tpm(**{"dep-a": 100, "dep-b": 10}))
    router: Final = _router("usage-based-routing-v2", handle)
    _cool_down(router, "dep-b", started=time.time())

    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] == "dep-a", "dep-b has the lowest tpm but is cooling down"
    assert len(_usage_reads(handle)) == 1


@pytest.mark.asyncio
async def test_batched_read_ignores_an_expired_cooldown():
    router: Final = _router("usage-based-routing-v2", FakeUsageHandle(redis=_tpm(**{"dep-a": 100, "dep-b": 10})))
    _cool_down(router, "dep-b", started=time.time() - 61)

    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] == "dep-b"


@pytest.mark.asyncio
async def test_a_failed_batched_read_leaves_simple_shuffle_routing():
    redis = redis_cache_double()
    redis.async_batch_get_cache = AsyncMock(side_effect=ConnectionError("redis unavailable"))
    router = _router("simple-shuffle")
    router.update_redis_cache(cache=redis)

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}


def _armed_store() -> tuple[RouterCooldownStore, FakeCooldowns]:
    handle: Final = FakeCooldowns(has_redis=True)
    store: Final = RouterCooldownStore(handle)
    store.put("dep-a", {"exception_received": "429", "status_code": "429", "timestamp": 1.0, "cooldown_time": 60}, 60)
    return store, handle


@pytest.mark.asyncio
async def test_routing_takes_the_armed_cooldown_read_from_the_native_batch_once_then_reads_on_its_own():
    store, handle = _armed_store()
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm_cooldowns(store, ["dep-a", "dep-b", "dep-c"])
        armed: Final = routing_prefetch.armed()
        first: Final = await routing_prefetch.read_cooldowns(store, ["dep-a", "dep-b"], None)
        declared_after_first: Final = list(handle.declared)
        second: Final = await routing_prefetch.read_cooldowns(store, ["dep-a", "dep-b"], None)

    assert armed
    assert declared_after_first == [("dep-a", "dep-b", "dep-c")]
    assert handle.declared == [("dep-a", "dep-b", "dep-c"), ("dep-a", "dep-b")], "a second read goes to the store"
    assert first == second == (("dep-a", store.snapshot()["dep-a"]),)
    assert natives.built[0].events == ["flush_pre", "resolve"]


@pytest.mark.asyncio
async def test_an_armed_cooldown_read_that_does_not_cover_the_routing_ids_is_read_again():
    store, _ = _armed_store()
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm_cooldowns(store, ["dep-b"])
        active: Final = await routing_prefetch.read_cooldowns(store, ["dep-a", "dep-b"], None)

    assert active == (("dep-a", store.snapshot()["dep-a"]),)
    assert natives.built[0].events == ["flush_pre", "resolve"]


def _redis_down() -> list[tuple[str, CooldownCacheValue]]:
    raise ConnectionError("redis down")


class _UnreachableCooldowns(FakeCooldowns):
    """Redis fails the declared batch read, then answers the store's own read."""

    def active(
        self, model_ids: Sequence[str], send: object
    ) -> FakeNativePending[list[tuple[str, CooldownCacheValue]]] | ReadyPending[list[tuple[str, CooldownCacheValue]]]:
        answered: Final = super().active(model_ids, send)
        return fake_pending(send, _redis_down) if len(self.declared) == 1 else answered


@pytest.mark.asyncio
async def test_a_failed_armed_cooldown_read_falls_back_to_the_stores_own_read():
    handle: Final = _UnreachableCooldowns(has_redis=True)
    store: Final = RouterCooldownStore(handle)
    store.put("dep-a", {"exception_received": "429", "status_code": "429", "timestamp": 1.0, "cooldown_time": 60}, 60)

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm_cooldowns(store, ["dep-a"])
        active: Final = await routing_prefetch.read_cooldowns(store, ["dep-a"], None)

    assert handle.declared == [("dep-a",), ("dep-a",)], "the failed batch read, then the store's own"
    assert active == (("dep-a", store.snapshot()["dep-a"]),)


def test_arming_the_cooldown_read_outside_a_request_or_without_redis_declares_nothing():
    store, handle = _armed_store()
    offline: Final = FakeCooldowns()
    natives: Final = FakeNatives()

    routing_prefetch.arm_cooldowns(store, ["dep-a"])
    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm_cooldowns(RouterCooldownStore(offline), ["dep-a"])
        armed: Final = routing_prefetch.armed()

    assert handle.declared == []
    assert offline.declared == []
    assert not armed
    assert natives.built == []
