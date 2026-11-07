"""
The router's pre-call reads. Cooldowns go through the native store, and the tpm/rpm selector
counters (`LowestTPMLoggingHandler_v2`) ride one MGET on the Python Redis.
"""

import time
from collections.abc import Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm import Router
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.router_utils import routing_prefetch
from tests.unit.caching.test_redis_batch import FakeNativePending, FakeNatives, ReadyPending, fake_pending, fake_send
from tests.unit.conftest import redis_cache_double
from tests.unit.redis.stores.test_router_cooldowns import FakeCooldowns

_MODEL_GROUP = "claude"
_MESSAGES = [{"role": "user", "content": "ping"}]


def _deployment(deployment_id: str) -> dict:
    return {
        "model_name": _MODEL_GROUP,
        "litellm_params": {"model": "anthropic/claude-x", "api_key": "test", "mock_response": "pong"},
        "model_info": {"id": deployment_id},
    }


def _redis_answering(values_by_key_prefix: dict[str, object]) -> MagicMock:
    """A Redis double that answers each key from its minute-less prefix and records every MGET."""

    def _mget(key_list, parent_otel_span=None):
        return {key: values_by_key_prefix.get(key.rsplit(":", 1)[0], values_by_key_prefix.get(key)) for key in key_list}

    redis = redis_cache_double()
    redis.async_batch_get_cache = AsyncMock(side_effect=_mget)
    return redis


def _router(redis: MagicMock, routing_strategy: str) -> Router:
    router = Router(
        model_list=[_deployment("dep-a"), _deployment("dep-b")],
        routing_strategy=routing_strategy,
    )
    router.update_redis_cache(cache=redis)
    return router


def _redis_key_families(redis: MagicMock) -> list[list[str]]:
    return [
        sorted(key.rsplit(":", 1)[0] if ":tpm:" in key or ":rpm:" in key else key for key in call.args[0])
        for call in redis.async_batch_get_cache.await_args_list
    ]


def _cool_down(router: Router, model_id: str, started: float) -> None:
    router.cooldown_cache.cooldown_store.put(
        model_id, {"exception_received": "429", "status_code": "429", "timestamp": started, "cooldown_time": 60}, 60
    )


@pytest.mark.asyncio
async def test_usage_based_routing_reads_the_counters_in_one_python_redis_round_trip():
    redis = _redis_answering({})
    router = _router(redis, "usage-based-routing-v2")

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert _redis_key_families(redis) == [
        [
            "dep-a:anthropic/claude-x:rpm",
            "dep-a:anthropic/claude-x:tpm",
            "dep-b:anthropic/claude-x:rpm",
            "dep-b:anthropic/claude-x:tpm",
        ]
    ], "the usage counters must arrive in one MGET, without cooldown keys"


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

    redis: Final = _redis_answering({})
    router: Final = _router(redis, "usage-based-routing-v2")
    router.lowesttpm_logger_v2 = OldSignatureV2(router_cache=router.cache)
    router.cache.async_batch_get_cache = AsyncMock(wraps=router.cache.async_batch_get_cache)

    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert _redis_key_families(redis) == [
        [
            "dep-a:anthropic/claude-x:rpm",
            "dep-a:anthropic/claude-x:tpm",
            "dep-b:anthropic/claude-x:rpm",
            "dep-b:anthropic/claude-x:tpm",
        ]
    ], "the subclassed strategy must still get the batched read, not a second MGET"
    router.cache.async_batch_get_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_simple_shuffle_reads_nothing_from_the_python_redis():
    redis = _redis_answering({})
    router = _router(redis, "simple-shuffle")

    await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert _redis_key_families(redis) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tpm_a", "tpm_b", "expected"),
    [(100, 10, "dep-b"), (10, 100, "dep-a"), (None, 10, "dep-a"), (10, None, "dep-b")],
)
async def test_batched_counters_pick_the_deployment_the_strategy_picks_reading_alone(tpm_a, tpm_b, expected):
    counters = {"dep-a:anthropic/claude-x:tpm": tpm_a, "dep-b:anthropic/claude-x:tpm": tpm_b}
    routed = _router(_redis_answering(counters), "usage-based-routing-v2")
    alone = _router(_redis_answering(counters), "usage-based-routing-v2")

    routed_choice = await routed.async_get_available_deployment(
        model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES
    )
    alone_choice = await alone.lowesttpm_logger_v2.async_get_available_deployments(
        model_group=_MODEL_GROUP, healthy_deployments=alone.model_list, messages=_MESSAGES
    )

    assert routed_choice["model_info"]["id"] == alone_choice["model_info"]["id"] == expected


@pytest.mark.asyncio
async def test_batched_read_still_excludes_a_cooled_down_deployment():
    redis = _redis_answering({"dep-a:anthropic/claude-x:tpm": 100, "dep-b:anthropic/claude-x:tpm": 10})
    router = _router(redis, "usage-based-routing-v2")
    _cool_down(router, "dep-b", started=time.time())

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] == "dep-a", "dep-b has the lowest tpm but is cooling down"
    assert redis.async_batch_get_cache.await_count == 1


@pytest.mark.asyncio
async def test_batched_read_ignores_an_expired_cooldown():
    redis = _redis_answering({"dep-a:anthropic/claude-x:tpm": 100, "dep-b:anthropic/claude-x:tpm": 10})
    router = _router(redis, "usage-based-routing-v2")
    _cool_down(router, "dep-b", started=time.time() - 61)

    deployment = await router.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)

    assert deployment["model_info"]["id"] == "dep-b"


@pytest.mark.asyncio
async def test_a_failed_batched_read_degrades_like_the_two_failed_reads_did():
    redis = redis_cache_double()
    redis.async_batch_get_cache = AsyncMock(side_effect=ConnectionError("redis unavailable"))
    routed = _router(redis, "usage-based-routing-v2")
    alone = _router(redis, "usage-based-routing-v2")

    with pytest.raises(litellm.RateLimitError, match="No deployments available") as routed_error:
        await routed.async_get_available_deployment(model=_MODEL_GROUP, request_kwargs={}, messages=_MESSAGES)
    with pytest.raises(litellm.RateLimitError, match="No deployments available") as alone_error:
        await alone.lowesttpm_logger_v2.async_get_available_deployments(
            model_group=_MODEL_GROUP, healthy_deployments=alone.model_list, messages=_MESSAGES
        )

    assert str(routed_error.value) == str(alone_error.value)
    assert len(routed.cache.last_redis_batch_access_time) == 0, "a failed read must not throttle the next one"


@pytest.mark.asyncio
async def test_a_failed_batched_read_leaves_simple_shuffle_routing():
    redis = redis_cache_double()
    redis.async_batch_get_cache = AsyncMock(side_effect=ConnectionError("redis unavailable"))
    router = _router(redis, "simple-shuffle")

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
