"""One Redis pipeline per backend for the pre-call reads a request makes: rate limiter Lua groups, the
router's usage counter read, auth identity and spend counters all join the request batch."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final, Literal
from unittest.mock import MagicMock, patch

import pytest

from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import Batch, active_request_redis_batches, request_redis_batch_scope
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import cache_team_object
from litellm.proxy.auth.auth_object_prefetch import _CacheEntry, _write_back, prefetch_identity_keys
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores._route import route
from litellm.redis.stores.router_cooldowns import ActiveCooldowns, RouterCooldownStore
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, RouterUsageStore, UsageCounter
from litellm.router_api import RoutingContext
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage
from litellm.router_utils import routing_prefetch
from litellm.router_utils.cooldown_cache import CooldownCache
from litellm.router_utils.routing_prefetch import arm_cooldowns, read_cooldowns
from litellm.router_utils.routing_read_batch import RoutingReadBatch
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache
from tests.unit.redis.stores.router_usage_double import FakeUsageRead
from tests.unit.redis.stores.test_router_cooldowns import FakeCooldowns

from .redis_batch_double import (
    FakeNativePending,
    FakeNativeRequest,
    FakeNatives,
    FakeRedisCache,
    fake_pending,
    fake_send,
)


def _cooldown_key(model_id: str) -> str:
    return f"deployment:{model_id}:cooldown"


_MODEL_GROUP = "claude"
_FAR_FUTURE = 4_102_444_800.0  # 2100-01-01, a cooldown stamped then is still active


def _declared_before_the_first_flush(natives: FakeNatives) -> list[str]:
    events: Final = natives.built[0].events
    return events[: events.index("flush_pre")]


@dataclass
class _FakeCooldownsHandle:
    """A connected cooldown store whose ``active`` lands on the request's fake native batch."""

    has_redis: Final = True

    def active(self, model_ids: Sequence[str], send: FakeNativeRequest) -> FakeNativePending[ActiveCooldowns]:
        send.events.append("declare cooldowns")
        return FakeNativePending(send, lambda: ())


def _deployment(deployment_id: str) -> dict:
    return {
        "model_name": _MODEL_GROUP,
        "litellm_params": {"model": "anthropic/claude-x", "api_key": "test", "mock_response": "pong"},
        "model_info": {"id": deployment_id},
    }


def _router(redis_cache: FakeRedisCache, routing_strategy: str = "usage-based-routing-v2") -> Router:
    router = Router(model_list=[_deployment("dep-a"), _deployment("dep-b")], routing_strategy=routing_strategy)
    router.update_redis_cache(cache=redis_cache)
    return router


_PING: Final = [{"role": "user", "content": "ping"}]
_PINNED_MINUTE: Final = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)


def _python_cooldown_reads(redis_cache: FakeRedisCache) -> tuple[str, ...]:
    return tuple(
        key for command, keys in redis_cache.alone if command == "MGET" for key in keys if key.endswith(":cooldown")
    )


async def _route(router: Router) -> str:
    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, messages=_PING, request_kwargs={}
    )
    return deployment["model_info"]["id"]


def _counter(deployment_id: str, kind: Literal["tpm", "rpm"]) -> UsageCounter:
    return UsageCounter.for_deployment(deployment_id, "anthropic/claude-x", kind, "12-00")


_DEPLOYMENT_COUNTERS: Final = tuple(_counter(d, kind) for kind in ("tpm", "rpm") for d in ("dep-a", "dep-b"))


@dataclass
class _FakeUsageHandle:
    """A connected usage store whose ``usage`` reads land on the request's fake native batch, or alone for
    ``NativeSend.Now()``, recording what each read asked for and under which service target."""

    counts: dict[UsageCounter, float]
    fail: Exception | None = None
    reads: list[tuple[str, tuple[UsageCounter, ...], str | None]] = field(default_factory=list)
    has_redis: bool = True

    def usage(self, counters: Sequence[UsageCounter], send: object) -> FakeUsageRead:
        batched: Final = isinstance(send, FakeNativeRequest)
        self.reads.append(("batch" if batched else "alone", tuple(counters), current_service_target()))
        if batched:
            send.events.append("declare usage")

        fail: Final = self.fail

        def answer() -> list[float | None]:
            if fail is not None:
                raise fail
            return [self.counts.get(counter) for counter in counters]

        return FakeUsageRead(
            fake_pending(send, answer),
            lambda: self.reads.append(("release", tuple(counters), current_service_target())),
        )

    def get_many_sync(self, counters: Sequence[UsageCounter]) -> list[float | None]:
        raise AssertionError("a connected store reads through usage()")


def _usage_store(handle: _FakeUsageHandle) -> RouterUsageStore:
    return RouterUsageStore(handle)  # pyright: ignore[reportArgumentType]  # test double


@pytest.mark.asyncio
async def test_the_armed_usage_read_and_the_armed_cooldown_read_go_out_in_one_native_pipeline():
    handle: Final = _FakeUsageHandle({_counter("dep-a", "tpm"): 40})
    usage: Final = _usage_store(handle)
    cooldowns: Final = RouterCooldownStore(_FakeCooldownsHandle())
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        arm_cooldowns(cooldowns, ["dep-a", "dep-b"])
        routing_prefetch.arm(usage, _DEPLOYMENT_COUNTERS)
        active, counts = await asyncio.gather(
            read_cooldowns(cooldowns, ["dep-a", "dep-b"], None),
            routing_prefetch.read(usage, _DEPLOYMENT_COUNTERS, None),
        )

    assert active == ()
    assert counts == [40, None, None, None]
    assert _declared_before_the_first_flush(natives) == ["declare cooldowns", "declare usage"]
    assert [kind for kind, _, _ in handle.reads] == ["batch"]
    assert request.prefetched == {}


@pytest.mark.asyncio
async def test_the_armed_usage_read_is_declared_under_the_router_usage_family():
    handle: Final = _FakeUsageHandle({})

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm(_usage_store(handle), _DEPLOYMENT_COUNTERS)

    assert [target for _, _, target in handle.reads] == [ROUTER_USAGE_TARGET]


@pytest.mark.asyncio
async def test_a_usage_prefetch_that_does_not_cover_the_pick_gives_its_reservation_back_before_the_store_reads():
    handle: Final = _FakeUsageHandle({_counter("dep-b", "rpm"): 3})
    usage: Final = _usage_store(handle)

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm(usage, _DEPLOYMENT_COUNTERS[:1])
        counts: Final = await routing_prefetch.read(usage, _DEPLOYMENT_COUNTERS, None)
        assert request.prefetched == {}

    assert counts == [None, None, None, 3]
    assert [(kind, len(counters)) for kind, counters, _ in handle.reads] == [("batch", 1), ("release", 1), ("alone", 4)]


@pytest.mark.asyncio
async def test_a_failed_usage_prefetch_falls_back_to_the_stores_own_read():
    handle: Final = _FakeUsageHandle({_counter("dep-a", "tpm"): 9}, fail=ConnectionError("redis down"))
    usage: Final = _usage_store(handle)

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm(usage, _DEPLOYMENT_COUNTERS)
        handle.fail = None
        counts: Final = await routing_prefetch.read(usage, _DEPLOYMENT_COUNTERS, None)

    assert counts == [9, None, None, None]
    assert [kind for kind, _, _ in handle.reads] == ["batch", "alone"]


@pytest.mark.asyncio
async def test_arming_a_store_without_redis_declares_nothing():
    handle: Final = _FakeUsageHandle({}, has_redis=False)

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        routing_prefetch.arm(_usage_store(handle), _DEPLOYMENT_COUNTERS)

        assert request.prefetched == {}
    assert handle.reads == []


@pytest.mark.asyncio
async def test_a_usage_based_pick_reads_the_counters_the_prefetch_armed():
    handle: Final = _FakeUsageHandle({_counter("dep-a", "tpm"): 100})
    selector: Final = LowestTPMLoggingHandler_v2(usage=_usage_store(handle))
    deployments: Final[list[object]] = [_deployment("dep-a"), _deployment("dep-b")]
    read_batch: Final = RoutingReadBatch(usage_selector=selector)

    with (
        patch("litellm.router_strategy.lowest_tpm_rpm_v2.get_utc_datetime", return_value=_PINNED_MINUTE),
        request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send),  # pyright: ignore[reportArgumentType]  # test double
    ):
        routing_prefetch.arm(selector.usage, _DEPLOYMENT_COUNTERS)
        await read_batch.read_usage(deployments, None)
        with PrefetchedUsage.scoped(read_batch.prefetched_usage):
            picked: Final = await selector.async_get_available_deployments(_MODEL_GROUP, deployments, messages=_PING)

    selector.cancel_sync_task()
    assert picked["model_info"]["id"] == "dep-b"
    assert [kind for kind, _, _ in handle.reads] == ["batch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_strategy", ["simple-shuffle", "usage-based-routing-v2"])
async def test_a_cooldown_recorded_through_the_store_excludes_its_deployment_without_a_python_cooldown_read(
    routing_strategy: str,
):
    redis_cache = FakeRedisCache()
    router = _router(redis_cache, routing_strategy=routing_strategy)
    router.cooldown_cache.cooldown_store.put(
        "dep-a",
        {"exception_received": "429", "status_code": "429", "timestamp": _FAR_FUTURE, "cooldown_time": 60},
        60,
    )

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        picks = {await _route(router) for _ in range(5)}

    assert picks == {"dep-b"}
    assert _python_cooldown_reads(redis_cache) == ()


@pytest.mark.asyncio
async def test_arming_outside_a_request_scope_is_a_no_op():
    redis_cache = FakeRedisCache()
    router = _router(redis_cache)
    router.arm_routing_read_prefetch(_MODEL_GROUP, {})
    assert active_request_redis_batches() is None


@pytest.mark.asyncio
async def test_simple_shuffle_cooldown_prefetch_does_not_arm_usage_reads_or_read_eagerly():
    redis_cache = FakeRedisCache()
    router = _router(redis_cache, routing_strategy="simple-shuffle")

    with request_redis_batch_scope() as request:
        RoutingContext(router).arm_cooldown_prefetch()
        assert set(request.prefetched) == {"routing_cooldowns"}
        deployment = await _route(router)

    assert deployment in {"dep-a", "dep-b"}
    assert redis_cache.alone == []


def _router_recording_cooldown_reads() -> tuple[Router, FakeCooldowns]:
    """A router beside other groups whose cooldown store records the ids of every read it is asked for."""
    handle: Final = FakeCooldowns(has_redis=True)
    router: Final = Router(
        model_list=[
            _deployment("dep-a"),
            _deployment("dep-b"),
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "test", "mock_response": "pong"},
                "model_info": {"id": "dep-wild"},
            },
            {
                "model_name": "other",
                "litellm_params": {"model": "anthropic/other", "api_key": "test", "mock_response": "pong"},
                "model_info": {"id": "dep-other"},
            },
        ],
        model_group_alias={"claude-alias": _MODEL_GROUP},
        routing_strategy="simple-shuffle",
    )
    router.cooldown_cache = CooldownCache(store=RouterCooldownStore(handle), default_cooldown_time=60)
    return router, handle


def _reads(handle: FakeCooldowns) -> list[frozenset[str]]:
    return [frozenset(ids) for ids in handle.declared]


async def _pick(router: Router, model: str) -> str:
    deployment: Final = await router.async_get_available_deployment(model=model, messages=_PING, request_kwargs={})
    return deployment["model_info"]["id"]


_CLAUDE: Final = frozenset({"dep-a", "dep-b"})
_WILD: Final = frozenset({"dep-wild"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("armed_for", "routed", "reads", "pickable"),
    [
        pytest.param(
            None, _MODEL_GROUP, [_CLAUDE | {"dep-wild", "dep-other"}], _CLAUDE, id="no_model_reads_every_deployment"
        ),
        pytest.param(_MODEL_GROUP, _MODEL_GROUP, [_CLAUDE], _CLAUDE, id="group"),
        pytest.param("claude-alias", "claude-alias", [_CLAUDE], _CLAUDE, id="alias"),
        pytest.param("openai/gpt-4o", "openai/gpt-4o", [_WILD], _WILD, id="wildcard"),
        pytest.param("other", _MODEL_GROUP, [frozenset({"dep-other"}), _CLAUDE], _CLAUDE, id="uncovered"),
        pytest.param("bedrock/nothing-serves-this", _MODEL_GROUP, [_CLAUDE], _CLAUDE, id="no_candidates_arms_nothing"),
    ],
)
async def test_the_auth_cooldown_prefetch_reads_only_the_requested_models_candidates_and_routing_takes_it(
    armed_for: str | None, routed: str, reads: list[frozenset[str]], pickable: frozenset[str]
):
    router, handle = _router_recording_cooldown_reads()

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        RoutingContext(router).arm_cooldown_prefetch(armed_for)
        picked: Final = await _pick(router, routed)

    assert (_reads(handle), picked in pickable) == (reads, True)


@pytest.mark.asyncio
async def test_the_request_cooldown_prefetch_reads_only_the_requested_groups_candidates_and_routing_takes_it():
    router, handle = _router_recording_cooldown_reads()

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        RoutingContext(router).arm_read_prefetch(_MODEL_GROUP, {"metadata": {"user_api_key_team_id": "team-1"}})
        picked: Final = await _pick(router, _MODEL_GROUP)

    assert (_reads(handle), picked in _CLAUDE) == ([_CLAUDE], True)


def _user_entry() -> tuple[_CacheEntry, LiteLLM_UserTable]:
    entry = _CacheEntry("user-1", "user_row", LiteLLM_UserTable, 42)
    return entry, LiteLLM_UserTable(user_id="user-1", max_budget=None, spend=0.0)


@pytest.mark.asyncio
async def test_auth_write_back_rides_the_native_pre_call_batch_and_the_scope_drains_what_nobody_awaited():
    cache, fake = shared_cache()
    natives: Final = FakeNatives()
    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await _write_back([_user_entry()], cache)
        assert cache.get_local("user-1")["user_id"] == "user-1"
        assert natives.built[0].events == ["declare write_back (('user-1', 42),)"]
        await request.flush_pre_call()
    assert natives.built[0].events[-1] == "flush_pre"
    assert fake.commands == []


@pytest.mark.asyncio
async def test_auth_write_back_outside_a_scope_writes_through_as_before():
    cache, fake = shared_cache()
    await _write_back([_user_entry()], cache)
    assert fake.commands == [("WRITE_BACK", (("user-1", 42),))]


@pytest.mark.asyncio
async def test_a_key_the_request_read_as_absent_is_not_read_again_by_a_per_key_get():
    cache, fake = shared_cache()
    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["absent-key"], cache)
        assert await cache.async_get_cache("absent-key") is None
        await cache.async_set_cache("absent-key", {"v": 1}, ttl=5)
        assert await cache.async_get_cache("absent-key") == {"v": 1}
        await request.flush_pre_call()
    assert fake.commands == []


@pytest.mark.asyncio
async def test_management_object_writes_inside_a_request_ride_its_pipeline_and_write_through_outside():
    cache, fake = shared_cache()
    natives: Final = FakeNatives()
    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await cache.async_set_cache("team_id:t1", {"team_id": "t1"}, ttl=60)
        await cache.async_set_cache("hashed-key-object", {"token": "hashed-key-object"}, ttl=60)
        assert cache.get_local("team_id:t1") == {"team_id": "t1"}
        assert await cache.async_get_cache("hashed-key-object") == {"token": "hashed-key-object"}
        await request.flush_pre_call()
    assert natives.built[0].events == [
        "declare set team_id:t1 ttl=60.0",
        "declare set hashed-key-object ttl=60.0",
        "flush_pre",
    ]
    await cache.async_set_cache("team_id:t2", {"team_id": "t2"}, ttl=60)
    assert fake.commands == [("SET", "team_id:t2", 60.0)]


@pytest.mark.asyncio
async def test_a_team_refresh_inside_a_request_sends_its_set_and_alias_del_in_one_pipeline_before_returning():
    redis_cache = FakeRedisCache()
    cache, fake = shared_cache(connection=redis_cache)
    usage_cache = DualCache(redis_cache=redis_cache)
    usage_cache.in_memory_cache.set_cache("team_id:t1", "stale team")
    usage_cache.in_memory_cache.set_cache("team_alias:alpha", "stale alias")
    cache.set_local("team_alias:alpha", "stale alias", None)
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.internal_usage_cache = InternalUsageCache(dual_cache=usage_cache)
    team = LiteLLM_TeamTableCachedObj(team_id="t1", team_alias="alpha")
    natives: Final = FakeNatives()
    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await cache_team_object("t1", team, cache, proxy_logging_obj)
        assert natives.built[0].events == [
            "declare set team_id:t1 ttl=60.0",
            "declare del team_alias:alpha",
            "flush_pre",
        ], "the alias DEL must reach Redis before the refresh returns, or another request can refill memory from it"
        assert usage_cache.in_memory_cache.get_cache("team_id:t1") is None
        assert usage_cache.in_memory_cache.get_cache("team_alias:alpha") is None
        assert cache.get_local("team_alias:alpha") is None
        assert cache.get_local("team_id:t1")["team_id"] == "t1"
        await request.flush_pre_call()
    assert fake.commands == [] and redis_cache.alone == []


@pytest.mark.asyncio
async def test_a_changed_default_in_memory_ttl_reaches_the_native_store():
    cache, fake = shared_cache()
    cache.update_cache_ttl(default_in_memory_ttl=5, default_redis_ttl=None)
    assert [settings.memory_ttl for settings in fake.settings] == [None, 5]


@pytest.mark.asyncio
async def test_identity_prefetch_is_one_read_after_which_hits_and_misses_alike_cost_no_read():
    cache, fake = shared_cache(FakeAuthCache(redis={"key-hit": {"k": "key-hit"}}))
    natives: Final = FakeNatives()
    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["key-hit", "end_user_id:eu-miss", "key-hit"], cache)
        assert await cache.async_get_cache("key-hit") == {"k": "key-hit"}
        assert await cache.async_get_cache("end_user_id:eu-miss") is None
    assert natives.built[0].events == ["declare auth_cache end_user_id:eu-miss,key-hit", "flush_pre"]
    assert fake.commands == []
    assert cache.get_local("end_user_id:eu-miss") is None


@pytest.mark.asyncio
async def test_the_auth_prefetch_and_the_armed_cooldown_read_go_out_in_one_native_pipeline():
    cache, _fake = shared_cache(FakeAuthCache(redis={"key-hit": {"k": "key-hit"}}))
    cooldowns: Final = RouterCooldownStore(_FakeCooldownsHandle())
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["key-hit"], cache, before_read=lambda: arm_cooldowns(cooldowns, ["dep-a"]))
        arm_cooldowns(cooldowns, ["dep-a"])
        active = await read_cooldowns(cooldowns, ["dep-a"], None)
        await request.flush_pre_call()

    assert active == ()
    assert len(natives.built) == 1
    assert _declared_before_the_first_flush(natives) == ["declare cooldowns", "declare auth_cache key-hit"]
    assert not [event for event in natives.built[0].events[2:] if event.startswith("declare")]


@pytest.mark.asyncio
async def test_a_prefetch_served_from_memory_arms_no_early_cooldown_read():
    cache, _fake = shared_cache()
    cache.set_local("key-hit", {"k": "key-hit"}, None)
    armed: Final[list[str]] = []  # mutable-ok: records the arming callback

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["key-hit"], cache, before_read=lambda: armed.append("armed"))

    assert armed == []


@pytest.mark.asyncio
async def test_awaiting_a_native_read_runs_the_native_hooks_then_flushes_the_native_pre_batch_once_per_request():
    natives: Final = FakeNatives()

    with request_redis_batch_scope(
        post_call_deadline=7, native_factory=natives, native_send_factory=fake_send
    ) as request:  # pyright: ignore[reportArgumentType]  # test double
        native: Final = request.native()
        request.add_pre_call_flush_hook(lambda: native.events.append("hook"))
        result: Final = request.native_result(FakeNativePending(native, lambda: "answer"))
        declared_only: Final = list(native.events)
        answer: Final = await result
        await request.flush_pre_call()

    assert request.native() is native
    assert [built.post_call_deadline for built in natives.built] == [7]
    assert declared_only == []
    assert answer == "answer"
    assert native.events == ["hook", "flush_pre", "resolve", "flush_pre"]


@pytest.mark.asyncio
async def test_a_pre_call_flush_hook_declares_on_its_own_request_whichever_request_awaits_the_flush():
    natives: Final = FakeNatives()
    declared_on: Final[list[object]] = []

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as owner:  # pyright: ignore[reportArgumentType]  # test double
        owner.add_pre_call_flush_hook(lambda: declared_on.append(route(Batch.PRE_CALL).send))
        result: Final = owner.native_result(FakeNativePending(owner.native(), lambda: "answer"))
    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as other:  # pyright: ignore[reportArgumentType]  # test double
        other.native()
        answer: Final = await result

    assert answer == "answer"
    assert declared_on == [natives.built[0]]


@pytest.mark.asyncio
async def test_a_request_that_no_native_store_asks_for_builds_no_native_batches():
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await request.flush_pre_call()
        await request.flush_post_call()

    assert natives.built == []
