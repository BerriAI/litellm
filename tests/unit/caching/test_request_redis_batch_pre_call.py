"""One Redis pipeline per backend for the pre-call reads a request makes: rate limiter Lua groups, the
router's usage counter read, auth identity and spend counters all join the request batch."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import Batch, active_request_redis_batches, request_redis_batch_scope
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import cache_team_object
from litellm.proxy.auth.auth_object_prefetch import _CacheEntry, _write_back, prefetch_identity_keys
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitUnverifiableError,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores._route import route
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.redis.stores.router_cooldowns import ActiveCooldowns, RouterCooldownStore
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, RouterUsageStore, UsageCounter
from litellm.router_api import RoutingContext
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage
from litellm.router_utils import routing_prefetch
from litellm.router_utils.routing_prefetch import arm_cooldowns, read_cooldowns
from litellm.router_utils.routing_read_batch import RoutingReadBatch
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits
from tests.unit.redis.stores.router_usage_double import FakeUsageRead

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


_OK: Final = [0, 1, 1700000000]


def _over_limit(current: int, limit: int) -> list[int]:
    return [1, 1, current, limit]


def _check_replies(replies: dict[str, object]) -> Callable[..., Any]:
    """``CHECK_AND_INCREMENT_BY_N_SCRIPT`` per descriptor window: OK unless ``replies`` holds another reply or an
    exception for it."""

    async def check(keys: list[str], args: list[object]) -> object:
        reply: Final = replies.get(keys[0], _OK)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return check


def _limiter(
    replies: dict[str, object] | None = None, fail_closed: bool = False
) -> tuple[PROXY_MaxParallelRequestsHandler_v3, FakeRateLimits]:
    fakes: Final = FakeRateLimits(
        check_and_increment_by_n_script=_check_replies(replies or {}),
        window_guarded_token_increment_script=AsyncMock(return_value=[1, 0]),
    )
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(dual_cache=DualCache()),
        fail_closed_resolver=lambda: fail_closed,
        rate_limit_store=fakes.store(),
    )
    return limiter, fakes


def _descriptor(key: str, value: str, rpm: int) -> RateLimitDescriptor:
    return {"key": key, "value": value, "rate_limit": {"requests_per_unit": rpm}}


def _refunds(fakes: FakeRateLimits) -> list[tuple[str, float]]:
    refund_script: Final = fakes.window_guarded_token_increment_script
    assert isinstance(refund_script, AsyncMock)
    return [(call.kwargs["keys"][1], call.kwargs["args"][1]) for call in refund_script.await_args_list]


def _declared_before_the_first_flush(natives: FakeNatives) -> list[str]:
    events: Final = natives.built[0].events
    return events[: events.index("flush_pre")]


@pytest.mark.asyncio
async def test_descriptor_lua_calls_share_one_native_pipeline_and_each_keeps_its_result():
    limiter, _fakes = _limiter()
    natives: Final = FakeNatives()
    descriptors = [
        _descriptor("api_key", "k1", 10),
        _descriptor("model_per_key", "k1:gpt", 5),
        _descriptor("team", "t1", 20),
    ]

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=descriptors,
            increments=[{"requests": 1}, {"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OK"
    assert [s["descriptor_key"] for s in response["statuses"]] == ["api_key", "model_per_key", "team"]
    assert _declared_before_the_first_flush(natives) == ["declare check_and_increment"] * 3


@pytest.mark.asyncio
async def test_an_over_limit_descriptor_in_the_pipeline_refunds_the_groups_that_were_applied():
    limiter, fakes = _limiter({"{team:t1}:window": _over_limit(21, 20)})

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OVER_LIMIT"
    assert response["statuses"][0]["descriptor_key"] == "team"
    assert _refunds(fakes) == [("{api_key:k1}:requests", -1)]


@pytest.mark.asyncio
async def test_an_over_limit_descriptor_also_refunds_the_groups_the_pipeline_incremented_after_it():
    limiter, fakes = _limiter({"{api_key:k1}:window": _over_limit(11, 10)})

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[
                _descriptor("api_key", "k1", 10),
                _descriptor("team", "t1", 20),
                _descriptor("model_per_key", "k1:gpt", 5),
            ],
            increments=[{"requests": 1}, {"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OVER_LIMIT"
    assert response["statuses"][0]["descriptor_key"] == "api_key"
    assert _refunds(fakes) == [("{team:t1}:requests", -1), ("{model_per_key:k1:gpt}:requests", -1)]


@pytest.mark.asyncio
async def test_a_redis_denial_stands_when_another_pipelined_group_fails():
    limiter, fakes = _limiter(
        {"{api_key:k1}:window": _over_limit(11, 10), "{team:t1}:window": ValueError("script blew up")}
    )

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[
                _descriptor("api_key", "k1", 10),
                _descriptor("team", "t1", 20),
                _descriptor("model_per_key", "k1:gpt", 5),
            ],
            increments=[{"requests": 1}, {"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OVER_LIMIT"  # not the in-memory fallback's verdict
    assert response["statuses"][0]["descriptor_key"] == "api_key"
    assert _refunds(fakes) == [("{model_per_key:k1:gpt}:requests", -1)]


@pytest.mark.asyncio
async def test_one_failed_lua_group_refunds_the_other_pipelined_groups_and_falls_back_to_in_memory():
    limiter, fakes = _limiter({"{api_key:k1}:window": ValueError("script blew up")})

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OK"
    assert len(response["statuses"]) == 2  # in-memory enforcement covered both descriptors
    assert _refunds(fakes) == [("{team:t1}:requests", -1)]
    assert limiter.rate_limits.local((RateLimitScope("api_key", "k1").counter("requests"),)) == (1,)


@pytest.mark.parametrize(
    "replies, refunded",
    [
        ({"{api_key:k1}:window": ValueError("script blew up")}, [("{team:t1}:requests", -1)]),
        (
            {"{api_key:k1}:window": ConnectionError("redis down"), "{team:t1}:window": ConnectionError("redis down")},
            [],
        ),
    ],
    ids=["one_group_failed", "pipeline_failed"],
)
@pytest.mark.asyncio
async def test_fail_closed_rejects_when_a_pipelined_lua_group_cannot_be_verified(
    replies: dict[str, object], refunded: list[tuple[str, float]]
):
    limiter, fakes = _limiter(replies, fail_closed=True)

    with request_redis_batch_scope(), pytest.raises(RateLimitUnverifiableError) as exc:
        await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert exc.value.status_code == 503
    assert _refunds(fakes) == refunded


@pytest.mark.asyncio
async def test_a_pipeline_failure_refunds_nothing_and_falls_back_to_in_memory_enforcement():
    limiter, fakes = _limiter(
        {"{api_key:k1}:window": ConnectionError("redis down"), "{team:t1}:window": ConnectionError("redis down")}
    )

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OK"
    assert len(response["statuses"]) == 2
    assert _refunds(fakes) == []


@pytest.mark.asyncio
async def test_without_a_request_scope_descriptor_groups_run_one_at_a_time_and_stop_at_the_first_denial():
    calls: Final[list[str]] = []  # mutable-ok: one window key per script call

    async def check(keys: list[str], args: list[object]) -> object:
        calls.append(keys[0])
        return _over_limit(11, 10) if keys[0] == "{team:t1}:window" else _OK

    limiter, fakes = _limiter()
    fakes.check_and_increment_by_n_script = check

    response = await limiter.atomic_check_and_increment_by_n(
        descriptors=[
            _descriptor("api_key", "k1", 10),
            _descriptor("team", "t1", 20),
            _descriptor("model_per_key", "k1:gpt", 5),
        ],
        increments=[{"requests": 1}, {"requests": 1}, {"requests": 1}],
    )

    assert response["overall_code"] == "OVER_LIMIT"
    assert calls == ["{api_key:k1}:window", "{team:t1}:window"]
    assert _refunds(fakes) == [("{api_key:k1}:requests", -1)]


@dataclass
class _FakeCooldownsHandle:
    """A connected cooldown store whose ``active`` lands on the request's fake native batch."""

    has_redis: Final = True

    def active(self, model_ids: Sequence[str], send: FakeNativeRequest) -> FakeNativePending[ActiveCooldowns]:
        send.events.append("declare cooldowns")
        return FakeNativePending(send, lambda: ())


@pytest.mark.asyncio
async def test_the_admission_script_and_the_armed_cooldown_read_go_out_in_one_native_pipeline():
    async def window(keys: list[str], args: list[object]) -> object:
        return ["1700000000", 1]

    fakes: Final = FakeRateLimits(batch_rate_limiter_script=window)
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(dual_cache=DualCache()), rate_limit_store=fakes.store()
    )
    cooldowns: Final = RouterCooldownStore(_FakeCooldownsHandle())
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        arm_cooldowns(cooldowns, ["dep-a"])
        response = await limiter.should_rate_limit([_descriptor("api_key", "k1", 10)])
        active = await read_cooldowns(cooldowns, ["dep-a"], None)

    assert response["overall_code"] == "OK"
    assert active == ()
    assert len(natives.built) == 1
    assert _declared_before_the_first_flush(natives) == ["declare cooldowns", "declare sliding_window"]
    assert request.prefetched == {}


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
