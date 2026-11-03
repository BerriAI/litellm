"""One Redis pipeline per backend for the pre-call reads a request makes: rate limiter Lua groups, the
router's usage counter read, auth identity and spend counters all join the request batch."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.caching.dual_cache as dual_cache_module
from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import active_request_redis_batches, request_redis_batch_scope
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import cache_team_object
from litellm.proxy.auth.auth_object_prefetch import _CacheEntry, _write_back, prefetch_identity_keys
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitUnverifiableError,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.redis.stores.router_cooldowns import ActiveCooldowns, RouterCooldownStore
from litellm.router_api import RoutingContext
from litellm.router_utils.routing_prefetch import ArmedRead, TierRead, arm_cooldowns, read_cooldowns
from litellm.router_utils.routing_read_batch import (
    ROUTER_USAGE_TARGET,
    RoutingPrefetch,
)
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits

from .test_redis_batch import (
    FakeClient,
    FakeNativePending,
    FakeNativeRequest,
    FakeNatives,
    FakeRedisCache,
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


def _lua_ok_replies(command: tuple[Any, ...]) -> Any:
    if command[0] == "MGET":
        return [None for _ in command[1:]]
    if command[0] == "SET":
        return True
    raise AssertionError(command)


@pytest.mark.asyncio
async def test_descriptor_lua_calls_share_one_native_pipeline_and_each_keeps_its_result():
    limiter, _fakes = _limiter()
    natives: Final = FakeNatives()
    descriptors = [
        _descriptor("api_key", "k1", 10),
        _descriptor("model_per_key", "k1:gpt", 5),
        _descriptor("team", "t1", 20),
    ]

    with request_redis_batch_scope(native_factory=natives):  # pyright: ignore[reportArgumentType]  # test double
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
    """A connected cooldown store whose ``declare_active`` lands on the request's fake native batch."""

    def declare_active(
        self, request: FakeNativeRequest, model_ids: Sequence[str]
    ) -> FakeNativePending[ActiveCooldowns]:
        request.events.append("declare cooldowns")
        return FakeNativePending(request, lambda: ())


@pytest.mark.asyncio
async def test_the_admission_script_and_the_armed_cooldown_read_go_out_in_one_native_pipeline():
    async def window(keys: list[str], args: list[object]) -> object:
        return ["1700000000", 1]

    fakes: Final = FakeRateLimits(batch_rate_limiter_script=window)
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(dual_cache=DualCache()), rate_limit_store=fakes.store()
    )
    cooldowns: Final = RouterCooldownStore(
        FakeRedisCache(FakeClient(_lua_ok_replies)), handle=lambda _c: _FakeCooldownsHandle()
    )  # pyright: ignore[reportArgumentType]  # test double
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives) as request:  # pyright: ignore[reportArgumentType]  # test double
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


_PINNED_MINUTE: Final = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
_START: Final = 1_000_000.0
_PING: Final = [{"role": "user", "content": "ping"}]


@pytest.fixture
def pinned_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Pins the usage counters' minute and the router cache's read throttle clock; tests move `now[0]`."""
    now: Final = [_START]  # mutable-ok: the test advances the pinned clock
    monkeypatch.setattr("litellm.router_strategy.lowest_tpm_rpm_v2.get_utc_datetime", lambda: _PINNED_MINUTE)
    monkeypatch.setattr(dual_cache_module.time, "time", lambda: now[0])
    return now


def _counter_keys(router: Router, deployment_ids: tuple[str, ...] = ("dep-a", "dep-b")) -> frozenset[str]:
    deployments: Final = [d for d in router.model_list if d["model_info"]["id"] in deployment_ids]
    tpm, rpm = router.lowesttpm_logger_v2.usage_counter_keys(deployments)
    return frozenset(tpm + rpm)


def _tpm_key(router: Router, deployment_id: str) -> str:
    deployment: Final = next(d for d in router.model_list if d["model_info"]["id"] == deployment_id)
    return router.lowesttpm_logger_v2.usage_counter_keys([deployment])[0][0]


def _counter_mgets(router: Router, client: FakeClient, redis_cache: FakeRedisCache) -> tuple[frozenset[str], ...]:
    counters: Final = _counter_keys(router)
    pipelined: Final = tuple(
        frozenset(command[1:])
        for pipeline in client.pipelines
        for command in pipeline.commands
        if command[0] == "MGET" and counters.intersection(command[1:])
    )
    alone: Final = tuple(
        frozenset(keys) for command, keys in redis_cache.alone if command == "MGET" and counters.intersection(keys)
    )
    return pipelined + alone


def _python_cooldown_reads(client: FakeClient, redis_cache: FakeRedisCache) -> tuple[str, ...]:
    pipelined: Final = tuple(
        key
        for pipeline in client.pipelines
        for command in pipeline.commands
        if command[0] == "MGET"
        for key in command[1:]
    )
    alone: Final = tuple(key for command, keys in redis_cache.alone if command == "MGET" for key in keys)
    return tuple(key for key in pipelined + alone if key.endswith(":cooldown"))


def _replies_with(values: dict[str, object]) -> Callable[[tuple[Any, ...]], Any]:
    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":
            return [None if values.get(key) is None else json.dumps(values[key]) for key in command[1:]]
        return _lua_ok_replies(command)

    return replies


async def _route(router: Router) -> str:
    deployment: Final = await router.async_get_available_deployment(
        model=_MODEL_GROUP, messages=_PING, request_kwargs={}
    )
    return deployment["model_info"]["id"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_armed_routing_read_is_one_pipeline_and_routing_issues_no_read_of_its_own():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment = await _route(router)

    assert deployment in {"dep-a", "dep-b"}
    assert len(client.pipelines) == 1
    commands = client.pipelines[0].commands
    assert [c[0] for c in commands] == ["MGET"]
    assert frozenset(commands[0][1:]) == _counter_keys(router)
    assert redis_cache.alone == []


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_strategy", ["simple-shuffle", "usage-based-routing-v2"])
async def test_a_cooldown_recorded_through_the_store_excludes_its_deployment_without_a_python_cooldown_read(
    routing_strategy: str,
):
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
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
    assert _python_cooldown_reads(client, redis_cache) == ()


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_a_prefetch_that_does_not_cover_the_routing_keys_is_ignored_and_routing_reads_itself():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        armed = request.prefetched["routing_read"]
        assert isinstance(armed, ArmedRead)
        request.prefetched["routing_read"] = ArmedRead(
            read=TierRead(armed.read.cache, ("other",)),
            fetched=armed.fetched,
            result=armed.result,
            previous=armed.previous,
        )
        deployment = await _route(router)
        assert request.prefetched == {}

    assert deployment in {"dep-a", "dep-b"}
    assert [frozenset(keys) for command, keys in redis_cache.alone if command == "MGET"] == [_counter_keys(router)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_a_prefetch_with_incomplete_usage_keys_releases_its_reservations():
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    router: Final = _router(redis_cache)

    with request_redis_batch_scope():
        RoutingPrefetch.arm(RoutingContext(router), router.lowesttpm_logger_v2, router.model_list[:1])
        deployment: Final = await _route(router)

    assert deployment in {"dep-a", "dep-b"}
    assert [frozenset(keys) for command, keys in redis_cache.alone if command == "MGET"] == [_counter_keys(router)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_a_failed_prefetch_falls_back_to_the_selectors_own_read():
    client = FakeClient(_lua_ok_replies, fail=ConnectionError("redis down"))
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment = await _route(router)

    assert deployment in {"dep-a", "dep-b"}
    assert [frozenset(keys) for command, keys in redis_cache.alone if command == "MGET"] == [_counter_keys(router)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_the_armed_routing_read_is_declared_under_the_router_usage_family():
    """The prefetch is declared before routing runs under a target of its own, so the pipeline that
    carries it renders ``redis.pipeline router_usage`` instead of a bare ``redis.pipeline``."""
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)
    pipeline_targets: list[str | None] = []  # mutable-ok: filled by the recording hook

    async def record(**kwargs: object) -> None:
        pipeline_targets.append(current_service_target())

    redis_cache.service_logger_obj.async_service_success_hook = record  # pyright: ignore[reportAttributeAccessIssue]  # fake, records the hook call

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await _route(router)
    async with asyncio.timeout(5):
        while not pipeline_targets:
            await asyncio.sleep(0.01)

    assert pipeline_targets == [ROUTER_USAGE_TARGET]


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_an_abandoned_prefetch_still_backfills_the_counters_it_read():
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    router: Final = _router(redis_cache)
    client.reply_for = _replies_with({_tpm_key(router, "dep-a"): 100})

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await request.flush_all()
    after_abandoned: Final = len(_counter_mgets(router, client, redis_cache))

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment: Final = await _route(router)

    assert deployment == "dep-b"
    assert after_abandoned == 1
    assert len(_counter_mgets(router, client, redis_cache)) == 1
    assert redis_cache.alone == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_prefetch_settlement_keeps_newer_memory_values_and_backfills_misses():
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    router: Final = _router(redis_cache)
    dep_a: Final = _tpm_key(router, "dep-a")
    dep_b: Final = _tpm_key(router, "dep-b")
    client.reply_for = _replies_with({dep_a: 5, dep_b: 7})
    usage_cache: Final = router.lowesttpm_logger_v2.router_cache

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        usage_cache.in_memory_cache.set_cache(dep_a, 50)
        await request.flush_all()

    assert await usage_cache.async_batch_get_cache([dep_a, dep_b], local_only=True) == [50, 7]


@pytest.mark.asyncio
async def test_an_abandoned_prefetch_whose_mget_fails_releases_its_reservation(pinned_clock: list[float]):
    mget_replies: Final = iter((ConnectionError("redis down"), None))
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    router: Final = _router(redis_cache)

    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":
            failure: Final = next(mget_replies)
            if isinstance(failure, Exception):
                return failure
            return ["100" if key == _tpm_key(router, "dep-a") else None for key in command[1:]]
        return _lua_ok_replies(command)

    client.reply_for = replies

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await request.flush_all()
    after_failure: Final = len(_counter_mgets(router, client, redis_cache))

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment: Final = await _route(router)

    assert after_failure == 1
    assert len(_counter_mgets(router, client, redis_cache)) == 2
    assert deployment == "dep-b"


@pytest.mark.asyncio
async def test_arming_outside_a_request_scope_is_a_no_op():
    redis_cache = FakeRedisCache(FakeClient(_lua_ok_replies))
    router = _router(redis_cache)
    router.arm_routing_read_prefetch(_MODEL_GROUP, {})
    assert active_request_redis_batches() is None


@pytest.mark.asyncio
async def test_simple_shuffle_arms_no_python_prefetch_and_reads_no_cooldown_key():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache, routing_strategy="simple-shuffle")

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        assert set(request.prefetched) == {"routing_cooldowns"}
        deployment = await _route(router)

    assert deployment in {"dep-a", "dep-b"}
    assert client.pipelines == []
    assert redis_cache.alone == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_requests_within_the_read_interval_read_the_counters_from_redis_once():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)
    request_round_trips: list[tuple[int, int]] = []  # mutable-ok: one entry per simulated request

    for _ in range(3):
        pipeline_count = len(client.pipelines)
        alone_count = len(redis_cache.alone)
        with request_redis_batch_scope():
            router.arm_routing_read_prefetch(_MODEL_GROUP, {})
            await _route(router)
        request_round_trips.append((len(client.pipelines) - pipeline_count, len(redis_cache.alone) - alone_count))

    assert len(_counter_mgets(router, client, redis_cache)) == 1
    assert request_round_trips[1:] == [(0, 0), (0, 0)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("pinned_clock")
async def test_concurrent_requests_share_one_counter_read_per_interval():
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    router: Final = _router(redis_cache)
    first_armed: Final = asyncio.Event()
    both_armed: Final = asyncio.Event()

    async def route_after_both_requests_arm() -> str:
        with request_redis_batch_scope():
            router.arm_routing_read_prefetch(_MODEL_GROUP, {})
            if first_armed.is_set():
                both_armed.set()
            else:
                first_armed.set()
            await both_armed.wait()
            return await _route(router)

    deployments: Final = await asyncio.gather(route_after_both_requests_arm(), route_after_both_requests_arm())

    assert set(deployments) <= {"dep-a", "dep-b"}
    assert len(_counter_mgets(router, client, redis_cache)) == 1


@pytest.mark.asyncio
async def test_the_prefetch_reads_the_counters_again_once_the_read_interval_elapses(pinned_clock: list[float]):
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await _route(router)

    client.reply_for = _replies_with({_tpm_key(router, "dep-a"): 100})
    pinned_clock[0] = _START + router.lowesttpm_logger_v2.router_cache.redis_batch_cache_expiry + 1
    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment = await _route(router)

    assert len(_counter_mgets(router, client, redis_cache)) == 2
    assert deployment == "dep-b"


@pytest.mark.asyncio
async def test_the_prefetch_mget_carries_only_the_keys_whose_read_is_due(pinned_clock: list[float]):
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)
    usage_cache = router.lowesttpm_logger_v2.router_cache

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await _route(router)

    for key in _counter_keys(router, ("dep-a",)):
        usage_cache.in_memory_cache.set_cache(key, 1)
    pinned_clock[0] = _START + usage_cache.redis_batch_cache_expiry + 1
    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await _route(router)

    second_pipeline_mgets = tuple(command for command in client.pipelines[1].commands if command[0] == "MGET")

    assert len(client.pipelines) == 2
    assert len(second_pipeline_mgets) == 1
    assert frozenset(second_pipeline_mgets[0][1:]) == _counter_keys(router, ("dep-b",))


@pytest.mark.asyncio
async def test_two_backends_flush_concurrently_one_pipeline_each():
    a_client, b_client = FakeClient(_lua_ok_replies), FakeClient(_lua_ok_replies)
    a, b = FakeRedisCache(a_client), FakeRedisCache(b_client)
    with request_redis_batch_scope() as request:
        ra = request.batch(a).mget(["x", "y"])
        rb = request.batch(b).mget(["x"])
        await asyncio.gather(ra, rb)
    assert len(a_client.pipelines) == 1 and len(b_client.pipelines) == 1


class _SameServerCache(FakeRedisCache):
    def __init__(self, client: FakeClient, namespace: str | None = None, **redis_kwargs: object) -> None:
        super().__init__(client, namespace)
        self.redis_kwargs = redis_kwargs


@pytest.mark.asyncio
async def test_caches_built_from_the_same_connection_settings_share_the_request_pipeline():
    client = FakeClient(_lua_ok_replies)
    proxy_cache = _SameServerCache(client, host="r", port=6379, db=0)
    router_cache = _SameServerCache(FakeClient(_lua_ok_replies), port="6379", host="r", db=0, password=None)
    other_cache = _SameServerCache(FakeClient(_lua_ok_replies), host="r", port=6380, db=0)
    with request_redis_batch_scope() as request:
        assert request.batch(proxy_cache) is request.batch(router_cache)
        assert request.batch(proxy_cache) is not request.batch(other_cache)
        a = request.batch(proxy_cache).mget(["a"])
        b = request.batch(router_cache).mget(["b"])
        await asyncio.gather(a, b)
    assert len(client.pipelines) == 1
    assert [c[0] for c in client.pipelines[0].commands] == ["MGET", "MGET"]


@pytest.mark.asyncio
async def test_caches_on_one_server_with_different_namespaces_keep_their_own_key_prefix():
    proxy_client, router_client = FakeClient(_lua_ok_replies), FakeClient(_lua_ok_replies)
    proxy_cache = _SameServerCache(proxy_client, namespace="proxy", host="r", port=6379, db=0)
    router_cache = _SameServerCache(router_client, namespace="router", host="r", port=6379, db=0)
    with request_redis_batch_scope() as request:
        await asyncio.gather(request.batch(proxy_cache).mget(["a"]), request.batch(router_cache).mget(["b"]))
    sent: Final = tuple(
        tuple(command for pipe in client.pipelines for command in pipe.commands)
        for client in (proxy_client, router_client)
    )
    assert sent == ((("MGET", "proxy:a"),), (("MGET", "router:b"),)), "each cache reads under its own namespace"


def _user_entry() -> tuple[_CacheEntry, LiteLLM_UserTable]:
    entry = _CacheEntry("user-1", "user_row", LiteLLM_UserTable, 42)
    return entry, LiteLLM_UserTable(user_id="user-1", max_budget=None, spend=0.0)


@pytest.mark.asyncio
async def test_auth_write_back_rides_the_native_pre_call_batch_and_the_scope_drains_what_nobody_awaited():
    cache, fake = shared_cache()
    natives: Final = FakeNatives()
    with request_redis_batch_scope(native_factory=natives) as request:  # pyright: ignore[reportArgumentType]  # test double
        await _write_back([_user_entry()], cache)
        assert cache.get_local("user-1")["user_id"] == "user-1"
        assert natives.built[0].events == ["declare write_back (('user-1', 42),)"]
        await request.flush_all()
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
    with request_redis_batch_scope(native_factory=FakeNatives()) as request:  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["absent-key"], cache)
        assert await cache.async_get_cache("absent-key") is None
        await cache.async_set_cache("absent-key", {"v": 1}, ttl=5)
        assert await cache.async_get_cache("absent-key") == {"v": 1}
        await request.flush_all()
    assert fake.commands == []


@pytest.mark.asyncio
async def test_management_object_writes_inside_a_request_ride_its_pipeline_and_write_through_outside():
    cache, fake = shared_cache()
    natives: Final = FakeNatives()
    with request_redis_batch_scope(native_factory=natives) as request:  # pyright: ignore[reportArgumentType]  # test double
        await cache.async_set_cache("team_id:t1", {"team_id": "t1"}, ttl=60)
        await cache.async_set_cache("hashed-key-object", {"token": "hashed-key-object"}, ttl=60)
        assert cache.get_local("team_id:t1") == {"team_id": "t1"}
        assert await cache.async_get_cache("hashed-key-object") == {"token": "hashed-key-object"}
        await request.flush_all()
    assert natives.built[0].events == [
        "declare set team_id:t1 ttl=60.0",
        "declare set hashed-key-object ttl=60.0",
        "flush_pre",
    ]
    await cache.async_set_cache("team_id:t2", {"team_id": "t2"}, ttl=60)
    assert fake.commands == [("SET", "team_id:t2", 60.0)]


@pytest.mark.asyncio
async def test_a_team_refresh_inside_a_request_sends_its_set_and_alias_del_in_one_pipeline_before_returning():
    redis_cache = FakeRedisCache(FakeClient(_lua_ok_replies))
    cache, fake = shared_cache(connection=redis_cache)
    usage_cache = DualCache(redis_cache=redis_cache)
    usage_cache.in_memory_cache.set_cache("team_id:t1", "stale team")
    usage_cache.in_memory_cache.set_cache("team_alias:alpha", "stale alias")
    cache.set_local("team_alias:alpha", "stale alias", None)
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.internal_usage_cache = InternalUsageCache(dual_cache=usage_cache)
    team = LiteLLM_TeamTableCachedObj(team_id="t1", team_alias="alpha")
    natives: Final = FakeNatives()
    with request_redis_batch_scope(native_factory=natives) as request:  # pyright: ignore[reportArgumentType]  # test double
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
        await request.flush_all()
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
    with request_redis_batch_scope(native_factory=natives):  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["key-hit", "end_user_id:eu-miss", "key-hit"], cache)
        assert await cache.async_get_cache("key-hit") == {"k": "key-hit"}
        assert await cache.async_get_cache("end_user_id:eu-miss") is None
    assert natives.built[0].events == ["declare auth_cache end_user_id:eu-miss,key-hit", "flush_pre"]
    assert fake.commands == []
    assert cache.get_local("end_user_id:eu-miss") is None


@pytest.mark.asyncio
async def test_the_auth_prefetch_and_the_armed_cooldown_read_go_out_in_one_native_pipeline():
    cache, _fake = shared_cache(FakeAuthCache(redis={"key-hit": {"k": "key-hit"}}))
    cooldowns: Final = RouterCooldownStore(
        FakeRedisCache(FakeClient(_lua_ok_replies)), handle=lambda _c: _FakeCooldownsHandle()
    )  # pyright: ignore[reportArgumentType]  # test double
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives) as request:  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["key-hit"], cache, before_read=lambda: arm_cooldowns(cooldowns, ["dep-a"]))
        arm_cooldowns(cooldowns, ["dep-a"])
        active = await read_cooldowns(cooldowns, ["dep-a"], None)
        await request.flush_all()

    assert active == ()
    assert len(natives.built) == 1
    assert _declared_before_the_first_flush(natives) == ["declare cooldowns", "declare auth_cache key-hit"]
    assert not [event for event in natives.built[0].events[2:] if event.startswith("declare")]


@pytest.mark.asyncio
async def test_a_prefetch_served_from_memory_arms_no_early_cooldown_read():
    cache, _fake = shared_cache()
    cache.set_local("key-hit", {"k": "key-hit"}, None)
    armed: Final[list[str]] = []  # mutable-ok: records the arming callback

    with request_redis_batch_scope(native_factory=FakeNatives()):  # pyright: ignore[reportArgumentType]  # test double
        await prefetch_identity_keys(["key-hit"], cache, before_read=lambda: armed.append("armed"))

    assert armed == []


@pytest.mark.asyncio
async def test_awaiting_a_native_read_runs_the_native_hooks_then_flushes_the_native_pre_batch_once_per_request():
    natives: Final = FakeNatives()

    with request_redis_batch_scope(post_call_deadline=7, native_factory=natives) as request:  # pyright: ignore[reportArgumentType]  # test double
        native: Final = request.native()
        request.add_native_flush_hook(lambda: native.events.append("hook"))
        result: Final = request.native_result(FakeNativePending(native, lambda: "answer"))
        declared_only: Final = list(native.events)
        answer: Final = await result
        await request.flush_all()

    assert request.native() is native
    assert [built.post_call_deadline for built in natives.built] == [7]
    assert declared_only == []
    assert answer == "answer"
    assert native.events == ["hook", "flush_pre", "resolve", "hook", "flush_pre"]


@pytest.mark.asyncio
async def test_a_request_that_no_native_store_asks_for_builds_no_native_batches():
    natives: Final = FakeNatives()

    with request_redis_batch_scope(native_factory=natives) as request:  # pyright: ignore[reportArgumentType]  # test double
        await request.flush_all()
        await request.flush_post_call()

    assert natives.built == []
