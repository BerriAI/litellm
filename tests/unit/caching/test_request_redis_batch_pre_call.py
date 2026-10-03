"""One Redis pipeline per backend for the pre-call reads a request makes: rate limiter Lua groups, the
router's cooldown and usage read, auth identity and spend counters all join the request batch."""

from __future__ import annotations

import asyncio
import hashlib
import json
from itertools import chain
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.caching.dual_cache as dual_cache_module
from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import active_request_redis_batches, request_redis_batch_scope
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import _cache_team_object
from litellm.proxy.auth.auth_object_prefetch import _CacheEntry, _write_back, prefetch_identity_keys
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    CHECK_AND_INCREMENT_BY_N_SCRIPT,
    RateLimitDescriptor,
    RateLimitUnverifiableError,
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.cooldown_cache import ROUTER_COOLDOWNS_TARGET, CooldownCache
from litellm.router_utils.routing_read_batch import (
    ROUTER_COOLDOWNS_USAGE_TARGET,
    ROUTER_USAGE_TARGET,
    RoutingPrefetch,
    _routing_read_target,  # pyright: ignore[reportPrivateUsage]  # the family rule under test
)

from .test_redis_batch import FakeClient, FakeRedisCache, replies

_MODEL_GROUP = "claude"
_FAR_FUTURE = 4_102_444_800.0  # 2100-01-01, a cooldown stamped then is still active


def sha_of(script: str) -> str:
    return hashlib.sha1(script.encode()).hexdigest()  # noqa: S324


def _limiter(redis_cache: FakeRedisCache, fail_closed: bool = False) -> _PROXY_MaxParallelRequestsHandler_v3:
    dual_cache = DualCache()
    limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(dual_cache=dual_cache),
        fail_closed_resolver=lambda: fail_closed,
    )
    dual_cache.attach_redis_cache(redis_cache)  # after init: the fake has no server to register scripts on
    limiter.check_and_increment_by_n_script = AsyncMock(
        side_effect=AssertionError("descriptor groups must ride the request pipeline")
    )
    limiter.window_guarded_token_increment_script = AsyncMock(return_value=[1, 0])
    return limiter


def _descriptor(key: str, value: str, rpm: int) -> RateLimitDescriptor:
    return {"key": key, "value": value, "rate_limit": {"requests_per_unit": rpm}}


def _refunds(limiter: _PROXY_MaxParallelRequestsHandler_v3) -> list[tuple[str, float]]:
    refund_script = limiter.window_guarded_token_increment_script
    assert isinstance(refund_script, AsyncMock)
    return [(call.kwargs["keys"][1], call.kwargs["args"][1]) for call in refund_script.await_args_list]


def _lua_ok_replies(command: tuple[Any, ...]) -> Any:
    if command[0] == "EVALSHA":
        return [0, 1, 1700000000]  # OK: one counter, new_counter=1, window_start
    if command[0] == "MGET":
        return [None for _ in command[1:]]
    if command[0] == "SET":
        return True
    raise AssertionError(command)


@pytest.mark.asyncio
async def test_descriptor_lua_calls_share_one_pipeline_and_each_keeps_its_result():
    client = FakeClient(_lua_ok_replies)
    limiter = _limiter(FakeRedisCache(client))
    descriptors = [
        _descriptor("api_key", "k1", 10),
        _descriptor("model_per_key", "k1:gpt", 5),
        _descriptor("team", "t1", 20),
    ]

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=descriptors,
            increments=[{"requests": 1}, {"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OK"
    assert [s["descriptor_key"] for s in response["statuses"]] == ["api_key", "model_per_key", "team"]
    assert len(client.pipelines) == 1
    evalshas = [c for c in client.pipelines[0].commands if c[0] == "EVALSHA"]
    assert len(evalshas) == 3
    assert {c[1] for c in evalshas} == {sha_of(CHECK_AND_INCREMENT_BY_N_SCRIPT)}
    assert [c[3] for c in evalshas] == ["{api_key:k1}:window", "{model_per_key:k1:gpt}:window", "{team:t1}:window"]


@pytest.mark.asyncio
async def test_an_over_limit_descriptor_in_the_pipeline_refunds_the_groups_that_were_applied():
    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "EVALSHA" and command[3] == "{team:t1}:window":
            return [1, 1, 21, 20]  # OVER_LIMIT on its first counter
        return _lua_ok_replies(command)

    client = FakeClient(replies)
    redis_cache = FakeRedisCache(client)
    limiter = _limiter(redis_cache)

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OVER_LIMIT"
    assert response["statuses"][0]["descriptor_key"] == "team"
    assert _refunds(limiter) == [("{api_key:k1}:requests", -1.0)]
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_an_over_limit_descriptor_also_refunds_the_groups_the_pipeline_incremented_after_it():
    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "EVALSHA" and command[3] == "{api_key:k1}:window":
            return [1, 1, 11, 10]  # OVER_LIMIT on the first group; the later groups already incremented
        return _lua_ok_replies(command)

    client = FakeClient(replies)
    redis_cache = FakeRedisCache(client)
    limiter = _limiter(redis_cache)

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
    assert _refunds(limiter) == [("{team:t1}:requests", -1.0), ("{model_per_key:k1:gpt}:requests", -1.0)]
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_a_redis_denial_stands_when_another_pipelined_group_fails():
    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "EVALSHA" and command[3] == "{api_key:k1}:window":
            return [1, 1, 11, 10]  # OVER_LIMIT
        if command[0] == "EVALSHA" and command[3] == "{team:t1}:window":
            return ValueError("script blew up")
        return _lua_ok_replies(command)

    client = FakeClient(replies)
    redis_cache = FakeRedisCache(client)
    limiter = _limiter(redis_cache)

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
    assert _refunds(limiter) == [("{model_per_key:k1:gpt}:requests", -1.0)]
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_one_failed_lua_group_refunds_the_other_pipelined_groups_and_falls_back_to_in_memory():
    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "EVALSHA" and command[3] == "{api_key:k1}:window":
            return ValueError("script blew up")
        return _lua_ok_replies(command)

    client = FakeClient(replies)
    redis_cache = FakeRedisCache(client)
    limiter = _limiter(redis_cache)

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OK"
    assert len(response["statuses"]) == 2  # in-memory enforcement covered both descriptors
    assert _refunds(limiter) == [("{team:t1}:requests", -1.0)]
    assert len(client.pipelines) == 1


@pytest.mark.parametrize(
    "client, refunded",
    [
        (
            FakeClient(
                lambda command: (
                    ValueError("script blew up")
                    if command[0] == "EVALSHA" and command[3] == "{api_key:k1}:window"
                    else _lua_ok_replies(command)
                )
            ),
            [("{team:t1}:requests", -1.0)],
        ),
        (FakeClient(_lua_ok_replies, fail=ConnectionError("redis down")), []),
    ],
    ids=["one_group_failed", "pipeline_failed"],
)
@pytest.mark.asyncio
async def test_fail_closed_rejects_when_a_pipelined_lua_group_cannot_be_verified(
    client: FakeClient, refunded: list[tuple[str, float]]
):
    limiter = _limiter(FakeRedisCache(client), fail_closed=True)

    with request_redis_batch_scope(), pytest.raises(RateLimitUnverifiableError) as exc:
        await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert exc.value.status_code == 503
    assert _refunds(limiter) == refunded
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_a_pipeline_failure_refunds_nothing_and_falls_back_to_in_memory_enforcement():
    client = FakeClient(_lua_ok_replies, fail=ConnectionError("redis down"))
    limiter = _limiter(FakeRedisCache(client))

    with request_redis_batch_scope():
        response = await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )

    assert response["overall_code"] == "OK"
    assert len(response["statuses"]) == 2
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_without_a_request_scope_descriptor_groups_run_the_script_directly_as_before():
    client = FakeClient(_lua_ok_replies)
    limiter = _limiter(FakeRedisCache(client))
    limiter.check_and_increment_by_n_script = AsyncMock(return_value=[0, 1, 1700000000])

    response = await limiter.atomic_check_and_increment_by_n(
        descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
        increments=[{"requests": 1}, {"requests": 1}],
    )

    assert response["overall_code"] == "OK"
    assert limiter.check_and_increment_by_n_script.await_count == 2
    assert client.pipelines == []


def _deployment(deployment_id: str) -> dict:
    return {
        "model_name": _MODEL_GROUP,
        "litellm_params": {"model": "anthropic/claude-x", "api_key": "test", "mock_response": "pong"},
        "model_info": {"id": deployment_id},
    }


def _router(redis_cache: FakeRedisCache, routing_strategy: str = "usage-based-routing-v2") -> Router:
    router = Router(model_list=[_deployment("dep-a"), _deployment("dep-b")], routing_strategy=routing_strategy)
    router._update_redis_cache(cache=redis_cache)
    return router


@pytest.mark.asyncio
async def test_armed_routing_read_rides_the_admission_pipeline_and_routing_issues_no_read_of_its_own():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)
    limiter = _limiter(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10), _descriptor("team", "t1", 20)],
            increments=[{"requests": 1}, {"requests": 1}],
        )
        deployment = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert len(client.pipelines) == 1
    commands = client.pipelines[0].commands
    assert [c[0] for c in commands] == ["MGET", "EVALSHA", "EVALSHA"]
    mget_keys = set(commands[0][1:])
    assert {CooldownCache.get_cooldown_cache_key("dep-a"), CooldownCache.get_cooldown_cache_key("dep-b")} <= mget_keys
    assert any(":tpm:" in key for key in mget_keys) and any(":rpm:" in key for key in mget_keys)
    assert redis_cache.alone == []


@pytest.mark.asyncio
async def test_a_cooldown_recorded_locally_after_the_prefetch_left_still_excludes_its_deployment():
    expired = {"exception_received": "429", "status_code": "429", "timestamp": 0.0, "cooldown_time": 60}

    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":  # Redis holds a stale cooldown for dep-b and nothing for dep-a
            return [
                json.dumps(expired) if key == CooldownCache.get_cooldown_cache_key("dep-b") else None
                for key in command[1:]
            ]
        return _lua_ok_replies(command)

    client = FakeClient(replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)
    cooldown_store = router.cooldown_cache.cooldown_store
    assert cooldown_store.in_memory_cache is not None

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        cooldown_store.in_memory_cache.set_cache(
            CooldownCache.get_cooldown_cache_key("dep-a"),
            {"exception_received": "429", "status_code": "429", "timestamp": _FAR_FUTURE, "cooldown_time": 60},
        )
        picks = {
            (
                await router.async_get_available_deployment(
                    model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
                )
            )["model_info"]["id"]
            for _ in range(5)
        }

    assert picks == {"dep-b"}
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_a_prefetch_that_does_not_cover_the_routing_keys_is_ignored_and_routing_reads_itself():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        armed = request.prefetched["routing_read"]
        assert isinstance(armed, RoutingPrefetch)
        request.prefetched["routing_read"] = RoutingPrefetch(
            keys=frozenset({"other"}),
            fetched=armed.fetched,
            result=armed.result,
            reservations=armed.reservations,
        )
        deployment = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )
        assert request.prefetched == {}

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert len(redis_cache.alone) == 1  # the shared cooldown+usage read, one round trip as in P1


@pytest.mark.asyncio
async def test_a_prefetch_with_incomplete_usage_keys_releases_cooldown_reservations():
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    router: Final = _router(redis_cache)
    cooldown_keys: Final = frozenset(
        {
            CooldownCache.get_cooldown_cache_key("dep-a"),
            CooldownCache.get_cooldown_cache_key("dep-b"),
        }
    )

    with request_redis_batch_scope():
        RoutingPrefetch.arm(router, router.lowesttpm_logger_v2, router.model_list[:1])
        deployment: Final = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    fallback_cooldown_mgets: Final = tuple(
        keys for command, keys in redis_cache.alone if command == "MGET" and cooldown_keys.issubset(keys)
    )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert len(fallback_cooldown_mgets) == 1


@pytest.mark.asyncio
async def test_a_failed_prefetch_falls_back_to_the_shared_read():
    client = FakeClient(_lua_ok_replies, fail=ConnectionError("redis down"))
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert len(redis_cache.alone) == 1
    cooldown_keys: Final = frozenset(
        {
            CooldownCache.get_cooldown_cache_key("dep-a"),
            CooldownCache.get_cooldown_cache_key("dep-b"),
        }
    )
    fallback_cooldown_mgets: Final = tuple(
        keys for command, keys in redis_cache.alone if command == "MGET" and cooldown_keys.issubset(keys)
    )
    assert len(fallback_cooldown_mgets) == 1


@pytest.mark.asyncio
async def test_the_armed_routing_read_is_declared_under_the_router_cooldowns_family():
    """The prefetch is declared before routing runs under a target of its own, so the pipeline that
    carries it renders ``redis.pipeline router_cooldowns`` instead of a bare ``redis.pipeline``."""
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache, routing_strategy="simple-shuffle")
    pipeline_targets: list[str | None] = []  # mutable-ok: filled by the recording hook

    async def record(**kwargs: object) -> None:
        pipeline_targets.append(current_service_target())

    redis_cache.service_logger_obj.async_service_success_hook = record  # pyright: ignore[reportAttributeAccessIssue]  # fake, records the hook call

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )
    await asyncio.gather(*(t for t in asyncio.all_tasks() if t is not asyncio.current_task()))

    assert pipeline_targets == [ROUTER_COOLDOWNS_TARGET]


@pytest.mark.asyncio
async def test_an_abandoned_prefetch_still_backfills_the_cooldown_it_read(monkeypatch):
    clock: Final = 1_000_000.0
    monkeypatch.setattr(dual_cache_module.time, "time", lambda: clock)
    cooldown_key: Final = CooldownCache.get_cooldown_cache_key("dep-a")
    active_cooldown: Final = {
        "exception_received": "429",
        "status_code": "429",
        "timestamp": _FAR_FUTURE,
        "cooldown_time": 60,
    }

    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":
            return [json.dumps(active_cooldown) if key == cooldown_key else None for key in command[1:]]
        return _lua_ok_replies(command)

    client: Final = FakeClient(replies)
    redis_cache: Final = FakeRedisCache(client)
    redis_cache.store[cooldown_key] = active_cooldown
    router: Final = _router(redis_cache, routing_strategy="simple-shuffle")
    limiter: Final = _limiter(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10)],
            increments=[{"requests": 1}],
        )

    pipeline_count: Final = len(client.pipelines)
    first_cooldown_mgets: Final = tuple(
        command
        for command in chain.from_iterable(pipeline.commands for pipeline in client.pipelines[:pipeline_count])
        if command[0] == "MGET" and cooldown_key in command[1:]
    )
    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment: Final = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )
        cooldowns: Final = await router.cooldown_cache.async_get_active_cooldowns(["dep-a"], parent_otel_span=None)

    second_cooldown_mgets: Final = tuple(
        command
        for command in chain.from_iterable(pipeline.commands for pipeline in client.pipelines[pipeline_count:])
        if command[0] == "MGET" and cooldown_key in command[1:]
    )

    assert deployment["model_info"]["id"] == "dep-b"
    assert [model_id for model_id, _ in cooldowns] == ["dep-a"]
    assert len(first_cooldown_mgets) == 1
    assert second_cooldown_mgets == ()
    assert redis_cache.alone == []


@pytest.mark.asyncio
async def test_prefetch_settlement_keeps_newer_memory_values_and_backfills_misses(monkeypatch):
    clock: Final = 1_000_000.0
    monkeypatch.setattr(dual_cache_module.time, "time", lambda: clock)
    dep_a_key: Final = CooldownCache.get_cooldown_cache_key("dep-a")
    dep_b_key: Final = CooldownCache.get_cooldown_cache_key("dep-b")
    old_cooldown: Final = {
        "exception_received": "429",
        "status_code": "429",
        "timestamp": _FAR_FUTURE,
        "cooldown_time": 60,
    }
    newer_memory_cooldown: Final = {
        "exception_received": "429",
        "status_code": "429",
        "timestamp": _FAR_FUTURE + 1,
        "cooldown_time": 60,
    }
    redis_only_cooldown: Final = {
        "exception_received": "429",
        "status_code": "429",
        "timestamp": _FAR_FUTURE + 2,
        "cooldown_time": 60,
    }

    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":
            return [json.dumps(redis_cache.store[key]) if key in redis_cache.store else None for key in command[1:]]
        return _lua_ok_replies(command)

    client: Final = FakeClient(replies)
    redis_cache: Final = FakeRedisCache(client)
    redis_cache.store[dep_a_key] = old_cooldown
    redis_cache.store[dep_b_key] = redis_only_cooldown
    router: Final = _router(redis_cache, routing_strategy="simple-shuffle")
    memory_cache: Final = router.cooldown_cache.cooldown_store.in_memory_cache
    assert memory_cache is not None

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        memory_cache.set_cache(dep_a_key, newer_memory_cooldown)
        await request.flush_all()

    prefetched_mgets: Final = tuple(command for command in client.pipelines[0].commands if command[0] == "MGET")

    assert len(prefetched_mgets) == 1
    assert frozenset(prefetched_mgets[0][1:]) == frozenset({dep_a_key, dep_b_key})
    assert memory_cache.get_cache(dep_a_key) == newer_memory_cooldown
    assert memory_cache.get_cache(dep_b_key) == redis_only_cooldown


@pytest.mark.asyncio
async def test_an_abandoned_prefetch_whose_mget_fails_releases_its_reservation(monkeypatch):
    clock: Final = 1_000_000.0
    monkeypatch.setattr(dual_cache_module.time, "time", lambda: clock)
    cooldown_key: Final = CooldownCache.get_cooldown_cache_key("dep-a")
    active_cooldown: Final = {
        "exception_received": "429",
        "status_code": "429",
        "timestamp": _FAR_FUTURE,
        "cooldown_time": 60,
    }
    mget_replies: Final = iter((ConnectionError("redis down"), None))

    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":
            response: Final = next(mget_replies)
            if isinstance(response, Exception):
                return response
            return [json.dumps(active_cooldown) if key == cooldown_key else None for key in command[1:]]
        return _lua_ok_replies(command)

    client: Final = FakeClient(replies)
    redis_cache: Final = FakeRedisCache(client)
    redis_cache.store[cooldown_key] = active_cooldown
    router: Final = _router(redis_cache, routing_strategy="simple-shuffle")

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await request.flush_all()

    pipeline_count: Final = len(client.pipelines)
    first_cooldown_mgets: Final = tuple(
        command
        for command in chain.from_iterable(pipeline.commands for pipeline in client.pipelines[:pipeline_count])
        if command[0] == "MGET" and cooldown_key in command[1:]
    )
    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment: Final = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )
        cooldowns: Final = await router.cooldown_cache.async_get_active_cooldowns(["dep-a"], parent_otel_span=None)

    second_cooldown_mgets: Final = tuple(
        command
        for command in chain.from_iterable(pipeline.commands for pipeline in client.pipelines[pipeline_count:])
        if command[0] == "MGET" and cooldown_key in command[1:]
    )

    assert deployment["model_info"]["id"] == "dep-b"
    assert [model_id for model_id, _ in cooldowns] == ["dep-a"]
    assert len(first_cooldown_mgets) == 1
    assert len(second_cooldown_mgets) == 1


@pytest.mark.asyncio
async def test_a_cooldown_that_leaves_memory_before_routing_is_read_again(monkeypatch):
    clock: Final = 1_000_000.0
    monkeypatch.setattr(dual_cache_module.time, "time", lambda: clock)
    cooldown_key: Final = CooldownCache.get_cooldown_cache_key("dep-a")
    active_cooldown: Final = {
        "exception_received": "429",
        "status_code": "429",
        "timestamp": _FAR_FUTURE,
        "cooldown_time": 60,
    }
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    redis_cache.store[cooldown_key] = active_cooldown
    router: Final = _router(redis_cache, routing_strategy="simple-shuffle")
    cooldown_store: Final = router.cooldown_cache.cooldown_store
    memory_cache: Final = cooldown_store.in_memory_cache
    assert memory_cache is not None
    memory_cache.set_cache(cooldown_key, active_cooldown)

    with request_redis_batch_scope() as request:
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await request.flush_all()
        memory_cache.delete_cache(cooldown_key)
        deployment: Final = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    prefetched_mgets: Final = tuple(command for command in client.pipelines[0].commands if command[0] == "MGET")
    fallback_cooldown_mgets: Final = tuple(
        keys for command, keys in redis_cache.alone if command == "MGET" and cooldown_key in keys
    )

    assert len(prefetched_mgets) == 1
    assert prefetched_mgets[0][1:] == (CooldownCache.get_cooldown_cache_key("dep-b"),)
    assert deployment["model_info"]["id"] == "dep-b"
    assert fallback_cooldown_mgets == ((cooldown_key,),)


@pytest.mark.asyncio
async def test_arming_outside_a_request_scope_is_a_no_op():
    redis_cache = FakeRedisCache(FakeClient(_lua_ok_replies))
    router = _router(redis_cache)
    router.arm_routing_read_prefetch(_MODEL_GROUP, {})
    assert active_request_redis_batches() is None


@pytest.mark.asyncio
async def test_simple_shuffle_prefetches_only_its_cooldown_read_into_the_admission_pipeline():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache, routing_strategy="simple-shuffle")
    limiter = _limiter(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10)],
            increments=[{"requests": 1}],
        )
        deployment = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    assert deployment["model_info"]["id"] in {"dep-a", "dep-b"}
    assert len(client.pipelines) == 1
    commands = client.pipelines[0].commands
    assert [c[0] for c in commands] == ["MGET", "EVALSHA"]
    assert set(commands[0][1:]) == {
        CooldownCache.get_cooldown_cache_key("dep-a"),
        CooldownCache.get_cooldown_cache_key("dep-b"),
    }
    assert redis_cache.alone == []

    shuffle = Router(model_list=[_deployment("dep-a")], routing_strategy="simple-shuffle")
    shuffle._update_redis_cache(cache=redis_cache)
    with request_redis_batch_scope() as request:
        shuffle.arm_routing_read_prefetch(_MODEL_GROUP, {})
        armed = request.prefetched["routing_read"]
        assert isinstance(armed, RoutingPrefetch)
        assert armed.keys == {CooldownCache.get_cooldown_cache_key("dep-a")}  # no usage counters for shuffle


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_strategy", ["simple-shuffle", "usage-based-routing-v2"])
@pytest.mark.parametrize("with_limiter", [True, False])
async def test_requests_within_the_cooldown_read_interval_read_cooldowns_from_redis_once(
    routing_strategy: str, with_limiter: bool
):
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache, routing_strategy=routing_strategy)
    limiter = _limiter(redis_cache)
    request_round_trips: list[tuple[int, int]] = []

    for _ in range(3):
        pipeline_count = len(client.pipelines)
        alone_count = len(redis_cache.alone)
        with request_redis_batch_scope():
            router.arm_routing_read_prefetch(_MODEL_GROUP, {})
            if with_limiter:
                await limiter.atomic_check_and_increment_by_n(
                    descriptors=[_descriptor("api_key", "k1", 10)],
                    increments=[{"requests": 1}],
                )
            await router.async_get_available_deployment(
                model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
            )
        request_round_trips.append((len(client.pipelines) - pipeline_count, len(redis_cache.alone) - alone_count))

    pipeline_mgets = [command for pipeline in client.pipelines for command in pipeline.commands if command[0] == "MGET"]
    alone_mgets = [keys for command, keys in redis_cache.alone if command == "MGET"]
    cooldown_keys = {
        CooldownCache.get_cooldown_cache_key("dep-a"),
        CooldownCache.get_cooldown_cache_key("dep-b"),
    }
    cooldown_mgets = [command[1:] for command in pipeline_mgets if cooldown_keys.intersection(command[1:])] + [
        keys for keys in alone_mgets if cooldown_keys.intersection(keys)
    ]

    assert len(cooldown_mgets) == 1
    if not with_limiter:
        assert request_round_trips[1:] == [(0, 0), (0, 0)]


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_cooldown_read_per_interval():
    client: Final = FakeClient(_lua_ok_replies)
    redis_cache: Final = FakeRedisCache(client)
    router: Final = _router(redis_cache)
    first_armed: Final = asyncio.Event()
    both_armed: Final = asyncio.Event()

    async def route_after_both_requests_arm():
        with request_redis_batch_scope():
            router.arm_routing_read_prefetch(_MODEL_GROUP, {})
            if first_armed.is_set():
                both_armed.set()
            else:
                first_armed.set()
            await both_armed.wait()
            return await router.async_get_available_deployment(
                model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
            )

    deployments: Final = await asyncio.gather(route_after_both_requests_arm(), route_after_both_requests_arm())
    cooldown_keys: Final = frozenset(
        {
            CooldownCache.get_cooldown_cache_key("dep-a"),
            CooldownCache.get_cooldown_cache_key("dep-b"),
        }
    )
    cooldown_mgets: Final = tuple(
        command
        for pipeline in client.pipelines
        for command in pipeline.commands
        if command[0] == "MGET" and cooldown_keys.intersection(command[1:])
    )

    assert all(deployment["model_info"]["id"] in {"dep-a", "dep-b"} for deployment in deployments)
    assert len(cooldown_mgets) == 1


@pytest.mark.asyncio
async def test_the_prefetch_reads_cooldowns_again_once_the_read_interval_elapses(monkeypatch):
    first_time: Final = 1_000_000.0
    monkeypatch.setattr(dual_cache_module.time, "time", lambda: first_time)
    active_cooldown = {
        "exception_received": "429",
        "status_code": "429",
        "timestamp": _FAR_FUTURE,
        "cooldown_time": 60,
    }
    mget_results = iter((None, active_cooldown))

    def replies(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":
            result = next(mget_results)
            return [
                None if result is None or key != CooldownCache.get_cooldown_cache_key("dep-a") else json.dumps(result)
                for key in command[1:]
            ]
        return _lua_ok_replies(command)

    client = FakeClient(replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache, routing_strategy="simple-shuffle")
    cooldown_store = router.cooldown_cache.cooldown_store

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    monkeypatch.setattr(
        dual_cache_module.time,
        "time",
        lambda: first_time + cooldown_store.redis_batch_cache_expiry + 1,
    )
    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        deployment = await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    cooldown_keys = {
        CooldownCache.get_cooldown_cache_key("dep-a"),
        CooldownCache.get_cooldown_cache_key("dep-b"),
    }
    cooldown_mgets = [
        command
        for pipeline in client.pipelines
        for command in pipeline.commands
        if command[0] == "MGET" and cooldown_keys.intersection(command[1:])
    ]
    assert len(cooldown_mgets) == 2
    assert deployment["model_info"]["id"] == "dep-b"


@pytest.mark.asyncio
async def test_the_prefetch_mget_carries_only_the_keys_whose_read_is_due(monkeypatch):
    first_time: Final = 1_000_000.0
    monkeypatch.setattr(dual_cache_module.time, "time", lambda: first_time)
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache, routing_strategy="usage-based-routing-v2")
    cooldown_store = router.cooldown_cache.cooldown_store
    usage_cache = router.lowesttpm_logger_v2.router_cache
    time_offset = cooldown_store.redis_batch_cache_expiry + 0.5

    assert time_offset < usage_cache.redis_batch_cache_expiry

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    monkeypatch.setattr(dual_cache_module.time, "time", lambda: first_time + time_offset)
    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    cooldown_keys = frozenset(
        {
            CooldownCache.get_cooldown_cache_key("dep-a"),
            CooldownCache.get_cooldown_cache_key("dep-b"),
        }
    )
    second_pipeline_mgets = tuple(command for command in client.pipelines[1].commands if command[0] == "MGET")

    assert len(client.pipelines) == 2
    assert len(second_pipeline_mgets) == 1
    assert frozenset(second_pipeline_mgets[0][1:]) == cooldown_keys


@pytest.mark.asyncio
async def test_two_backends_flush_concurrently_one_pipeline_each():
    a_client, b_client = FakeClient(_lua_ok_replies), FakeClient(_lua_ok_replies)
    a, b = FakeRedisCache(a_client), FakeRedisCache(b_client)
    with request_redis_batch_scope() as request:
        ra = request.batch(a).mget(["x", "y"])
        rb = request.batch(b).mget(["x"])
        await asyncio.gather(ra, rb)
    assert len(a_client.pipelines) == 1 and len(b_client.pipelines) == 1


@pytest.mark.asyncio
async def test_a_single_lua_group_rides_the_pipeline_with_the_armed_routing_read():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    router = _router(redis_cache)
    limiter = _limiter(redis_cache)

    with request_redis_batch_scope():
        router.arm_routing_read_prefetch(_MODEL_GROUP, {})
        await limiter.atomic_check_and_increment_by_n(
            descriptors=[_descriptor("api_key", "k1", 10)],
            increments=[{"requests": 1}],
        )
        await router.async_get_available_deployment(
            model=_MODEL_GROUP, messages=[{"role": "user", "content": "ping"}], request_kwargs={}
        )

    assert len(client.pipelines) == 1
    assert [c[0] for c in client.pipelines[0].commands] == ["MGET", "EVALSHA"]
    assert redis_cache.alone == []


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
async def test_auth_write_back_rides_the_next_round_trip_and_the_scope_drains_what_nobody_awaited():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    cache = UserApiKeyCache(redis_cache=redis_cache)
    with request_redis_batch_scope() as request:
        await _write_back([_user_entry()], cache)
        assert client.pipelines == []  # not sent yet: the SET waits for the next round trip
        await request.batch(redis_cache).mget(["spend:key:k1"])
        assert len(client.pipelines) == 1
        kinds = [c[0] for c in client.pipelines[0].commands]
        assert kinds == ["MGET", "SET"] or kinds == ["SET", "MGET"]
        set_command = next(c for c in client.pipelines[0].commands if c[0] == "SET")
        assert set_command[1] == "user-1" and set_command[3] == 42
        assert json.loads(set_command[2])["user_id"] == "user-1"
        assert cache.in_memory_cache.get_cache("user-1") is not None

        await _write_back([_user_entry()], cache)
        assert len(client.pipelines) == 1
        await request.flush_all()
    assert len(client.pipelines) == 2
    assert [c[0] for c in client.pipelines[1].commands] == ["SET"]


@pytest.mark.asyncio
async def test_auth_write_back_outside_a_scope_writes_through_as_before():
    redis_cache = FakeRedisCache(FakeClient(_lua_ok_replies))
    cache = UserApiKeyCache(redis_cache=redis_cache)
    await _write_back([_user_entry()], cache)
    assert [(op[0], [(key, ttl) for key, _value, ttl in op[1]]) for op in redis_cache.alone] == [
        ("SET_PIPELINE", [("user-1", 42)])
    ]


@pytest.mark.asyncio
async def test_a_key_the_request_mget_read_as_absent_is_not_read_again_by_a_per_key_get():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    cache = UserApiKeyCache(redis_cache=redis_cache)
    with request_redis_batch_scope() as request:
        assert await request.batch(redis_cache).mget(["absent-key"]) == {"absent-key": None}
        assert await cache.async_get_cache("absent-key") is None
        assert redis_cache.alone == [] and len(client.pipelines) == 1
        await cache.async_set_cache("absent-key", {"v": 1}, ttl=5)
        await request.flush_all()
    assert [c[:2] for c in client.pipelines[1].commands] == [("SET", "absent-key")]


@pytest.mark.asyncio
async def test_management_object_writes_inside_a_request_ride_its_pipeline_and_write_through_outside():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    cache = UserApiKeyCache(redis_cache=redis_cache)
    with request_redis_batch_scope() as request:
        await cache.async_set_cache("team_id:t1", {"team_id": "t1"}, ttl=60)
        await cache.async_set_cache("hashed-key-object", {"token": "hashed-key-object"}, ttl=60)
        assert client.pipelines == []
        assert cache.in_memory_cache.get_cache("team_id:t1") == {"team_id": "t1"}
        assert await cache.async_get_cache("hashed-key-object") == {"token": "hashed-key-object"}
        await request.flush_all()
    assert sorted((c[0], c[1], c[3]) for c in client.pipelines[0].commands) == [
        ("SET", "hashed-key-object", 60),
        ("SET", "team_id:t1", 60),
    ]
    await cache.async_set_cache("team_id:t2", {"team_id": "t2"}, ttl=60)
    assert len(client.pipelines) == 1
    assert redis_cache.alone == [("SET", "team_id:t2", {"team_id": "t2"})]


@pytest.mark.asyncio
async def test_a_team_refresh_inside_a_request_sends_its_set_and_alias_del_in_one_pipeline_before_returning():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    cache = UserApiKeyCache(redis_cache=redis_cache)
    usage_cache = DualCache(redis_cache=redis_cache)
    usage_cache.in_memory_cache.set_cache("team_id:t1", "stale team")
    usage_cache.in_memory_cache.set_cache("team_alias:alpha", "stale alias")
    cache.in_memory_cache.set_cache("team_alias:alpha", "stale alias")
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.internal_usage_cache = InternalUsageCache(dual_cache=usage_cache)
    team = LiteLLM_TeamTableCachedObj(team_id="t1", team_alias="alpha")
    with request_redis_batch_scope() as request:
        await _cache_team_object("t1", team, cache, proxy_logging_obj)
        assert [c[:2] for c in client.pipelines[0].commands] == [("SET", "team_id:t1"), ("DEL", "team_alias:alpha")], (
            "the alias DEL must reach Redis before the refresh returns, or another request can refill memory from it"
        )
        assert redis_cache.alone == []
        assert usage_cache.in_memory_cache.get_cache("team_id:t1") is None
        assert usage_cache.in_memory_cache.get_cache("team_alias:alpha") is None
        assert cache.in_memory_cache.get_cache("team_alias:alpha") is None
        assert cache.in_memory_cache.get_cache("team_id:t1")["team_id"] == "t1"
        await request.flush_all()
    assert len(client.pipelines) == 1 and redis_cache.alone == []


@pytest.mark.asyncio
async def test_a_pipelined_management_write_without_a_ttl_expires_in_redis_like_the_direct_path():
    client = FakeClient(_lua_ok_replies)
    redis_cache = FakeRedisCache(client)
    cache = UserApiKeyCache(redis_cache=redis_cache)
    cache.update_cache_ttl(default_in_memory_ttl=5, default_redis_ttl=None)
    with request_redis_batch_scope() as request:
        await cache.async_set_cache("team_id:t1", {"team_id": "t1"})
        await request.flush_all()
    assert [(c[0], c[1], c[3]) for c in client.pipelines[0].commands] == [("SET", "team_id:t1", 5)]


@pytest.mark.asyncio
async def test_identity_prefetch_is_one_mget_after_which_hits_and_misses_alike_cost_no_read():
    client = FakeClient(replies)
    redis_cache = FakeRedisCache(client)
    cache = UserApiKeyCache(redis_cache=redis_cache)
    with request_redis_batch_scope():
        await prefetch_identity_keys(["key-hit", "end_user_id:eu-miss", "key-hit"], cache)
        assert [c[0] for c in client.pipelines[0].commands] == ["MGET"]
        assert sorted(client.pipelines[0].commands[0][1:]) == ["end_user_id:eu-miss", "key-hit"]
        assert await cache.async_get_cache("key-hit") == {"k": "key-hit"}
        assert await cache.async_get_cache("end_user_id:eu-miss") is None
    assert len(client.pipelines) == 1 and redis_cache.alone == []
    assert cache.in_memory_cache.get_cache("end_user_id:eu-miss") is None


@pytest.mark.parametrize(
    ("cooldown_keys", "usage_keys", "expected"),
    [
        (("cooldown:a",), (), ROUTER_COOLDOWNS_TARGET),
        ((), ("usage:a",), ROUTER_USAGE_TARGET),
        (("cooldown:a",), ("usage:a",), ROUTER_COOLDOWNS_USAGE_TARGET),
    ],
)
def test_the_routing_read_family_follows_the_keys_that_are_actually_due(
    cooldown_keys: tuple[str, ...], usage_keys: tuple[str, ...], expected: str
):
    """A routing MGET is ``router_cooldowns`` when only cooldown keys go out, ``router_usage`` when the
    cooldowns were already in memory and only usage counters go out, and the combined family otherwise."""
    assert _routing_read_target(cooldown_keys, usage_keys) == expected
