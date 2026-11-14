"""The post-call writes of a request: spend counters, rate-limit token scripts and slot releases and deployment TPM
all ride the native post-call batch, which goes out once the success/failure callbacks have run (or at the deadline
when no callback phase closes it). The response-cache SET stays direct so the next identical request can hit it while
the callbacks are still running."""

from __future__ import annotations

import datetime
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final

import pytest

from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

import litellm
from litellm.caching.caching import Cache
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_batch import (
    active_request_redis_batches,
    drain_post_call_redis_batches,
    flush_post_call_redis_batches,
    request_redis_batch_scope,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    ParallelSlotAcquisition,
    PROXY_MaxParallelRequestsHandler_v3,
    RequestRateLimiterStash,
)
from litellm.proxy.spend_tracking.spend_counter_batch import PendingSpendIncrement
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import CounterIncrement, RateLimitScope
from litellm.redis.stores.router_usage import CounterIncrement as UsageIncrement
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.types.utils import ModelResponse
from tests.unit.redis.stores.fake_rate_limits import FakeRateLimits
from tests.unit.redis.stores.router_usage_double import FakeUsageHandle, usage_store

from .redis_batch_double import FakeNatives, FakeRedisCache, fake_send


async def _script_outside_the_pipeline(keys: Sequence[str], args: Sequence[object]) -> object:
    raise AssertionError("post-call scripts must ride the post-call pipeline")


class PostCallFakeRedisCache(FakeRedisCache):
    """Records the direct (non-pipelined) writes an owner falls back to."""

    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]:
        return _script_outside_the_pipeline

    async def async_delete_cache(self, key: str, **kwargs: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # the fake drops RedisCache's unused kwargs
        self.alone.append(("DEL", key))
        self.store.pop(key, None)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.alone.append(("SET", key, dict(kwargs)))
        self.store[key] = value


def _limiter(fakes: FakeRateLimits) -> PROXY_MaxParallelRequestsHandler_v3:
    return PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=InternalUsageCache(dual_cache=DualCache()), rate_limit_store=fakes.store()
    )


_K1: Final = RateLimitScope("api_key", "k1")


def _slot_stash(slot_id: str, *gauges: RateLimitScope) -> RequestRateLimiterStash:
    return RequestRateLimiterStash(parallel_slot=ParallelSlotAcquisition(slot_id=slot_id, gauges=gauges))


def _token_ops(*scopes: RateLimitScope) -> list[CounterIncrement]:
    return [CounterIncrement(counter=scope.counter("tokens"), amount=10, ttl=60) for scope in scopes]


def _scripts(calls: list[tuple[str, list[str], list[object]]], reply: object) -> Callable[..., Awaitable[object]]:
    async def script(keys: list[str], args: list[object]) -> object:
        calls.append(("script", keys, args))
        if isinstance(reply, Exception):
            raise reply
        return reply

    return script


def _response_cache(redis_cache: FakeRedisCache) -> Cache:
    cache = Cache(type="local")
    cache.type = "redis"  # pyright: ignore[reportAttributeAccessIssue]  # the fake stands in for the Redis backend
    cache.cache = redis_cache
    return cache


def _tpm_selector(handle: FakeUsageHandle) -> LowestTPMLoggingHandler_v2:
    selector: Final = LowestTPMLoggingHandler_v2(usage=usage_store(handle), routing_args={"ttl": 60})
    selector.cancel_sync_task()
    return selector


def _tpm_kwargs() -> Mapping[str, object]:
    return {
        "standard_logging_object": {
            "model_group": "gpt",
            "model_id": "dep-a",
            "hidden_params": {"litellm_model_name": "openai/gpt-4o-mini"},
            "total_tokens": 42,
        },
        "litellm_params": {"metadata": {}},
    }


@pytest.mark.asyncio
async def test_post_call_owners_declare_on_the_native_post_call_batch_until_the_callbacks_finish():
    redis_cache = PostCallFakeRedisCache()
    calls: list[tuple[str, list[str], list[object]]] = []  # mutable-ok: filled by the fake scripts
    fakes = FakeRateLimits(token_increment_script=_scripts(calls, [10.0]), parallel_release_script=_scripts(calls, [0]))
    limiter = _limiter(fakes)
    usage = FakeUsageHandle(redis={})
    tpm = _tpm_selector(usage)

    with request_redis_batch_scope():
        await tpm.async_log_success_event(_tpm_kwargs(), None, None, None)
        await limiter.async_increment_tokens_with_ttl_preservation(_token_ops(_K1))
        await limiter._release_stashed_parallel_slot(_slot_stash("slot-1", _K1), None, in_logging_callback=True)
        await flush_post_call_redis_batches()

    assert fakes.post_call == ["token_increment", "parallel_release"]
    assert calls == [
        ("script", ["{api_key:k1}:tokens"], [10, 60]),
        ("script", ["{api_key:k1}:max_parallel_requests"], ["slot-1"]),
    ]
    tpm_counter = next(iter(usage.local))
    assert usage.calls == [("post_call", (UsageIncrement(tpm_counter, 42, 60),))]
    assert (tpm_counter.deployment_id, tpm_counter.model, tpm_counter.kind) == ("dep-a", "openai/gpt-4o-mini", "tpm")
    assert usage.local == {tpm_counter: 42}
    assert redis_cache.alone == []


@pytest.mark.asyncio
async def test_outside_a_request_the_tpm_increment_goes_out_on_its_own_pipeline():
    usage = FakeUsageHandle(redis={})

    await _tpm_selector(usage).async_log_success_event(_tpm_kwargs(), None, None, None)

    assert [name for name, _ in usage.calls] == ["pipeline"]
    assert usage.local == usage.redis == {next(iter(usage.local)): 42}


@pytest.mark.asyncio
async def test_the_response_cache_set_reaches_redis_before_the_post_call_pipeline_goes_out():
    redis_cache = PostCallFakeRedisCache()
    response_cache = _response_cache(redis_cache)
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt", "ttl": 120}
    cache_key = response_cache.get_cache_key(**kwargs)

    with request_redis_batch_scope():
        await response_cache.async_add_cache({"id": "resp"}, **kwargs)
        assert redis_cache.store[cache_key]["response"] == {"id": "resp"}
        await flush_post_call_redis_batches()

    (direct_set,) = redis_cache.alone
    assert (direct_set[0], direct_set[1], direct_set[2]["ttl"]) == ("SET", cache_key, 120)


@pytest.mark.asyncio
async def test_a_chat_response_written_through_the_handler_dual_cache_is_in_memory_and_redis_at_once():
    redis_cache = PostCallFakeRedisCache()
    response_cache = _response_cache(redis_cache)
    handler_cache = DualCache(redis_cache=redis_cache, in_memory_cache=InMemoryCache())
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt", "ttl": 120}
    cache_key = response_cache.get_cache_key(**kwargs)

    with request_redis_batch_scope():
        await response_cache.async_add_cache('{"id": "resp"}', dynamic_cache_object=handler_cache, **kwargs)
        in_memory = await handler_cache.in_memory_cache.async_get_cache(cache_key)
        assert in_memory["response"] == '{"id": "resp"}'
        assert redis_cache.store[cache_key]["response"] == '{"id": "resp"}'
        await flush_post_call_redis_batches()

    (direct_set,) = redis_cache.alone
    assert (direct_set[0], direct_set[1], direct_set[2]["ttl"]) == ("SET", cache_key, 120)


@pytest.mark.asyncio
async def test_a_failed_slot_release_script_releases_the_slot_in_memory():
    calls: list[tuple[str, list[str], list[object]]] = []  # mutable-ok: filled by the fake script
    fakes = FakeRateLimits(parallel_release_script=_scripts(calls, Exception("ERR Lua")))
    limiter = _limiter(fakes)
    limiter.rate_limits.set_local(_K1.gauge, {"slot-1": 1.0, "slot-2": 1.0}, None)

    with request_redis_batch_scope():
        await limiter._release_stashed_parallel_slot(_slot_stash("slot-1", _K1), None, in_logging_callback=True)
        await flush_post_call_redis_batches()

    assert fakes.post_call == ["parallel_release"]
    assert limiter.rate_limits.local((_K1.gauge,)) == ({"slot-2": 1.0},)


@pytest.mark.asyncio
async def test_a_slot_released_before_the_response_reaches_redis_at_once_not_on_the_post_call_batch():
    calls: list[tuple[str, list[str], list[object]]] = []  # mutable-ok: filled by the fake script
    fakes = FakeRateLimits(parallel_release_script=_scripts(calls, [0]))
    limiter = _limiter(fakes)
    limiter.rate_limits.set_local(_K1.gauge, {"slot-1": 1.0}, None)

    with request_redis_batch_scope():
        await limiter._release_stashed_parallel_slot(_slot_stash("slot-1", _K1), None)
        assert calls == [("script", ["{api_key:k1}:max_parallel_requests"], ["slot-1"])]
        assert limiter.rate_limits.local((_K1.gauge,)) == (0,)
        await flush_post_call_redis_batches()

    assert fakes.post_call == []


@pytest.mark.asyncio
async def test_a_released_slot_is_free_locally_at_once_and_the_older_redis_count_does_not_overwrite_the_gauge():
    calls: list[tuple[str, list[str], list[object]]] = []  # mutable-ok: filled by the fake script
    fakes = FakeRateLimits(parallel_release_script=_scripts(calls, [2]))
    limiter = _limiter(fakes)
    limiter.rate_limits.set_local(_K1.gauge, {"slot-1": 1.0, "slot-2": 1.0, "slot-3": 1.0}, None)

    with request_redis_batch_scope():
        await limiter._release_stashed_parallel_slot(_slot_stash("slot-1", _K1), None, in_logging_callback=True)
        assert limiter.rate_limits.local((_K1.gauge,)) == ({"slot-2": 1.0, "slot-3": 1.0},)
        limiter.rate_limits.set_local(_K1.gauge, {"slot-2": 1.0, "slot-3": 1.0, "slot-4": 1.0}, None)
        await flush_post_call_redis_batches()

    assert calls == [("script", ["{api_key:k1}:max_parallel_requests"], ["slot-1"])]
    assert limiter.rate_limits.local((_K1.gauge,)) == ({"slot-2": 1.0, "slot-3": 1.0, "slot-4": 1.0},)


@pytest.mark.asyncio
async def test_outside_a_request_scope_the_response_cache_writes_directly():
    redis_cache = PostCallFakeRedisCache()
    response_cache = _response_cache(redis_cache)

    await response_cache.async_add_cache({"id": "resp"}, messages=[{"role": "user", "content": "hi"}], model="gpt")

    (direct_set,) = redis_cache.alone
    assert direct_set[0] == "SET"


@pytest.mark.asyncio
async def test_a_set_with_options_keeps_the_direct_path():
    redis_cache = PostCallFakeRedisCache()
    response_cache = _response_cache(redis_cache)

    with request_redis_batch_scope():
        await response_cache.async_add_cache(
            {"id": "r"}, messages=[{"role": "user", "content": "hi"}], model="gpt", nx=True
        )
        await flush_post_call_redis_batches()

    (direct_set,) = redis_cache.alone
    assert direct_set[0] == "SET" and direct_set[2]["nx"] is True


@pytest.mark.asyncio
async def test_the_success_handler_closes_the_post_call_batch_after_the_last_callback(monkeypatch):
    natives: Final = FakeNatives()
    flushes_seen_by_callbacks: list[int] = []

    class Counter(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            request = active_request_redis_batches()
            assert request is not None
            request.native_post_call()
            flushes_seen_by_callbacks.append(natives.built[0].events.count("flush_post"))

    monkeypatch.setattr(litellm, "_async_success_callback", [])
    logging_obj = LitellmLogging(
        model="test-model",
        messages=[],
        stream=False,
        call_type="completion",
        start_time=datetime.datetime.now(),
        litellm_call_id="post-call",
        function_id="post-call",
        dynamic_async_success_callbacks=[Counter(), Counter()],
    )
    logging_obj.update_environment_variables(litellm_params={"metadata": {}}, optional_params={})
    payload = {
        "id": "post-call",
        "call_type": "completion",
        "metadata": {},
        "model_group": "test-model",
        "model_parameters": {},
    }

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        await logging_obj.async_success_handler(result=ModelResponse(), standard_logging_object=payload)

    assert flushes_seen_by_callbacks == [0, 0]
    assert natives.built[0].events == ["flush_post"]


@pytest.mark.asyncio
async def test_spend_counter_increments_ride_the_native_post_call_batch(monkeypatch):
    from litellm.proxy import proxy_server

    natives = FakeNatives()
    spend = FakeSpendCounters(redis={"spend:key:k1": 7.0})
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend.store())
    pending = [PendingSpendIncrement("spend:key:k1", 0.5), PendingSpendIncrement("spend:team:t1", 0.5)]

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        await proxy_server._apply_spend_counter_increments(pending)
        declared = [(request, increments) for request, increments, _ in spend.post_call]
        assert natives.built[0].events == []
        await flush_post_call_redis_batches()

    assert declared == [(natives.built[0], (("spend:key:k1", 0.5), ("spend:team:t1", 0.5)))]
    assert natives.built[0].events == ["flush_post"]
    assert spend.commands == []


@pytest.mark.asyncio
async def test_the_request_sees_its_committed_spend_charge_settle_whoever_sends_the_batch(monkeypatch):
    from litellm.proxy import proxy_server

    spend = FakeSpendCounters(redis={})
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend.store())

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await proxy_server._apply_spend_counter_increments([PendingSpendIncrement("spend:key:k1", 0.5)])
        without_committed_spend = request.post_call_settlements
        await proxy_server._apply_spend_counter_increments(
            [PendingSpendIncrement("spend:key:k1", 0.5), PendingSpendIncrement("spend:committed:key:k1", 0.5)]
        )
        declared = [settlement.settled for settlement in request.post_call_settlements]
        spend.settle_post_call()
        sent = [settlement.settled for settlement in request.post_call_settlements]

    assert without_committed_spend == ()
    assert (declared, sent) == ([False], [True])


@pytest.mark.asyncio
async def test_native_post_call_ops_go_out_with_the_post_call_flush_not_the_request_boundary():
    natives = FakeNatives()

    with request_redis_batch_scope(post_call_deadline=60, native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        request.native_post_call()
        await request.flush_pre_call()
        at_the_boundary = list(natives.built[0].events)
        await flush_post_call_redis_batches()
    await drain_post_call_redis_batches()

    assert at_the_boundary == ["flush_pre"]
    assert natives.built[0].events == ["flush_pre", "flush_post"]


@pytest.mark.asyncio
async def test_native_post_call_ops_still_waiting_on_their_callbacks_are_drained_at_shutdown():
    natives = FakeNatives()

    with request_redis_batch_scope(post_call_deadline=60, native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        request.native_post_call()
        await request.flush_pre_call()
    await drain_post_call_redis_batches()
    await drain_post_call_redis_batches()

    assert natives.built[0].events == ["flush_pre", "flush_post"]
