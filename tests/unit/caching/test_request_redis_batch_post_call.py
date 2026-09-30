"""One Redis pipeline per backend for the post-call writes of a request: spend counters, rate-limit token
scripts and slot releases, deployment TPM and the response-cache SET all ride the post-call batch, which
goes out once the success/failure callbacks have run (or at the deadline when no callback phase closes it)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_batch import (
    active_post_call_redis_batch,
    active_request_redis_batches,
    drain_post_call_redis_batches,
    flush_post_call_redis_batches,
    request_redis_batch_scope,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PARALLEL_RELEASE_SCRIPT,
    TOKEN_INCREMENT_SCRIPT,
    ParallelSlotAcquisition,
    RequestRateLimiterStash,
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.spend_tracking.spend_counter_batch import PendingSpendIncrement
from litellm.proxy.utils import InternalUsageCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import ModelResponse

from .test_redis_batch import FakeClient, FakeRedisCache


async def _script_outside_the_pipeline(keys: Sequence[str], args: Sequence[object]) -> object:
    raise AssertionError("post-call scripts must ride the post-call pipeline")


class PostCallFakeRedisCache(FakeRedisCache):
    """Records the direct (non-pipelined) writes an owner falls back to."""

    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]:
        return _script_outside_the_pipeline

    async def async_increment_pipeline(
        self, increment_list: list[RedisPipelineIncrementOperation], **kwargs: object
    ) -> list[float]:
        return [await self.async_increment(op["key"], op["increment_value"]) for op in increment_list]

    async def async_delete_cache(self, key: str, **kwargs: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # the fake drops RedisCache's unused kwargs
        self.alone.append(("DEL", key))
        self.store.pop(key, None)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.alone.append(("SET", key, dict(kwargs)))
        self.store[key] = value


def sha_of(script: str) -> str:
    return hashlib.sha1(script.encode()).hexdigest()  # noqa: S324


def _ok_replies(command: tuple[object, ...]) -> object:
    match command[0]:
        case "INCRBYFLOAT":
            return b"7.5"
        case "EXPIRE":
            return 1
        case "SET":
            return True
        case "EVALSHA":
            return [3, 0]
        case "MGET":
            return [json.dumps({"spend": 1.0}) for _ in command[1:]]
    raise AssertionError(command)


async def _run_ready_callbacks(client: FakeClient) -> None:
    for _ in range(20):
        if client.pipelines:
            return
        await asyncio.sleep(0)


def _names(client: FakeClient, index: int = 0) -> list[str]:
    return [command[0] for command in client.pipelines[index].commands]


def _limiter(redis_cache: FakeRedisCache) -> _PROXY_MaxParallelRequestsHandler_v3:
    dual_cache = DualCache()
    dual_cache.attach_redis_cache(redis_cache)
    return _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(dual_cache=dual_cache))


def _slot_stash(slot_id: str, *counter_keys: str) -> RequestRateLimiterStash:
    return RequestRateLimiterStash(parallel_slot=ParallelSlotAcquisition(slot_id=slot_id, counter_keys=list(counter_keys)))


def _token_ops(*keys: str) -> list[RedisPipelineIncrementOperation]:
    return [RedisPipelineIncrementOperation(key=key, increment_value=10, ttl=60) for key in keys]


def _response_cache(redis_cache: FakeRedisCache) -> Cache:
    cache = Cache(type="local")
    cache.type = "redis"  # pyright: ignore[reportAttributeAccessIssue]  # the fake stands in for the Redis backend
    cache.cache = redis_cache
    return cache


def _tpm_router(redis_cache: FakeRedisCache) -> tuple[LowestTPMLoggingHandler_v2, DualCache]:
    router_cache = DualCache()
    router_cache.attach_redis_cache(redis_cache)
    return LowestTPMLoggingHandler_v2(router_cache=router_cache, routing_args={"ttl": 60}), router_cache


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
async def test_every_post_call_owner_rides_one_pipeline_that_goes_out_when_the_callbacks_are_done():
    client = FakeClient(_ok_replies)
    redis_cache = PostCallFakeRedisCache(client)
    limiter = _limiter(redis_cache)
    response_cache = _response_cache(redis_cache)
    tpm, router_cache = _tpm_router(redis_cache)

    with request_redis_batch_scope():
        await response_cache.async_add_cache(
            {"id": "resp"}, messages=[{"role": "user", "content": "hi"}], model="gpt", ttl=120
        )
        await tpm.async_log_success_event(_tpm_kwargs(), None, None, None)
        await limiter.async_increment_tokens_with_ttl_preservation(_token_ops("{api_key:k1}:tokens"))
        await limiter._release_stashed_parallel_slot(
            _slot_stash("slot-1", "{api_key:k1}:parallel"), None, in_logging_callback=True
        )
        assert client.pipelines == []  # nothing goes out while the callbacks are still declaring
        await flush_post_call_redis_batches()

    assert len(client.pipelines) == 1
    assert _names(client) == ["SET", "INCRBYFLOAT", "EXPIRE", "EVALSHA", "EVALSHA"]
    evalshas = [c for c in client.pipelines[0].commands if c[0] == "EVALSHA"]
    assert [c[1] for c in evalshas] == [sha_of(TOKEN_INCREMENT_SCRIPT), sha_of(PARALLEL_RELEASE_SCRIPT)]
    assert redis_cache.alone == []
    assert (
        await router_cache.in_memory_cache.async_get_cache(
            next(k for k in router_cache.in_memory_cache.cache_dict if ":tpm:" in k)
        )
        == 42
    )


@pytest.mark.asyncio
async def test_the_response_cache_write_is_the_same_set_the_direct_path_issues():
    client = FakeClient(_ok_replies)
    redis_cache = PostCallFakeRedisCache(client)
    response_cache = _response_cache(redis_cache)
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt", "ttl": 120}

    with request_redis_batch_scope():
        await response_cache.async_add_cache({"id": "resp"}, **kwargs)
        await flush_post_call_redis_batches()

    cache_key = response_cache.get_cache_key(**kwargs)
    (command,) = client.pipelines[0].commands
    assert (command[0], command[1], command[3]) == ("SET", cache_key, 120)
    assert json.loads(command[2])["response"] == {"id": "resp"}


@pytest.mark.asyncio
async def test_a_chat_response_written_through_the_handler_dual_cache_lands_in_memory_and_rides_the_pipeline():
    client = FakeClient(_ok_replies)
    redis_cache = PostCallFakeRedisCache(client)
    response_cache = _response_cache(redis_cache)
    handler_cache = DualCache(redis_cache=redis_cache, in_memory_cache=InMemoryCache())
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt", "ttl": 120}

    with request_redis_batch_scope():
        await response_cache.async_add_cache('{"id": "resp"}', dynamic_cache_object=handler_cache, **kwargs)
        cache_key = response_cache.get_cache_key(**kwargs)
        in_memory = await handler_cache.in_memory_cache.async_get_cache(cache_key)
        assert in_memory["response"] == '{"id": "resp"}'
        assert redis_cache.alone == []
        await flush_post_call_redis_batches()

    (command,) = client.pipelines[0].commands
    assert (command[0], command[1], command[3]) == ("SET", cache_key, 120)


@pytest.mark.asyncio
async def test_a_failed_operation_fails_only_its_owner_and_the_owner_applies_its_own_fallback():
    def replies(command: tuple[object, ...]) -> object:
        if command[0] == "EVALSHA" and command[3] == "{api_key:k1}:tokens":
            return Exception("ERR Lua")
        return _ok_replies(command)

    client = FakeClient(replies)
    redis_cache = PostCallFakeRedisCache(client)
    limiter = _limiter(redis_cache)
    response_cache = _response_cache(redis_cache)

    with request_redis_batch_scope():
        await response_cache.async_add_cache({"id": "resp"}, messages=[{"role": "user", "content": "hi"}], model="gpt")
        await limiter.async_increment_tokens_with_ttl_preservation(_token_ops("{api_key:k1}:tokens"))
        await limiter.async_increment_tokens_with_ttl_preservation(_token_ops("{team:t1}:tokens"))
        await flush_post_call_redis_batches()

    assert len(client.pipelines) == 1
    # the failed group falls back to the plain increment (memory + Redis), the healthy group does not
    assert redis_cache.alone == [("INCRBYFLOAT", "{api_key:k1}:tokens", 10)]
    assert await limiter.internal_usage_cache.dual_cache.in_memory_cache.async_get_cache("{api_key:k1}:tokens") == 10
    assert await limiter.internal_usage_cache.dual_cache.in_memory_cache.async_get_cache("{team:t1}:tokens") is None


@pytest.mark.asyncio
async def test_a_failed_slot_release_script_releases_the_slot_in_memory():
    def replies(command: tuple[object, ...]) -> object:
        if command[0] == "EVALSHA":
            return Exception("ERR Lua")
        return _ok_replies(command)

    redis_cache = PostCallFakeRedisCache(FakeClient(replies))
    limiter = _limiter(redis_cache)
    memory = limiter.internal_usage_cache.dual_cache.in_memory_cache
    await memory.async_set_cache("{api_key:k1}:parallel", {"slot-1": 1.0, "slot-2": 1.0})

    with request_redis_batch_scope():
        await limiter._release_stashed_parallel_slot(
            _slot_stash("slot-1", "{api_key:k1}:parallel"), None, in_logging_callback=True
        )
        await flush_post_call_redis_batches()

    assert await memory.async_get_cache("{api_key:k1}:parallel") == {"slot-2": 1.0}


class DirectScriptFakeRedisCache(PostCallFakeRedisCache):
    """Records the release script a pre-response caller runs outside the pipeline."""

    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]:
        async def run(keys: Sequence[str], args: Sequence[object]) -> object:
            self.alone.append(("EVALSHA", tuple(keys), tuple(args)))
            return [0 for _ in keys]

        return run


@pytest.mark.asyncio
async def test_a_slot_released_before_the_response_reaches_redis_at_once_not_on_the_pipeline():
    client = FakeClient(_ok_replies)
    redis_cache = DirectScriptFakeRedisCache(client)
    limiter = _limiter(redis_cache)
    memory = limiter.internal_usage_cache.dual_cache.in_memory_cache
    await memory.async_set_cache("{api_key:k1}:parallel", {"slot-1": 1.0})

    with request_redis_batch_scope():
        await limiter._release_stashed_parallel_slot(_slot_stash("slot-1", "{api_key:k1}:parallel"), None)
        assert redis_cache.alone == [("EVALSHA", ("{api_key:k1}:parallel",), ("slot-1",))]
        assert await memory.async_get_cache("{api_key:k1}:parallel") == 0
        await flush_post_call_redis_batches()

    assert client.pipelines == []


@pytest.mark.asyncio
async def test_a_deferred_response_cache_set_without_a_ttl_expires_in_redis_like_the_direct_path():
    client = FakeClient(_ok_replies)
    redis_cache = PostCallFakeRedisCache(client)
    dual_cache = DualCache(redis_cache=redis_cache, in_memory_cache=InMemoryCache(), default_in_memory_ttl=300)

    await dual_cache.async_set_cache("direct", {"id": "resp"})
    with request_redis_batch_scope():
        await dual_cache.async_set_cache_post_call("deferred", {"id": "resp"}, None)
        await flush_post_call_redis_batches()

    (command,) = client.pipelines[0].commands
    assert (command[0], command[1], command[3]) == ("SET", "deferred", redis_cache.alone[0][2]["ttl"])
    assert command[3] == 300


@pytest.mark.asyncio
async def test_a_released_slot_is_free_locally_at_once_and_the_older_redis_count_does_not_overwrite_the_gauge():
    def replies(command: tuple[object, ...]) -> object:
        if command[0] == "EVALSHA":
            return [2]
        return _ok_replies(command)

    redis_cache = PostCallFakeRedisCache(FakeClient(replies))
    limiter = _limiter(redis_cache)
    memory = limiter.internal_usage_cache.dual_cache.in_memory_cache
    await memory.async_set_cache("{api_key:k1}:parallel", {"slot-1": 1.0, "slot-2": 1.0, "slot-3": 1.0})

    with request_redis_batch_scope():
        await limiter._release_stashed_parallel_slot(
            _slot_stash("slot-1", "{api_key:k1}:parallel"), None, in_logging_callback=True
        )
        assert await memory.async_get_cache("{api_key:k1}:parallel") == {"slot-2": 1.0, "slot-3": 1.0}
        await memory.async_set_cache("{api_key:k1}:parallel", {"slot-2": 1.0, "slot-3": 1.0, "slot-4": 1.0})
        await flush_post_call_redis_batches()

    assert await memory.async_get_cache("{api_key:k1}:parallel") == {"slot-2": 1.0, "slot-3": 1.0, "slot-4": 1.0}


@pytest.mark.asyncio
async def test_failure_refunds_ride_the_post_call_pipeline_and_count_in_memory_at_once():
    client = FakeClient(_ok_replies)
    dual_cache = DualCache()
    dual_cache.attach_redis_cache(PostCallFakeRedisCache(client))
    refund = [RedisPipelineIncrementOperation(key="{api_key:k1}:tokens", increment_value=-500, ttl=60)]

    with request_redis_batch_scope():
        await dual_cache.async_increment_cache_pipeline_post_call(refund)
        assert await dual_cache.in_memory_cache.async_get_cache("{api_key:k1}:tokens") == -500
        assert client.pipelines == []
        await flush_post_call_redis_batches()

    assert client.pipelines[0].commands[0] == ("INCRBYFLOAT", "{api_key:k1}:tokens", -500)


@pytest.mark.asyncio
async def test_outside_a_request_scope_owners_write_directly_as_before():
    client = FakeClient(_ok_replies)
    redis_cache = PostCallFakeRedisCache(client)
    dual_cache = DualCache()
    dual_cache.attach_redis_cache(redis_cache)
    response_cache = _response_cache(redis_cache)

    await dual_cache.async_increment_cache_post_call("dep:tpm", 42, ttl=60)
    await response_cache.async_add_cache({"id": "resp"}, messages=[{"role": "user", "content": "hi"}], model="gpt")

    assert client.pipelines == []
    assert redis_cache.alone[0] == ("INCRBYFLOAT", "dep:tpm", 42)
    assert active_post_call_redis_batch(redis_cache) is None


@pytest.mark.asyncio
async def test_a_set_with_options_keeps_the_direct_path():
    client = FakeClient(_ok_replies)
    redis_cache = PostCallFakeRedisCache(client)
    response_cache = _response_cache(redis_cache)

    with request_redis_batch_scope():
        await response_cache.async_add_cache(
            {"id": "r"}, messages=[{"role": "user", "content": "hi"}], model="gpt", nx=True
        )
        await flush_post_call_redis_batches()

    assert client.pipelines == []
    (direct_set,) = redis_cache.alone
    assert direct_set[0] == "SET" and direct_set[2]["nx"] is True


@pytest.mark.asyncio
async def test_two_backends_get_one_post_call_pipeline_each():
    a_client, b_client = FakeClient(_ok_replies), FakeClient(_ok_replies)
    a, b = DualCache(), DualCache()
    a.attach_redis_cache(PostCallFakeRedisCache(a_client))
    b.attach_redis_cache(PostCallFakeRedisCache(b_client))

    with request_redis_batch_scope():
        await a.async_increment_cache_post_call("x", 1, ttl=None)
        await b.async_increment_cache_post_call("y", 1, ttl=None)
        await a.async_increment_cache_post_call("z", 1, ttl=None)
        await flush_post_call_redis_batches()

    assert len(a_client.pipelines) == 1 and len(b_client.pipelines) == 1
    assert [c[1] for c in a_client.pipelines[0].commands if c[0] == "INCRBYFLOAT"] == ["x", "z"]


@pytest.mark.asyncio
async def test_a_numeric_string_ttl_reaches_redis_as_the_direct_path_would_send_it():
    client = FakeClient(_ok_replies)
    response_cache = _response_cache(PostCallFakeRedisCache(client))
    kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt", "ttl": "3600"}

    with request_redis_batch_scope():
        await response_cache.async_add_cache({"id": "resp"}, **kwargs)
        await flush_post_call_redis_batches()

    (command,) = client.pipelines[0].commands
    assert (command[0], command[3]) == ("SET", 3600)


@pytest.mark.asyncio
async def test_post_call_writes_still_waiting_on_their_callbacks_are_drained_at_shutdown():
    client = FakeClient(_ok_replies)
    dual_cache = DualCache()
    dual_cache.attach_redis_cache(PostCallFakeRedisCache(client))

    with request_redis_batch_scope(post_call_deadline=60) as request:
        await dual_cache.async_increment_cache_post_call("x", 1, ttl=None)
        await request.flush_all()
    assert client.pipelines == []

    await drain_post_call_redis_batches()
    assert len(client.pipelines) == 1 and _names(client) == ["INCRBYFLOAT"]

    await drain_post_call_redis_batches()
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_a_post_call_batch_nobody_closes_goes_out_at_the_deadline(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient(_ok_replies)
    dual_cache = DualCache()
    dual_cache.attach_redis_cache(PostCallFakeRedisCache(client))

    loop = asyncio.get_running_loop()
    armed_at = loop.time()

    with request_redis_batch_scope(post_call_deadline=60) as request:
        await dual_cache.async_increment_cache_post_call("x", 1, ttl=None)
        await request.flush_all()
        await _run_ready_callbacks(client)
        assert client.pipelines == [], "the request boundary drains the immediate batch, not the post-call one"

        monkeypatch.setattr(loop, "time", lambda: armed_at + 61)
        await _run_ready_callbacks(client)

    assert len(client.pipelines) == 1 and _names(client) == ["INCRBYFLOAT"]


@pytest.mark.asyncio
async def test_the_success_handler_closes_the_post_call_batch_after_the_last_callback(monkeypatch):
    client = FakeClient(_ok_replies)
    dual_cache = DualCache()
    dual_cache.attach_redis_cache(PostCallFakeRedisCache(client))
    pipelines_seen_by_callbacks: list[int] = []

    class Counter(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            await dual_cache.async_increment_cache_post_call("counted", 1, ttl=None)
            pipelines_seen_by_callbacks.append(len(client.pipelines))

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

    with request_redis_batch_scope():
        await logging_obj.async_success_handler(result=ModelResponse(), standard_logging_object=payload)

    assert pipelines_seen_by_callbacks == [0, 0]
    assert len(client.pipelines) == 1 and _names(client) == ["INCRBYFLOAT", "INCRBYFLOAT"]


@pytest.mark.asyncio
async def test_spend_counter_increments_ride_the_pipeline_and_settle_into_memory(monkeypatch):
    from litellm.proxy import proxy_server

    client = FakeClient(_ok_replies)
    spend_cache = DualCache()
    spend_cache.attach_redis_cache(PostCallFakeRedisCache(client))
    monkeypatch.setattr(proxy_server, "spend_counter_cache", spend_cache)
    pending = [PendingSpendIncrement("spend:key:k1", 0.5), PendingSpendIncrement("spend:team:t1", 0.5)]

    with request_redis_batch_scope():
        await proxy_server._apply_spend_counter_increments(pending)
        assert client.pipelines == []
        await flush_post_call_redis_batches()

    assert [c for c in client.pipelines[0].commands if c[0] == "INCRBYFLOAT"] == [
        ("INCRBYFLOAT", "spend:key:k1", 0.5),
        ("INCRBYFLOAT", "spend:team:t1", 0.5),
    ]
    assert spend_cache.in_memory_cache.get_cache("spend:key:k1") == 7.5


@pytest.mark.asyncio
async def test_a_spend_counter_whose_increment_failed_is_invalidated_not_trusted(monkeypatch):
    from litellm.proxy import proxy_server

    def replies(command: tuple[object, ...]) -> object:
        if command[0] == "INCRBYFLOAT" and command[1] == "spend:key:k1":
            return Exception("OOM")
        return _ok_replies(command)

    redis_cache = PostCallFakeRedisCache(FakeClient(replies))
    spend_cache = DualCache()
    spend_cache.attach_redis_cache(redis_cache)
    spend_cache.in_memory_cache.set_cache("spend:key:k1", 3.0)
    spend_cache.in_memory_cache.set_cache("spend:team:t1", 3.0)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", spend_cache)

    with request_redis_batch_scope():
        await proxy_server._apply_spend_counter_increments(
            [PendingSpendIncrement("spend:key:k1", 0.5), PendingSpendIncrement("spend:team:t1", 0.5)]
        )
        await flush_post_call_redis_batches()

    assert spend_cache.in_memory_cache.get_cache("spend:key:k1") is None
    assert redis_cache.alone == [("DEL", "spend:key:k1")]
    assert spend_cache.in_memory_cache.get_cache("spend:team:t1") == 7.5


@pytest.mark.asyncio
async def test_a_cancelled_post_call_flush_keeps_the_shared_spend_counter_and_counts_the_spend_locally(monkeypatch):
    from litellm.proxy import proxy_server

    redis_cache = PostCallFakeRedisCache(
        FakeClient(_ok_replies, fail=asyncio.CancelledError())  # pyright: ignore[reportArgumentType]  # a cancel raised mid-pipeline
    )
    spend_cache = DualCache()
    spend_cache.attach_redis_cache(redis_cache)
    spend_cache.in_memory_cache.set_cache("spend:key:k1", 3.0)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", spend_cache)

    with request_redis_batch_scope():
        await proxy_server._apply_spend_counter_increments(
            [PendingSpendIncrement("spend:key:k1", 0.5), PendingSpendIncrement("spend:team:t1", 0.5)]
        )
        with pytest.raises(asyncio.CancelledError):
            await flush_post_call_redis_batches()

    assert redis_cache.alone == [], "a cancel says nothing about the shared counter, so Redis keeps it"
    assert spend_cache.in_memory_cache.get_cache("spend:key:k1") == 3.5, "the local copy counts the cancelled spend"
    assert spend_cache.in_memory_cache.get_cache("spend:team:t1") is None, "an absent local copy is not seeded"


@pytest.mark.asyncio
async def test_the_update_cache_read_armed_before_accounting_rides_the_pipeline_of_the_reconcile_read():
    from litellm.proxy.proxy_server import _read_update_cache_values, arm_update_cache_read

    client = FakeClient(_ok_replies)
    redis_cache = PostCallFakeRedisCache(client)
    cache = DualCache()
    cache.attach_redis_cache(redis_cache)
    keys = ["user-1", "team_id:t1"]

    with request_redis_batch_scope() as request:
        await arm_update_cache_read(keys, cache=cache)
        assert client.pipelines == []
        await request.batch(redis_cache).mget(["spend:key:k1"])  # the spend reconcile read of the same request
        values = await _read_update_cache_values(keys, None, cache=cache)

    assert len(client.pipelines) == 1
    assert client.pipelines[0].commands == [("MGET", "user-1", "team_id:t1"), ("MGET", "spend:key:k1")]
    assert values == {"user-1": {"spend": 1.0}, "team_id:t1": {"spend": 1.0}}
    assert redis_cache.alone == []
    assert active_request_redis_batches() is None


@pytest.mark.asyncio
async def test_an_update_cache_read_armed_for_other_keys_is_ignored_and_the_read_happens_as_before():
    from litellm.proxy.proxy_server import _read_update_cache_values, arm_update_cache_read

    redis_cache = PostCallFakeRedisCache(FakeClient(_ok_replies))
    redis_cache.store["team_id:t1"] = {"spend": 2.0}
    cache = DualCache()
    cache.attach_redis_cache(redis_cache)

    with request_redis_batch_scope():
        await arm_update_cache_read(["user-1"], cache=cache)
        values = await _read_update_cache_values(["team_id:t1"], None, cache=cache)

    assert values == {"team_id:t1": {"spend": 2.0}}
    assert ("MGET", ("team_id:t1",)) in redis_cache.alone


@pytest.mark.asyncio
async def test_the_update_cache_read_sees_a_cached_spend_written_while_the_spend_was_persisted(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.proxy_track_cost_callback import _update_database_and_spend_counters

    cached_user_spend = {"user-1": 1.0}

    def replies(command: tuple[object, ...]) -> object:
        if command[0] == "MGET":
            return [
                json.dumps({"spend": cached_user_spend[key]}) if key in cached_user_spend else b"0.5"
                for key in command[1:]
            ]
        return _ok_replies(command)

    client = FakeClient(replies)
    redis_cache = PostCallFakeRedisCache(client)
    spend_cache = DualCache()
    spend_cache.attach_redis_cache(redis_cache)
    user_cache = DualCache()
    user_cache.attach_redis_cache(redis_cache)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", spend_cache)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", user_cache)

    async def _read_on_the_request_pipeline_then_a_concurrent_callback_writes_the_user(**kwargs: object) -> bool:
        request = active_request_redis_batches()
        assert request is not None
        await request.batch(redis_cache).mget(["key-object"])
        cached_user_spend["user-1"] = 5.0
        return True

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock(
        side_effect=_read_on_the_request_pipeline_then_a_concurrent_callback_writes_the_user
    )
    reservation = {
        "reserved_cost": 0.5,
        "entries": [
            {
                "counter_key": "spend:key:k1",
                "entity_type": "Key",
                "entity_id": "k1",
                "reserved_cost": 0.5,
                "applied_adjustment": 0.0,
            }
        ],
        "finalized": False,
    }

    with request_redis_batch_scope():
        charged = await _update_database_and_spend_counters(
            proxy_logging_obj=proxy_logging_obj,
            increment_spend_counters=proxy_server.increment_spend_counters,
            user_api_key="k1",
            user_id="user-1",
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={},
            completion_response=None,
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
            response_cost=0.2,
            budget_reservation=reservation,
            update_cache_read_keys=("user-1",),
        )
        values = await proxy_server._read_update_cache_values(("user-1",), None)

    assert charged is True
    assert values == {"user-1": {"spend": 5.0}}, client.pipelines
