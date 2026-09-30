"""RedisBatch: independent operations share one pipeline, each keeps its own result and failure."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import pytest
from redis.exceptions import NoScriptError

from litellm._service_logger import ServiceLogging
from litellm.caching.redis_batch import (
    RedisBatch,
    active_request_redis_batch,
    request_redis_batch_scope,
)
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker
from litellm.caching.redis_cluster_cache import RedisClusterCache

SCRIPT = "return redis.call('GET', KEYS[1])"
SHA = hashlib.sha1(SCRIPT.encode()).hexdigest()  # noqa: S324


class FakePipeline:
    def __init__(self, reply_for: Callable[[tuple[object, ...]], object], fail: Exception | None) -> None:
        self.commands: list[tuple[Any, ...]] = []
        self.reply_for = reply_for
        self.fail = fail
        self.executed = False

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def mget(self, keys: Sequence[str]) -> FakePipeline:
        self.commands.append(("MGET", *keys))
        return self

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> FakePipeline:
        self.commands.append(("EVALSHA", sha, numkeys, *keys_and_args))
        return self

    def incrbyfloat(self, name: str, amount: float) -> FakePipeline:
        self.commands.append(("INCRBYFLOAT", name, amount))
        return self

    def expire(self, name: str, time: timedelta) -> FakePipeline:
        self.commands.append(("EXPIRE", name, int(time.total_seconds())))
        return self

    def set(self, name: str, value: str, ex: timedelta | None = None) -> FakePipeline:
        self.commands.append(("SET", name, value, None if ex is None else int(ex.total_seconds())))
        return self

    def delete(self, *names: str) -> FakePipeline:
        self.commands.append(("DEL", *names))
        return self

    async def execute(self, raise_on_error: bool = True) -> list[Any]:
        assert raise_on_error is False
        self.executed = True
        if self.fail is not None:
            raise self.fail
        return [self.reply_for(command) for command in self.commands]


class FakeClient:
    def __init__(self, reply_for: Callable[[tuple[object, ...]], object], fail: Exception | None = None) -> None:
        self.pipelines: list[FakePipeline] = []
        self.reply_for = reply_for
        self.fail = fail

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        assert transaction is False
        pipe = FakePipeline(self.reply_for, self.fail)
        self.pipelines.append(pipe)
        return pipe


class FakeRedisCache(RedisCache):
    def __init__(self, client: FakeClient, namespace: str | None = None) -> None:  # super().__init__ needs a server
        self.client = client
        self.namespace = namespace
        self._circuit_breaker = RedisCircuitBreaker(failure_threshold=5, recovery_timeout=30)
        self.service_logger_obj = ServiceLogging()
        self.default_ttl = None
        self.alone: list[tuple[str, Any]] = []
        self.store: dict[str, Any] = {}

    def init_async_client(self) -> FakeClient:  # pyright: ignore[reportIncompatibleMethodOverride]  # fake client, no server
        return self.client

    async def async_batch_get_cache(self, key_list: Sequence[str], **kwargs: object) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]  # records the direct read
        self.alone.append(("MGET", tuple(key_list)))
        return {key: self.store.get(key) for key in key_list}

    async def async_increment(self, key: str, value: float, ttl: int | None = None, **kwargs: object) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]  # records the direct write
        self.alone.append(("INCRBYFLOAT", key, value))
        self.store[key] = float(self.store.get(key, 0.0)) + value
        return self.store[key]

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # fake, no server
        self.alone.append(("SET", key, value))
        self.store[key] = value

    async def async_delete_cache(self, key: str) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # records the direct delete
        self.alone.append(("DEL", key))
        self.store.pop(key, None)

    async def async_set_cache_pipeline_with_ttls(self, cache_list: Sequence[tuple[str, object, float | None]]) -> None:
        self.alone.append(("SET_PIPELINE", tuple(cache_list)))
        for key, value, _ttl in cache_list:
            self.store[key] = value


class FakeClusterCache(RedisClusterCache, FakeRedisCache):
    def __init__(self, client: FakeClient) -> None:  # super().__init__ needs a server
        FakeRedisCache.__init__(self, client)


def replies(command: tuple[Any, ...]) -> Any:
    match command[0]:
        case "MGET":
            return [json.dumps({"k": key}) if key.endswith("hit") else None for key in command[1:]]
        case "EVALSHA":
            return [1, 2]
        case "INCRBYFLOAT":
            return b"3.5"
        case "EXPIRE":
            return 1
        case "SET":
            return True
        case "DEL":
            return 1
    raise AssertionError(command)


def make(fail: Exception | None = None, namespace: str | None = None) -> tuple[FakeRedisCache, FakeClient]:
    client = FakeClient(replies, fail)
    return FakeRedisCache(client, namespace), client


async def run_alone_script(keys: Sequence[str], args: Sequence[Any]) -> object:
    return ["alone", *keys, *args]


@pytest.mark.asyncio
async def test_one_pipeline_carries_every_declared_operation_and_awaiting_one_flushes_all() -> None:
    cache, client = make(namespace="ns")
    batch = RedisBatch(cache)
    got = batch.mget(["a:hit", "b", "a:hit"])
    script = batch.script(SCRIPT, run_alone_script, ["w"], [7, "x"])
    incr = batch.increment("cnt", 2.5, ttl=60)
    plain = batch.increment("cnt2", 1)
    assert client.pipelines == []

    assert await got == {"a:hit": {"k": "ns:a:hit"}, "b": None}
    assert script.done and incr.done and plain.done
    assert await script == [1, 2]
    assert await incr == 3.5
    assert await plain == 3.5
    assert batch.flushes == 1
    assert [pipe.commands for pipe in client.pipelines] == [
        [
            ("MGET", "ns:a:hit", "ns:b"),
            ("EVALSHA", SHA, 1, "ns:w", 7, "x"),
            ("INCRBYFLOAT", "ns:cnt", 2.5),
            ("EXPIRE", "ns:cnt", 60),
            ("INCRBYFLOAT", "ns:cnt2", 1),
        ]
    ]
    assert cache.alone == []


@pytest.mark.asyncio
async def test_operations_declared_after_a_flush_go_out_in_the_next_pipeline() -> None:
    cache, client = make()
    batch = RedisBatch(cache)
    await batch.mget(["a"])
    later = batch.increment("cnt", 1)
    assert not later.done
    assert await later == 3.5
    assert batch.flushes == 2
    assert [pipe.commands for pipe in client.pipelines] == [[("MGET", "a")], [("INCRBYFLOAT", "cnt", 1)]]


@pytest.mark.asyncio
async def test_a_failing_reply_fails_only_its_own_operation() -> None:
    def reply_for(command: tuple[Any, ...]) -> Any:
        if command[0] == "EVALSHA":
            return ValueError("script blew up")
        return replies(command)

    client = FakeClient(reply_for)
    cache = FakeRedisCache(client)
    batch = RedisBatch(cache)
    got = batch.mget(["a:hit"])
    script = batch.script(SCRIPT, run_alone_script, ["w"], [])
    assert await got == {"a:hit": {"k": "a:hit"}}
    with pytest.raises(ValueError, match="script blew up"):
        await script
    assert cache.alone == []


@pytest.mark.asyncio
async def test_a_reply_an_operation_cannot_decode_fails_only_that_operation() -> None:
    def reply_for(command: tuple[Any, ...]) -> Any:
        if command[0] == "MGET":
            return "not-a-list"
        return replies(command)

    client = FakeClient(reply_for)
    cache = FakeRedisCache(client)
    batch = RedisBatch(cache)
    got = batch.mget(["a:hit"])
    written = batch.set("w", {"k": 1})
    script = batch.script(SCRIPT, run_alone_script, ["w"], [])
    with pytest.raises(TypeError, match="MGET reply is not a list"):
        await got
    assert await written is None
    assert await script == [1, 2]
    assert len(client.pipelines) == 1


@pytest.mark.asyncio
async def test_pipeline_failure_fails_every_operation_and_trips_the_breaker() -> None:
    cache, _client = make(fail=ConnectionError("redis down"))
    batch = RedisBatch(cache)
    got = batch.mget(["a"])
    incr = batch.increment("cnt", 1)
    with pytest.raises(ConnectionError):
        await got
    with pytest.raises(ConnectionError):
        await incr
    assert cache._circuit_breaker._failure_count == 1  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_noscript_reply_reruns_that_script_through_the_registered_executor() -> None:
    def reply_for(command: tuple[Any, ...]) -> Any:
        if command[0] == "EVALSHA":
            return NoScriptError("NOSCRIPT No matching script")
        return replies(command)

    client = FakeClient(reply_for)
    cache = FakeRedisCache(client)
    batch = RedisBatch(cache)
    script = batch.script(SCRIPT, run_alone_script, ["w"], [1])
    incr = batch.increment("cnt", 1)
    assert await script == ["alone", "w", 1]
    assert await incr == 3.5
    assert batch.flushes == 1


@pytest.mark.asyncio
async def test_cluster_cache_runs_each_operation_on_its_own_path() -> None:
    client = FakeClient(replies)
    cache = FakeClusterCache(client)
    cache.store["a"] = 4
    batch = RedisBatch(cache)
    got = batch.mget(["a", "b"])
    incr = batch.increment("cnt", 2)
    assert await got == {"a": 4, "b": None}
    assert await incr == 2.0
    assert client.pipelines == []
    assert cache.alone == [("MGET", ("a", "b")), ("INCRBYFLOAT", "cnt", 2)]


@pytest.mark.asyncio
async def test_flush_hook_lets_a_lazy_reader_join_the_pipeline_that_is_going_out() -> None:
    cache, client = make()
    batch = RedisBatch(cache)
    joined: list[Any] = []
    batch.add_flush_hook(lambda: joined.append(batch.mget(["late"])))
    await batch.mget(["early"])
    assert len(joined) == 1 and joined[0].done
    assert await joined[0] == {"late": None}
    assert [pipe.commands for pipe in client.pipelines] == [[("MGET", "early"), ("MGET", "late")]]


@pytest.mark.asyncio
async def test_concurrent_awaiters_share_one_flush() -> None:
    cache, client = make()
    batch = RedisBatch(cache)
    first = batch.mget(["a"])
    second = batch.mget(["b"])
    results = await asyncio.gather(first._wait(), second._wait())  # pyright: ignore[reportPrivateUsage]
    assert results == [{"a": None}, {"b": None}]
    assert batch.flushes == 1
    assert len(client.pipelines) == 1


def test_request_scope_hands_out_one_batch_per_backend_and_nests() -> None:
    cache_a, _ = make()
    cache_b, _ = make()
    assert active_request_redis_batch(cache_a) is None
    with request_redis_batch_scope() as batches:
        first = active_request_redis_batch(cache_a)
        assert first is not None
        assert active_request_redis_batch(cache_a) is first
        assert active_request_redis_batch(cache_b) is not first
        with request_redis_batch_scope() as inner:
            assert inner is batches
            assert active_request_redis_batch(cache_a) is first
        assert active_request_redis_batch(cache_a) is first
        assert len(batches.batches) == 2
    assert active_request_redis_batch(cache_a) is None


@pytest.mark.asyncio
async def test_a_key_an_mget_read_as_absent_stays_known_missing_until_something_sets_it() -> None:
    cache, client = make()
    batch = RedisBatch(cache)
    values = await batch.mget(["a-hit", "b-miss"])
    assert values == {"a-hit": {"k": "a-hit"}, "b-miss": None}
    assert batch.read_as_missing("b-miss") is True
    assert batch.read_as_missing("a-hit") is False
    assert batch.read_as_missing("never-read") is False
    batch.set("b-miss", "now-present")
    assert batch.read_as_missing("b-miss") is False


@pytest.mark.asyncio
async def test_a_delete_rides_the_pipeline_under_the_namespace_and_reads_as_missing_afterwards() -> None:
    cache, client = make(namespace="ns")
    batch = RedisBatch(cache)
    gone = batch.delete("team_alias:x")
    got = batch.mget(["a-hit"])
    assert await gone is None
    assert await got == {"a-hit": {"k": "ns:a-hit"}}
    assert len(client.pipelines) == 1
    assert client.pipelines[0].commands[0] == ("DEL", "ns:team_alias:x")
    assert batch.read_as_missing("team_alias:x") is True
    assert cache.alone == []


@pytest.mark.asyncio
async def test_a_delete_on_a_cluster_cache_runs_as_its_own_del() -> None:
    client = FakeClient(replies)
    cache = FakeClusterCache(client)
    cache.store["team_alias:x"] = "stale"
    batch = RedisBatch(cache)
    assert await batch.delete("team_alias:x") is None
    assert cache.alone == [("DEL", "team_alias:x")]
    assert "team_alias:x" not in cache.store
    assert client.pipelines == []


@pytest.mark.asyncio
async def test_a_failed_mget_marks_nothing_as_missing() -> None:
    cache, client = make(fail=ConnectionError("down"))
    batch = RedisBatch(cache)
    with pytest.raises(ConnectionError):
        await batch.mget(["b-miss"])
    assert batch.read_as_missing("b-miss") is False
