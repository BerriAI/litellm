"""RedisBatch: several declared commands, exactly one pipeline execute."""

import asyncio

import pytest
import redis.exceptions
from redis.asyncio import RedisCluster

from litellm.caching.redis_batch import RedisBatch
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError

from .redis_batch_fakes import (
    FakeAsyncScript,
    FakeClusterClient,
    FakeRedisCache,
    RecordingPipeline,
    RecordingRedisClient,
    json_value,
)


def _cache(pipe: RecordingPipeline, namespace: str | None = None) -> FakeRedisCache:
    return FakeRedisCache(RecordingRedisClient([pipe]), namespace=namespace)


@pytest.mark.asyncio
async def test_mget_get_set_incrbyfloat_share_one_pipeline_and_decode_each_future():
    pipe = RecordingPipeline(results=[[json_value({"v": 1}), None], json_value(7), True, 2.5])
    batch = RedisBatch(_cache(pipe, namespace="ns"))

    mget_future = batch.mget(["a", "missing"])
    get_future = batch.get("b")
    set_future = batch.set("c", {"x": 1}, ttl=30)
    incr_future = batch.incrbyfloat("counter", 2.5, ttl=60)
    await batch.execute()

    assert pipe.execute_count == 1
    assert pipe.calls == [
        ("mget", ("ns:a", "ns:missing")),
        ("get", "ns:b"),
        ("set", "ns:c", '{"x": 1}', 30),
        ("incrbyfloat", "ns:counter", 2.5),
        ("expire", "ns:counter", 60),
    ]
    assert await mget_future == {"a": {"v": 1}, "missing": None}
    assert await get_future == 7
    assert await set_future is True
    assert await incr_future == 2.5


@pytest.mark.asyncio
async def test_a_failed_item_fails_only_its_own_future():
    pipe = RecordingPipeline(results=[redis.exceptions.ResponseError("WRONGTYPE"), json_value(3)])
    batch = RedisBatch(_cache(pipe))

    bad = batch.get("bad")
    good = batch.get("good")
    await batch.execute()

    with pytest.raises(redis.exceptions.ResponseError):
        await bad
    assert await good == 3


@pytest.mark.asyncio
async def test_a_connection_failure_fails_every_future_and_fires_the_failure_hook():
    down = ConnectionError("redis down")
    pipe = RecordingPipeline(error=down)
    cache = _cache(pipe)
    batch = RedisBatch(cache)

    futures = [batch.get("a"), batch.set("b", 1)]
    await batch.execute()

    for future in futures:
        with pytest.raises(ConnectionError) as excinfo:
            await future
        assert excinfo.value is down
    await asyncio.sleep(0)
    assert cache.service_logger_obj.failures == ["redis_batch.execute"]


@pytest.mark.asyncio
async def test_an_open_circuit_breaker_fails_every_future_without_a_pipeline():
    pipe = RecordingPipeline(results=[json_value(1)])
    cache = _cache(pipe)
    cache._circuit_breaker._failure_count = 2
    cache._circuit_breaker.record_failure(is_timeout=False)
    cache._circuit_breaker.record_failure(is_timeout=False)

    batch = RedisBatch(cache)
    future = batch.get("a")
    await batch.execute()

    with pytest.raises(RedisCircuitBreakerOpenError):
        await future
    assert pipe.calls == []
    assert pipe.execute_count == 0


@pytest.mark.asyncio
async def test_a_cluster_client_expands_mget_into_per_key_gets_and_reassembles_the_dict():
    pipe = RecordingPipeline(results=[json_value(1), None])
    client = FakeClusterClient([pipe])
    batch = RedisBatch(FakeRedisCache(client, namespace="ns"))

    future = batch.mget(["a", "b"])
    await batch.execute()

    assert pipe.calls == [("get", "ns:a"), ("get", "ns:b")]
    assert await future == {"a": 1, "b": None}


@pytest.mark.asyncio
async def test_evalsha_queues_one_evalsha_per_declared_call_and_registers_the_script():
    pipe = RecordingPipeline(results=[[5, "window"], [1, 2]])
    batch = RedisBatch(_cache(pipe))
    script = FakeAsyncScript(sha="deadbeef")

    first = batch.evalsha(script, keys=["{k}:window", "{k}:requests"], args=[10, 60])
    second = batch.evalsha(script, keys=["{j}:requests"], args=[11, 60])
    await batch.execute()

    assert pipe.calls == [
        ("evalsha", "deadbeef", 2, ("{k}:window", "{k}:requests", 10, 60)),
        ("evalsha", "deadbeef", 1, ("{j}:requests", 11, 60)),
    ]
    assert script in pipe.scripts
    assert await first == [5, "window"]
    assert await second == [1, 2]


@pytest.mark.asyncio
async def test_declaring_after_execute_is_a_programming_error():
    batch = RedisBatch(_cache(RecordingPipeline(results=[json_value(1)])))
    await batch.execute()

    with pytest.raises(RuntimeError):
        batch.get("a")


@pytest.mark.asyncio
async def test_delete_resolves_with_the_deleted_count():
    batch = RedisBatch(_cache(RecordingPipeline(results=[1])))
    future = batch.delete("a")
    await batch.execute()
    assert await future == 1


@pytest.mark.asyncio
async def test_a_cancelled_pipeline_execute_fails_every_future_and_propagates():
    pipe = RecordingPipeline(error=asyncio.CancelledError())
    batch = RedisBatch(_cache(pipe))

    futures = [batch.get("a"), batch.mget(["b", "c"]), batch.set("d", 1)]

    with pytest.raises(asyncio.CancelledError):
        await batch.execute()

    for future in futures:
        assert future.done()
        assert future.cancelled() or isinstance(future.exception(), asyncio.CancelledError)


@pytest.mark.asyncio
async def test_a_command_error_slot_logs_a_failure_event_not_a_success():
    cache = _cache(RecordingPipeline(results=[redis.exceptions.ResponseError("WRONGTYPE"), json_value(3)]))
    batch = RedisBatch(cache)

    bad = batch.get("bad")
    good = batch.get("good")
    await batch.execute()

    with pytest.raises(redis.exceptions.ResponseError):
        bad.result()
    assert await good == 3
    await asyncio.sleep(0)
    assert cache.service_logger_obj.failures == ["redis_batch.execute"]
    assert cache.service_logger_obj.successes == []


@pytest.mark.asyncio
async def test_discard_drops_the_declaration_and_cancels_its_future():
    pipe = RecordingPipeline(results=[json_value(1)])
    batch = RedisBatch(_cache(pipe))

    kept = batch.get("keep")
    dropped = batch.get("drop")
    batch.discard(dropped)
    await batch.execute()

    assert dropped.cancelled()
    assert pipe.calls == [("get", "keep")]
    assert await kept == 1
