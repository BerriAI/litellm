import asyncio
import logging
import threading
from collections.abc import Sequence
from typing import Final

import pytest
from fakeredis import TcpFakeServer

from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.connection import connect
from litellm.redis.stores.router_usage import CounterIncrement, RouterUsageStore, UsageCounter
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy
from litellm.rust_bridge._native import NativeRedisStoreError
from tests.unit.redis.stores.router_usage_double import FakeUsageHandle, usage_store

_A: Final = UsageCounter.for_deployment("dep-a", "openai/gpt-5.4-nano", "rpm", "12-34")
_B: Final = UsageCounter.for_deployment("dep-b", "openai/gpt-5.4-nano", "rpm", "12-34")


def _strategy(handle: FakeUsageHandle) -> BaseRoutingStrategy:
    return BaseRoutingStrategy(usage=usage_store(handle), should_batch_redis_writes=False, default_sync_interval=1)


@pytest.mark.asyncio
async def test_an_increment_counts_in_memory_now_and_queues_the_redis_write():
    handle: Final = FakeUsageHandle(redis={})
    strategy: Final = _strategy(handle)

    counted: Final = await strategy._increment_value_in_current_window(_A, 1, 60)  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    assert counted == 1
    assert handle.local == {_A: 1}
    assert handle.redis == {}
    assert strategy.redis_increment_operation_queue == [CounterIncrement(_A, 1, 60)]
    assert strategy.get_in_memory_keys_to_update() == {_A}


@pytest.mark.asyncio
async def test_a_push_sends_each_counters_queued_increments_summed_in_one_pipeline():
    handle: Final = FakeUsageHandle(redis={_A: 10})
    strategy: Final = _strategy(handle)
    for counter in (_A, _B, _A):
        await strategy._increment_value_in_current_window(counter, 1, 60)  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    totals: Final = await strategy._push_in_memory_increments_to_redis()  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    assert handle.calls == [("remote", (CounterIncrement(_A, 2, 60), CounterIncrement(_B, 1, 60)))]
    assert totals == {_A: 12, _B: 1}
    assert strategy.redis_increment_operation_queue == []


@pytest.mark.asyncio
async def test_a_sync_moves_memory_up_to_what_every_pod_counted_in_redis():
    handle: Final = FakeUsageHandle(redis={_A: 20})
    strategy: Final = _strategy(handle)
    await strategy._increment_value_in_current_window(_A, 3, 60)  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    await strategy._sync_in_memory_spend_with_redis()  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    assert handle.redis == {_A: 23}
    assert handle.local == {_A: 23}


@pytest.mark.asyncio
async def test_without_redis_a_sync_leaves_the_queue_and_memory_alone():
    handle: Final = FakeUsageHandle()
    strategy: Final = _strategy(handle)
    await strategy._increment_value_in_current_window(_A, 3, 60)  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    await strategy._sync_in_memory_spend_with_redis()  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    assert handle.local == {_A: 3}
    assert handle.calls == []
    assert strategy.redis_increment_operation_queue == [CounterIncrement(_A, 3, 60)]


@pytest.mark.asyncio
async def test_cache_keys_management():
    strategy: Final = _strategy(FakeUsageHandle())
    strategy.add_to_in_memory_keys_to_update(_A)
    strategy.add_to_in_memory_keys_to_update(_B)
    strategy.add_to_in_memory_keys_to_update(_A)

    assert strategy.get_in_memory_keys_to_update() == {_A, _B}
    strategy.reset_in_memory_keys_to_update()
    assert strategy.get_in_memory_keys_to_update() == set()


class _RefusedHandle(FakeUsageHandle):
    def increment_remote(self, increments: Sequence[CounterIncrement]) -> asyncio.Future[list[float] | None]:
        raise NativeRedisStoreError("breaker_open", "the Redis circuit breaker is open")


@pytest.mark.asyncio
async def test_push_refused_by_the_open_circuit_breaker_is_not_logged_as_an_error(caplog):
    """The sync loop pushes every 100 ms under usage-based routing, so an open breaker must not add an error line per cycle."""
    strategy: Final = _strategy(_RefusedHandle(redis={}))
    await strategy._increment_value_in_current_window(_A, 1, 60)  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    with caplog.at_level(logging.ERROR):
        await strategy._push_in_memory_increments_to_redis()  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer

    assert caplog.records == []
    assert strategy.redis_increment_operation_queue == []


@pytest.mark.asyncio
async def test_an_increment_counted_after_retire_still_reaches_redis():
    server: Final = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    redis_cache: Final = RedisCache(host="127.0.0.1", port=server.server_address[1])
    strategy: Final = BaseRoutingStrategy(
        usage=RouterUsageStore(connect(redis_cache).router_usage()),
        should_batch_redis_writes=True,
        default_sync_interval=60,
    )
    try:
        strategy.retire()
        await strategy._increment_value_in_current_window(_A, 3, 60)  # pyright: ignore[reportPrivateUsage]  # a step of the sync loop the strategy runs on a timer
        await asyncio.sleep(0.2)

        assert float(await redis_cache.async_get_cache("dep-a:openai/gpt-5.4-nano:rpm:12-34")) == 3
    finally:
        server.shutdown()
        server.server_close()
