import asyncio
import enum
import heapq
import logging as _logging
import time
from collections.abc import Awaitable, Callable, Sequence
from functools import partial as _partial
from typing import Final, TypeAlias
from typing import Protocol as _Protocol

from pydantic import TypeAdapter

from litellm import print_verbose
from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger as _verbose_router_logger
from litellm.caching.caching import DualCache, RedisCache
from litellm.caching.redis_cache import (
    RedisCircuitBreakerOpenError as _RedisCircuitBreakerOpenError,
)
from litellm.caching.redis_cache import (
    _is_redis_health_failure,  # pyright: ignore[reportPrivateUsage]  # RedisCache's connectivity classifier has no public counterpart
)
from litellm.caching.redis_cache import (
    log_redis_failure as _log_redis_failure,
)
from litellm.constants import DEFAULT_IN_MEMORY_TTL, DEFAULT_POLLING_INTERVAL
from litellm.exceptions import Timeout
from litellm.types.llms.base import LiteLLMBaseModel

SCHEDULER_QUEUE_TARGET: Final = "scheduler_queue"
QueueEntry: TypeAlias = tuple[int, str]
_QUEUE_ENTRIES: Final = TypeAdapter(list[QueueEntry])
_Queue: TypeAlias = tuple[QueueEntry, ...]
_QueueResult: TypeAlias = tuple[bool, _Queue]
_QUEUE_SNAPSHOT: Final = TypeAdapter(_Queue)
_RAW_QUEUE: Final = TypeAdapter[str | bytes | None](str | bytes | None)
_CAS_RESULT: Final = TypeAdapter(bool)
_QUEUE_READ: Final = "return redis.call('GET', KEYS[1])"
_QUEUE_COMPARE_SET: Final = """
local current = redis.call('GET', KEYS[1])
if (ARGV[1] == '0' and current ~= false) or
   (ARGV[1] == '1' and current ~= ARGV[2]) then
    return 0
end
if ARGV[4] == '' then
    redis.call('SET', KEYS[1], ARGV[3])
else
    redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[4])
end
return 1
"""


class SchedulerCacheKeys(enum.Enum):
    queue = "scheduler:queue"
    default_in_memory_ttl = DEFAULT_IN_MEMORY_TTL  # cache queue in-memory for 5s when redis cache available


class FlowItem(LiteLLMBaseModel):
    priority: int  # Priority between 0 and 255
    request_id: str
    model_name: str


class _LocalQueueCache(_Protocol):
    @property
    def default_in_memory_ttl(self) -> float | None: ...

    async def async_get_cache(self, key: str, *, local_only: bool) -> object: ...

    async def async_set_cache(self, key: str, value: Sequence[QueueEntry], *, local_only: bool) -> None: ...


async def _read_local_queue(cache: _LocalQueueCache, key: str) -> _Queue:
    local: Final = await cache.async_get_cache(key=key, local_only=True)
    return _QUEUE_SNAPSHOT.validate_python(local or ())


async def _write_local_queue(cache: _LocalQueueCache, key: str, queue: _Queue) -> None:
    await cache.async_set_cache(key=key, value=list(queue), local_only=True)


def _enqueue(entry: QueueEntry, queue: _Queue) -> _QueueResult:
    return False, queue if entry in queue else tuple(sorted((*queue, entry)))


def _remove(request_id: str, queue: _Queue) -> _QueueResult:
    return False, tuple(sorted(item for item in queue if item[1] != request_id))


def _poll(entry: QueueEntry, queue: _Queue) -> _QueueResult:
    pending: Final = tuple(sorted(queue if entry in queue else (*queue, entry)))
    return (True, pending[1:]) if pending[0] == entry else (False, pending)


def _redis_unavailable(error: Exception) -> bool:
    return isinstance(error, _RedisCircuitBreakerOpenError) or _is_redis_health_failure(error)


async def _read_redis_queue(cache: RedisCache, key: str) -> tuple[str | bytes | None, _Queue]:
    raw: Final = _RAW_QUEUE.validate_python(await cache.async_register_script(_QUEUE_READ)(keys=(key,), args=()))
    return raw, () if raw is None else _QUEUE_SNAPSHOT.validate_json(raw)


async def _mutate_redis_queue(
    cache: RedisCache, key: str, model_name: str, ttl: int | None, change: Callable[[_Queue], _QueueResult]
) -> _QueueResult:
    script: Final = cache.async_register_script(_QUEUE_COMPARE_SET)
    for _ in range(100):
        raw, queue = await _read_redis_queue(cache, key)
        admitted, updated = change(queue)
        if updated == queue:
            return admitted, updated
        if _CAS_RESULT.validate_python(
            await script(
                keys=(key,),
                args=(
                    "0" if raw is None else "1",
                    raw or "",
                    _QUEUE_SNAPSHOT.dump_json(updated),
                    "" if ttl is None else str(ttl),
                ),
            )
        ):
            return admitted, updated
    raise Timeout(
        message="Scheduler queue remained busy after concurrent updates", model=model_name, llm_provider="openai"
    )


async def _shared_queue_result(
    cache: _LocalQueueCache,
    redis_cache: RedisCache,
    key: str,
    model_name: str,
    change: Callable[[_Queue], _QueueResult],
) -> _QueueResult:
    try:
        return await _mutate_redis_queue(
            redis_cache,
            key,
            model_name,
            redis_cache.default_ttl if cache.default_in_memory_ttl is None else int(cache.default_in_memory_ttl),
            change,
        )
    except Exception as error:
        if not _redis_unavailable(error):
            raise
        _log_redis_failure(
            _verbose_router_logger, _logging.WARNING, "Scheduler using local queue while Redis is unavailable", error
        )
        return change(await _read_local_queue(cache, key))


class Scheduler:
    cache: DualCache

    def __init__(
        self,
        polling_interval: float | None = None,
        redis_cache: RedisCache | None = None,
    ):
        """
        polling_interval: float or null - frequency of polling queue. Default is 3ms.
        """
        self.queue: list[QueueEntry] = []
        default_in_memory_ttl: float | None = None
        if redis_cache is not None:
            # if redis-cache available frequently poll that instead of using in-memory.
            default_in_memory_ttl = SchedulerCacheKeys.default_in_memory_ttl.value
        self.cache = DualCache(redis_cache=redis_cache, default_in_memory_ttl=default_in_memory_ttl)
        self.polling_interval = polling_interval or DEFAULT_POLLING_INTERVAL  # default to 3ms

    @with_service_target(SCHEDULER_QUEUE_TARGET)
    async def _update_shared_queue(
        self, redis_cache: RedisCache, model_name: str, change: Callable[[_Queue], _QueueResult]
    ) -> bool:
        key: Final = f"{SchedulerCacheKeys.queue.value}:{model_name}"
        result: Final = await _shared_queue_result(self.cache, redis_cache, key, model_name, change)
        await _write_local_queue(self.cache, key, result[1])
        return result[0]

    async def add_request(self, request: FlowItem):
        redis_cache: Final = self.cache.redis_cache
        if isinstance(redis_cache, RedisCache):
            await self._update_shared_queue(
                redis_cache, request.model_name, _partial(_enqueue, (request.priority, request.request_id))
            )
            return
        # We use the priority directly, as lower values indicate higher priority
        # get the queue
        queue: Final = await self.get_queue(model_name=request.model_name)
        # update the queue
        heapq.heappush(queue, (request.priority, request.request_id))

        # save the queue
        await self.save_queue(queue=queue, model_name=request.model_name)

    async def poll(self, request: FlowItem, health_deployments: Sequence[object]) -> bool:
        """
        Return if request can be processed.

        Returns:
        - True:
            * If healthy deployments are available
            * OR If request at the top of queue
        - False:
            * If no healthy deployments available
            * AND request not at the top of queue

        A request the queue no longer holds (its cache key expired or a concurrent writer erased the entry)
        is put back at its priority so it keeps its place in the order instead of failing or jumping ahead
        """
        print_verbose(f"len(health_deployments): {len(health_deployments)}")
        if len(health_deployments) > 0:
            return True

        redis_cache: Final = self.cache.redis_cache
        if isinstance(redis_cache, RedisCache):
            return await self._update_shared_queue(
                redis_cache, request.model_name, _partial(_poll, (request.priority, request.request_id))
            )
        queue: Final = await self.get_queue(model_name=request.model_name)
        entry: Final = (request.priority, request.request_id)
        print_verbose(f"queue: {queue}, seeking {entry}")
        if entry not in queue:
            print_verbose(f"queue no longer holds {entry}, re-enqueueing it")
            heapq.heappush(queue, entry)
            if queue[0] != entry:
                await self.save_queue(queue=queue, model_name=request.model_name)
                return False
        elif queue[0] != entry:
            return False

        heapq.heappop(queue)
        await self.save_queue(queue=queue, model_name=request.model_name)
        print_verbose(f"Popped id: {request.request_id}")
        return True

    async def wait_for_turn(
        self,
        request: FlowItem,
        timeout: float,
        get_healthy_deployments: Callable[[], Awaitable[Sequence[object]]],
    ) -> None:
        try:
            await self.add_request(request=request)
            end_time: Final = time.monotonic() + timeout
            while time.monotonic() < end_time:
                if await self.poll(request=request, health_deployments=await get_healthy_deployments()):
                    return
                await asyncio.sleep(self.polling_interval)
        finally:
            await asyncio.shield(self.remove_request(request_id=request.request_id, model_name=request.model_name))
        raise Timeout(message="Request timed out while polling queue", model=request.model_name, llm_provider="openai")

    async def remove_request(self, request_id: str, model_name: str) -> None:
        """
        Remove a specific request from the priority queue for a model.
        Used when a request times out while waiting in the queue.
        """
        redis_cache: Final = self.cache.redis_cache
        if isinstance(redis_cache, RedisCache):
            await self._update_shared_queue(redis_cache, model_name, _partial(_remove, request_id))
            return
        queue: Final = await self.get_queue(model_name=model_name)
        filtered_queue: Final = [item for item in queue if item[1] != request_id]
        heapq.heapify(filtered_queue)  # restore heap invariant after filtering
        await self.save_queue(queue=filtered_queue, model_name=model_name)
        print_verbose(f"Removed request_id: {request_id} from queue for model: {model_name}")

    async def peek(self, id: str, model_name: str, health_deployments: list) -> bool:
        """Return if the id is at the top of the queue. Don't pop the value from heap."""
        queue: Final = await self.get_queue(model_name=model_name)
        if not queue:
            raise Exception(f"Incorrectly setup. Queue is invalid. Queue={queue}")

        # ------------
        # Setup values
        # ------------

        # Check if the id is at the top of the heap
        if queue[0][1] == id:
            return True

        return False

    def get_queue_status(self):
        """Get the status of items in the queue"""
        return self.queue

    @with_service_target(SCHEDULER_QUEUE_TARGET)
    async def get_queue(self, model_name: str) -> list[QueueEntry]:
        """
        Return a queue for that specific model group.

        Redis hands the queue back as JSON lists, so every entry is validated into the
        (priority, request_id) tuple the heap operations compare against.
        """
        redis_cache: Final = self.cache.redis_cache
        if isinstance(redis_cache, RedisCache):
            key: Final = f"{SchedulerCacheKeys.queue.value}:{model_name}"
            try:
                _, snapshot = await _read_redis_queue(redis_cache, key)
            except Exception as error:
                if not _redis_unavailable(error):
                    raise
                return list(await _read_local_queue(self.cache, key))
            await _write_local_queue(self.cache, key, snapshot)
            return list(snapshot)
        if self.cache is not None:
            _cache_key: Final = f"{SchedulerCacheKeys.queue.value}:{model_name}"
            response: Final = await self.cache.async_get_cache(key=_cache_key)
            if not isinstance(response, list):
                return []
            return _QUEUE_ENTRIES.validate_python(response)
        return self.queue

    @with_service_target(SCHEDULER_QUEUE_TARGET)
    async def save_queue(self, queue: list[QueueEntry], model_name: str) -> None:
        """
        Save the updated queue of the model group
        """
        if self.cache is not None:
            _cache_key: Final = f"{SchedulerCacheKeys.queue.value}:{model_name}"
            await self.cache.async_set_cache(key=_cache_key, value=queue)
