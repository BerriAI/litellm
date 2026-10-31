import asyncio
import enum
import heapq
import logging as _logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias
from typing import Protocol as _Protocol

from pydantic import TypeAdapter

from litellm import print_verbose
from litellm._internal_context import with_service_target
from litellm._logging import verbose_logger as _verbose_logger
from litellm.caching.caching import DualCache, RedisCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError as _RedisCircuitBreakerOpenError
from litellm.caching.redis_cache import log_redis_failure as _log_redis_failure
from litellm.constants import DEFAULT_IN_MEMORY_TTL, DEFAULT_POLLING_INTERVAL
from litellm.exceptions import Timeout
from litellm.types.llms.base import LiteLLMBaseModel

SCHEDULER_QUEUE_TARGET: Final = "scheduler_queue"
QueueEntry: TypeAlias = tuple[int, str]
_QUEUE_ENTRIES: Final = TypeAdapter(list[QueueEntry])


@dataclass(frozen=True, slots=True)
class _QueueRead:
    value: object
    from_redis: bool


@dataclass(frozen=True, slots=True)
class _QueueSnapshot:
    read_at: float
    entries: tuple[QueueEntry, ...]


class SchedulerCacheKeys(enum.Enum):
    queue = "scheduler:queue"
    default_in_memory_ttl = DEFAULT_IN_MEMORY_TTL  # cache queue in-memory for 5s when redis cache available


class FlowItem(LiteLLMBaseModel):
    priority: int  # Priority between 0 and 255
    request_id: str
    model_name: str


class _QueueCache(_Protocol):
    async def async_get_cache(self, key: str, *, local_only: bool = False, raise_on_error: bool = False) -> object: ...

    async def async_set_cache(self, key: str, value: object, *, local_only: bool = False) -> object: ...


def _validated_queue(response: object) -> tuple[QueueEntry, ...]:
    if not isinstance(response, list):
        return ()
    return tuple(_QUEUE_ENTRIES.validate_python(response))


async def _read_queue(cache: _QueueCache, redis_cache: _QueueCache | None, key: str) -> _QueueRead:
    if redis_cache is None:
        return _QueueRead(value=await cache.async_get_cache(key=key, local_only=True), from_redis=False)
    from redis.exceptions import RedisError as _RedisError

    try:
        response: Final = await redis_cache.async_get_cache(key=key, raise_on_error=True)
    except (_RedisError, _RedisCircuitBreakerOpenError) as error:
        _log_redis_failure(_verbose_logger, _logging.ERROR, "Scheduler queue using local snapshot", error)
        return _QueueRead(value=await cache.async_get_cache(key=key, local_only=True), from_redis=False)
    await cache.async_set_cache(key=key, value=[] if response is None else response, local_only=True)
    return _QueueRead(value=response, from_redis=True)


class Scheduler:
    cache: DualCache
    clock: Callable[[], float]
    _queue_snapshots: Mapping[str, _QueueSnapshot]
    _queue_read_tasks: Mapping[str, asyncio.Task[_QueueRead]]

    def __init__(
        self,
        polling_interval: float | None = None,
        redis_cache: RedisCache | None = None,
        clock: Callable[[], float] = time.monotonic,
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
        self.clock = clock
        self._queue_snapshots = MappingProxyType({})
        self._queue_read_tasks = MappingProxyType({})

    async def _read_queue_with_snapshot(self, key: str, read_at: float) -> _QueueRead:
        result: Final = await _read_queue(self.cache, self.cache.redis_cache, key)
        if not result.from_redis:
            return result
        entries: Final = _validated_queue(result.value)
        snapshot: Final = self._queue_snapshots.get(key)
        if snapshot is None or read_at > snapshot.read_at:
            self._queue_snapshots = MappingProxyType(
                {**self._queue_snapshots, key: _QueueSnapshot(read_at=read_at, entries=entries)}
            )
        return _QueueRead(value=list(entries), from_redis=True)

    def _clear_in_flight_read(self, task: asyncio.Task[_QueueRead]) -> None:
        self._queue_read_tasks = MappingProxyType(
            {key: current for key, current in self._queue_read_tasks.items() if current is not task}
        )

    async def _await_queue_read(self, task: asyncio.Task[_QueueRead]) -> _QueueRead:
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._clear_in_flight_read(task)

    async def _get_redis_queue(self, key: str) -> tuple[QueueEntry, ...]:
        now: Final = self.clock()
        snapshot: Final = self._queue_snapshots.get(key)
        if snapshot is not None and now - snapshot.read_at < self.polling_interval:
            return snapshot.entries
        task: Final = self._queue_read_tasks.get(key)
        if task is None:
            read_at: Final = now
            new_task: Final = asyncio.create_task(self._read_queue_with_snapshot(key, read_at))
            self._queue_read_tasks = MappingProxyType({**self._queue_read_tasks, key: new_task})
            new_task.add_done_callback(self._clear_in_flight_read)
            new_result: Final = await self._await_queue_read(new_task)
            return _validated_queue(new_result.value)
        shared_result: Final = await self._await_queue_read(task)
        return _validated_queue(shared_result.value)

    async def add_request(self, request: FlowItem):
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
        if self.cache is not None:
            _cache_key: Final = f"{SchedulerCacheKeys.queue.value}:{model_name}"
            if self.cache.redis_cache is not None:
                return list(await self._get_redis_queue(_cache_key))
            response: Final = await _read_queue(self.cache, self.cache.redis_cache, _cache_key)
            return list(_validated_queue(response.value))
        return self.queue

    @with_service_target(SCHEDULER_QUEUE_TARGET)
    async def save_queue(self, queue: list[QueueEntry], model_name: str) -> None:
        """
        Save the updated queue of the model group
        """
        if self.cache is not None:
            _cache_key: Final = f"{SchedulerCacheKeys.queue.value}:{model_name}"
            await self.cache.async_set_cache(key=_cache_key, value=queue)
            if self.cache.redis_cache is not None:
                self._queue_snapshots = MappingProxyType(
                    {
                        **self._queue_snapshots,
                        _cache_key: _QueueSnapshot(read_at=self.clock(), entries=tuple(queue)),
                    }
                )
