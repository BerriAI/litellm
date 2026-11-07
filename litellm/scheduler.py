import asyncio
import enum
import heapq
import logging as _logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
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


class SchedulerCacheKeys(enum.Enum):
    queue = "scheduler:queue"
    default_in_memory_ttl = DEFAULT_IN_MEMORY_TTL  # cache queue in-memory for 5s when redis cache available


_scheduler_redis_cache: RedisCache | None = None


def _get_scheduler_redis_cache() -> RedisCache | None:
    """Redis cache shared by router-side scheduler consumers. None until the router configures one."""
    global _scheduler_redis_cache
    return _scheduler_redis_cache


def set_scheduler_redis_cache(redis_cache: RedisCache | None) -> None:
    global _scheduler_redis_cache
    _scheduler_redis_cache = redis_cache


class FlowItem(LiteLLMBaseModel):
    priority: int  # Priority between 0 and 255
    request_id: str
    model_name: str
    deployment_id: str | None = None


class _DeploymentSlotLimits(_Protocol):
    def slots(self, deployment_id: str) -> tuple[int, int] | None:
        ...


class DeferredParallelSlotCheck:
    """Router-side check that fires after a queued request is admitted.

    When a proxy request opted into the queue via ``slot_wait_timeout``, the
    proxy's ``ParallelRequestLimiter`` deferred its 503/429 gate: instead of
    failing fast at entry it recorded a pending parallel slot reference on the
    request stash. This check runs on the router side, right after
    ``Scheduler.wait_for_turn`` admits the request and before the call is
    dispatched. If the deployment's slot is still taken at that moment it
    raises the same 503 the proxy gate would have raised, so a request that
    waited in the queue is not silently dispatched into an over-committed
    deployment.
    """

    def __init__(
        self,
        model_group: str,
        limits: Mapping[str, _DeploymentSlotLimits],
        deployment_id: str | None = None,
        deferred: bool = True,
    ) -> None:
        self.model_group: Final = model_group
        self.limits: Final = limits
        self.deployment_id: Final = deployment_id
        self.deferred: Final = deferred

    def __call__(self) -> None:
        if not self.deferred:
            return
        dep_id: Final = self.deployment_id if self.deployment_id is not None else next(iter(self.limits))
        limit: Final = self.limits.get(dep_id)
        slots: Final = limit.slots(dep_id) if limit is not None else None
        if slots is None:
            return
        in_flight, max_in_flight = slots
        if in_flight >= max_in_flight:
            from litellm.exceptions import ServiceUnavailableError

            raise ServiceUnavailableError(
                message=f"Temporary concurrency limit exceeded on model_group={self.model_group}",
                llm_provider="litellm_proxy",
                model=self.model_group,
                headers={"retry-after": "1"},
            )


class _QueueCache(_Protocol):
    async def async_get_cache(self, key: str, *, local_only: bool = False, raise_on_error: bool = False) -> object: ...

    async def async_set_cache(self, key: str, value: object, *, local_only: bool = False) -> object: ...


async def _read_queue(cache: _QueueCache, redis_cache: _QueueCache | None, key: str) -> object:
    if redis_cache is None:
        return await cache.async_get_cache(key=key, local_only=True)
    from redis.exceptions import RedisError as _RedisError

    try:
        response: Final = await redis_cache.async_get_cache(key=key, raise_on_error=True)
    except (_RedisError, _RedisCircuitBreakerOpenError) as error:
        _log_redis_failure(_verbose_logger, _logging.ERROR, "Scheduler queue using local snapshot", error)
        return await cache.async_get_cache(key=key, local_only=True)
    await cache.async_set_cache(key=key, value=[] if response is None else response, local_only=True)
    return response


class Scheduler:
    cache: DualCache

    def __init__(
        self,
        polling_interval: float | None = None,
        redis_cache: RedisCache | None = None,
        slot_limits: _DeploymentSlotLimits | None = None,
    ):
        """
        polling_interval: float or null - frequency of polling queue. Default is 3ms.
        slot_limits: optional router-side view of per-deployment in-flight counts,
        consulted only on the per-deployment queueing path.
        """
        set_scheduler_redis_cache(redis_cache)
        self.queue: Final[list[QueueEntry]] = []
        self.queues: Final[dict[str, list[QueueEntry]]] = {}
        self.slot_limits: Final = slot_limits
        default_in_memory_ttl: float | None = None
        if redis_cache is not None:
            # if redis-cache available frequently poll that instead of using in-memory.
            default_in_memory_ttl = SchedulerCacheKeys.default_in_memory_ttl.value
        self.cache = DualCache(redis_cache=redis_cache, default_in_memory_ttl=default_in_memory_ttl)
        self.polling_interval = polling_interval or DEFAULT_POLLING_INTERVAL  # default to 3ms

    async def add_request(self, request: FlowItem):
        # We use the priority directly, as lower values indicate higher priority
        cache_key: Final = self._cache_key(request.model_name, request.deployment_id)
        queue: Final = await self.get_queue(cache_key=cache_key)
        # update the queue
        heapq.heappush(queue, (request.priority, request.request_id))

        # save the queue
        await self.save_queue(queue=queue, cache_key=cache_key)

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
        if request.deployment_id is not None:
            limit: Final = self._slot_limits()
            slots: Final = limit.slots(request.deployment_id) if limit else None
            if slots is None:
                return True if health_deployments else await self._poll_legacy(request=request)
            if slots[0] < slots[1]:
                print_verbose(f"deployment {request.deployment_id} has a free slot")
                return await self._poll_legacy(request=request)
            print_verbose(f"deployment {request.deployment_id} has no free slot")
            return False

        if len(health_deployments) > 0:
            return True
        return await self._poll_legacy(request=request)

    def _cache_key(self, model_name: str, deployment_id: str | None) -> str:
        return f"{SchedulerCacheKeys.queue.value}:{model_name}:{deployment_id or '_'}"

    def _slot_limits(self) -> _DeploymentSlotLimits | None:
        return self.slot_limits

    async def _poll_legacy(self, request: FlowItem) -> bool:
        cache_key: Final = self._cache_key(request.model_name, request.deployment_id)
        queue: Final = await self.get_queue(cache_key=cache_key)
        entry: Final = (request.priority, request.request_id)
        print_verbose(f"queue: {queue}, seeking {entry}")
        if entry not in queue:
            print_verbose(f"queue no longer holds {entry}, re-enqueueing it")
            heapq.heappush(queue, entry)
            if queue[0] != entry:
                await self.save_queue(queue=queue, cache_key=cache_key)
                return False
        elif queue[0] != entry:
            return False

        heapq.heappop(queue)
        await self.save_queue(queue=queue, cache_key=cache_key)
        print_verbose(f"Popped id: {request.request_id}")
        return True

    async def wait_for_turn(
        self,
        request: FlowItem,
        timeout: float,
        get_healthy_deployments: Callable[[], Awaitable[Sequence[object]]],
    ) -> None:
        admitted: Final = False
        try:
            await self.add_request(request=request)
            end_time: Final = time.monotonic() + timeout
            while time.monotonic() < end_time:
                if await self.poll(request=request, health_deployments=await get_healthy_deployments()):
                    admitted = True
                    return
                await asyncio.sleep(self.polling_interval)
        finally:
            if not admitted:
                await asyncio.shield(
                    self.remove_request(
                        request_id=request.request_id,
                        model_name=request.model_name,
                        deployment_id=request.deployment_id,
                    )
                )
        if not admitted:
            raise Timeout(message="Request timed out while polling queue", model=request.model_name, llm_provider="openai")

    async def remove_request(self, request_id: str, model_name: str, deployment_id: str | None = None) -> None:
        """
        Remove a specific request from the priority queue for a model.
        Used when a request times out while waiting in the queue.
        """
        cache_key: Final = self._cache_key(model_name, deployment_id)
        queue: Final = await self.get_queue(cache_key=cache_key)
        filtered_queue: Final = [item for item in queue if item[1] != request_id]
        heapq.heapify(filtered_queue)  # restore heap invariant after filtering
        await self.save_queue(queue=filtered_queue, cache_key=cache_key)
        print_verbose(f"Removed request_id: {request_id} from queue for model: {model_name}")

    async def peek(self, id: str, model_name: str, health_deployments: list, deployment_id: str | None = None) -> bool:
        """Return if the id is at the top of the queue. Don't pop the value from heap."""
        cache_key: Final = self._cache_key(model_name, deployment_id)
        queue: Final = await self.get_queue(cache_key=cache_key)
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
        return self.queue if not self.queues else {key: queue for key, queue in self.queues.items() if queue}

    @with_service_target(SCHEDULER_QUEUE_TARGET)
    async def get_queue(self, cache_key: str) -> list[QueueEntry]:
        """
        Return a queue for that specific model group.

        Redis hands the queue back as JSON lists, so every entry is validated into the
        (priority, request_id) tuple the heap operations compare against.
        """
        if self.cache is not None:
            response: Final = await _read_queue(self.cache, self.cache.redis_cache, cache_key)
            if not isinstance(response, list):
                return []
            return _QUEUE_ENTRIES.validate_python(response)
        return self._in_memory_queue(cache_key)

    @with_service_target(SCHEDULER_QUEUE_TARGET)
    async def save_queue(self, queue: list[QueueEntry], cache_key: str) -> None:
        """
        Save the updated queue of the model group
        """
        if self.cache is not None:
            await self.cache.async_set_cache(key=cache_key, value=queue)
        else:
            target: Final = self._in_memory_queue(cache_key)
            target.clear()
            target.extend(queue)

    def _in_memory_queue(self, cache_key: str) -> list[QueueEntry]:
        model_name: Final = cache_key.split(f"{SchedulerCacheKeys.queue.value}:", 1)[1].rsplit(":", 1)[0]
        if cache_key == self._cache_key(model_name, None):
            return self.queue
        deployment_id: Final = cache_key.rsplit(":", 1)[1]
        return self.queues.setdefault(deployment_id, [])
