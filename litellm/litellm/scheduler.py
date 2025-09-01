import enum
import heapq
from typing import Optional

from pydantic import BaseModel

from litellm import print_verbose
from litellm.caching.caching import DualCache, RedisCache
from litellm.constants import DEFAULT_IN_MEMORY_TTL, DEFAULT_POLLING_INTERVAL


class SchedulerCacheKeys(enum.Enum):
    queue = "scheduler:queue"
    default_in_memory_ttl = (
        DEFAULT_IN_MEMORY_TTL  # cache queue in-memory for 5s when redis cache available
    )


class FlowItem(BaseModel):
    priority: int  # Priority between 0 and 255
    request_id: str
    model_name: str


class Scheduler:
    cache: DualCache

    def __init__(
        self,
        polling_interval: Optional[float] = None,
        redis_cache: Optional[RedisCache] = None,
    ):
        """
        polling_interval: float or null - frequency of polling queue. Default is 3ms.
        """
        self.queue: list = []
        default_in_memory_ttl: Optional[float] = None
        if redis_cache is not None:
            # if redis-cache available frequently poll that instead of using in-memory.
            default_in_memory_ttl = SchedulerCacheKeys.default_in_memory_ttl.value
        self.cache = DualCache(
            redis_cache=redis_cache, default_in_memory_ttl=default_in_memory_ttl
        )
        self.polling_interval = (
            polling_interval or DEFAULT_POLLING_INTERVAL
        )  # default to 3ms

    async def add_request(self, request: FlowItem):
        # We use the priority directly, as lower values indicate higher priority
        # get the queue
        queue = await self.get_queue(model_name=request.model_name)
        # update the queue
        heapq.heappush(queue, (request.priority, request.request_id))

        # save the queue
        await self.save_queue(queue=queue, model_name=request.model_name)

    async def poll(self, id: str, model_name: str, health_deployments: list) -> bool:
        """
        Return if request can be processed.

        Returns:
        - True:
            * If healthy deployments are available
            * OR If request at the top of queue
        - False:
            * If no healthy deployments available
            * AND request not at the top of queue
        """
        queue = await self.get_queue(model_name=model_name)
        if not queue:
            raise Exception(
                "Incorrectly setup. Queue is invalid. Queue={}".format(queue)
            )

        # ------------
        # Setup values
        # ------------

        print_verbose(f"len(health_deployments): {len(health_deployments)}")
        if len(health_deployments) == 0:
            print_verbose(f"queue: {queue}, seeking id={id}")
            # Check if the id is at the top of the heap
            if queue[0][1] == id:
                # Remove the item from the queue
                heapq.heappop(queue)
                print_verbose(f"Popped id: {id}")
                return True
            else:
                return False

        return True

    async def peek(self, id: str, model_name: str, health_deployments: list) -> bool:
        """Return if the id is at the top of the queue. Don't pop the value from heap."""
        queue = await self.get_queue(model_name=model_name)
        if not queue:
            raise Exception(
                "Incorrectly setup. Queue is invalid. Queue={}".format(queue)
            )

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

    async def get_queue(self, model_name: str) -> list:
        """
        Return a queue for that specific model group
        """
        if self.cache is not None:
            _cache_key = "{}:{}".format(SchedulerCacheKeys.queue.value, model_name)
            response = await self.cache.async_get_cache(key=_cache_key)
            if response is None or not isinstance(response, list):
                return []
            elif isinstance(response, list):
                return response
        return self.queue

    async def save_queue(self, queue: list, model_name: str) -> None:
        """
        Save the updated queue of the model group
        """
        if self.cache is not None:
            _cache_key = "{}:{}".format(SchedulerCacheKeys.queue.value, model_name)
            await self.cache.async_set_cache(key=_cache_key, value=queue)
        return None
