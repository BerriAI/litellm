import heapq, time
from pydantic import BaseModel
from typing import Optional
import enum
from litellm.caching import DualCache
from litellm import Router
from litellm import print_verbose


class SchedulerCacheKeys(enum.Enum):
    queue = "scheduler:queue"


class DefaultPriorities(enum.Enum):
    High = 0
    Medium = 128
    Low = 255


class FlowItem(BaseModel):
    priority: int  # Priority between 0 and 255
    request_id: str
    model_group: str


class Scheduler:
    cache: DualCache
    llm_router: Optional[Router] = None

    def __init__(self):
        self.queue = []
        self.cache = DualCache()

    def update_variables(self, llm_router: Router, cache: Optional[DualCache] = None):
        self.llm_router = llm_router
        if cache is not None:
            self.cache = cache

    async def add_request(self, request: FlowItem):
        # We use the priority directly, as lower values indicate higher priority
        # get the queue
        queue = await self.get_queue(model_group=request.model_group)
        # update the queue
        heapq.heappush(queue, (request.priority, request.request_id))

        # save the queue
        await self.save_queue(queue=queue, model_group=request.model_group)

    async def poll(self, id: str, model_group: str) -> bool:
        """Return if the id is at the top of the queue and if the token bucket allows processing"""
        queue = await self.get_queue(model_group=model_group)
        if not queue or not self.llm_router:
            raise Exception(
                "Incorrectly setup. Queue or Router is invalid. Queue={}, Router={}".format(
                    queue, self.llm_router
                )
            )

        # ------------
        # Setup values
        # ------------
        _healthy_deployments = await self.llm_router._async_get_healthy_deployments(
            model=model_group
        )

        print_verbose(f"len(_healthy_deployments): {len(_healthy_deployments)}")
        if len(_healthy_deployments) == 0:
            return False

        print_verbose(f"queue: {queue}, seeking id={id}")
        # Check if the id is at the top of the heap
        if queue[0][1] == id:
            # Remove the item from the queue
            heapq.heappop(queue)
            print_verbose(f"Popped id: {id}")
            return True

        return False

    async def peek(self, id: str, model_group: str) -> bool:
        """Return if the id is at the top of the queue. Don't pop the value from heap."""
        queue = await self.get_queue(model_group=model_group)
        if not queue or not self.llm_router:
            raise Exception(
                "Incorrectly setup. Queue or Router is invalid. Queue={}, Router={}".format(
                    queue, self.llm_router
                )
            )

        # ------------
        # Setup values
        # ------------
        _healthy_deployments = await self.llm_router._async_get_healthy_deployments(
            model=model_group
        )
        if len(_healthy_deployments) == 0:
            return False

        # Check if the id is at the top of the heap
        if queue[0][1] == id:
            return True

        return False

    def get_queue_status(self):
        """Get the status of items in the queue"""
        return self.queue

    async def get_queue(self, model_group: str) -> list:
        """
        Return a queue for that specific model group
        """
        if self.cache is not None:
            _cache_key = "{}:{}".format(SchedulerCacheKeys.queue.value, model_group)
            response = await self.cache.async_get_cache(key=_cache_key)
            if response is None or not isinstance(response, list):
                return []
            elif isinstance(response, list):
                return response
        return self.queue

    async def save_queue(self, queue: list, model_group: str) -> None:
        """
        Save the updated queue of the model group
        """
        if self.cache is not None:
            _cache_key = "{}:{}".format(SchedulerCacheKeys.queue.value, model_group)
            await self.cache.async_set_cache(key=_cache_key, value=queue)
        return None
