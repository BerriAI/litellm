import asyncio
import time
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_router_logger
from litellm.exceptions import RateLimitError, RateLimitErrorCategory, RateLimitType
from litellm.types.router import RouterErrors
from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class DeploymentSemaphore(asyncio.Semaphore):
    """A deployment's max_parallel_requests slots. ``queue_size=None`` parks callers without bound, like a plain
    ``asyncio.Semaphore``; otherwise a caller arriving while all slots are busy and ``queue_size`` callers already
    wait gets a 429 instead of being parked."""

    def __init__(self, max_parallel_requests: int, model_id: str, model_group: str, queue_size: int | None) -> None:
        super().__init__(max_parallel_requests)
        self.max_parallel_requests = max_parallel_requests
        self.model_id = model_id
        self.model_group = model_group
        self.queue_size = queue_size
        self.waiting = 0

    async def acquire(self) -> bool:
        if not self.locked():
            return await super().acquire()
        if self.queue_size is not None and self.waiting >= self.queue_size:
            raise RateLimitError(
                message=(
                    f"{RouterErrors.max_parallel_requests_queue_full.value} Deployment model_group={self.model_group}, "
                    f"id={self.model_id} has all max_parallel_requests={self.max_parallel_requests} slots in use and "
                    f"{self.waiting} requests already waiting, which is its max_parallel_requests_queue_size="
                    f"{self.queue_size}. Raise max_parallel_requests or max_parallel_requests_queue_size for this "
                    "deployment, or unset max_parallel_requests_queue_size to queue without a bound"
                ),
                llm_provider="",
                model=self.model_group,
                category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
                rate_limit_type=RateLimitType.CONCURRENT_REQUESTS,
            )
        self.waiting += 1
        queued_at: Final = time.perf_counter()
        verbose_router_logger.debug(
            "Deployment model_group=%s, id=%s has all max_parallel_requests=%s slots in use, request queued "
            "(waiting=%s, max_parallel_requests_queue_size=%s)",
            self.model_group,
            self.model_id,
            self.max_parallel_requests,
            self.waiting,
            self.queue_size,
        )
        try:
            return await super().acquire()
        finally:
            self.waiting -= 1
            verbose_router_logger.debug(
                "Deployment model_group=%s, id=%s request left the max_parallel_requests queue after %.1f ms",
                self.model_group,
                self.model_id,
                (time.perf_counter() - queued_at) * 1000,
            )


class InitalizeCachedClient:
    @staticmethod
    def set_max_parallel_requests_client(litellm_router_instance: LitellmRouter, model: dict):
        litellm_params: Final = model.get("litellm_params", {})
        model_id: Final = model["model_info"]["id"]
        rpm: Final = litellm_params.get("rpm", None)
        tpm: Final = litellm_params.get("tpm", None)
        max_parallel_requests: Final = litellm_params.get("max_parallel_requests", None)
        calculated_max_parallel_requests: Final = calculate_max_parallel_requests(
            rpm=rpm,
            max_parallel_requests=max_parallel_requests,
            tpm=tpm,
            default_max_parallel_requests=litellm_router_instance.default_max_parallel_requests,
        )
        if calculated_max_parallel_requests:
            deployment_queue_size: Final = litellm_params.get("max_parallel_requests_queue_size", None)
            semaphore: Final = DeploymentSemaphore(
                max_parallel_requests=calculated_max_parallel_requests,
                model_id=model_id,
                model_group=model.get("model_name", ""),
                queue_size=(
                    deployment_queue_size
                    if deployment_queue_size is not None
                    else litellm_router_instance.default_max_parallel_requests_queue_size
                ),
            )
            cache_key: Final = f"{model_id}_max_parallel_requests_client"
            litellm_router_instance.cache.set_cache(
                key=cache_key,
                value=semaphore,
                local_only=True,
            )

    @staticmethod
    def apply_default_max_parallel_requests_queue_size(
        litellm_router_instance: LitellmRouter, queue_size: int | None
    ) -> None:
        inheriting_semaphores: Final = (
            litellm_router_instance.cache.get_cache(
                key=f"{model['model_info']['id']}_max_parallel_requests_client", local_only=True
            )
            for model in litellm_router_instance.model_list
            if model["litellm_params"].get("max_parallel_requests_queue_size") is None
        )
        for semaphore in inheriting_semaphores:
            if isinstance(semaphore, DeploymentSemaphore):
                semaphore.queue_size = queue_size
