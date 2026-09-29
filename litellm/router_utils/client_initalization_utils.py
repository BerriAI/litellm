from types import TracebackType
from typing import TYPE_CHECKING, Any, Final

from litellm.exceptions import RateLimitError, RateLimitErrorCategory, RateLimitType
from litellm.types.router import RouterErrors
from litellm.utils import calculate_max_parallel_requests

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


class MaxParallelRequestsLimit:
    """A deployment's max_parallel_requests slots. A caller arriving while every slot is in use gets a 429 instead
    of waiting for one to free up."""

    def __init__(self, max_parallel_requests: int, model_id: str, model_group: str) -> None:
        self.max_parallel_requests: Final = max_parallel_requests
        self.model_id: Final = model_id
        self.model_group: Final = model_group
        self.in_flight = 0

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()

    def acquire(self) -> None:
        if self.in_flight >= self.max_parallel_requests:
            raise RateLimitError(
                message=(
                    f"{RouterErrors.max_parallel_requests_exceeded.value} Deployment model_group={self.model_group}, "
                    f"id={self.model_id} already has max_parallel_requests={self.max_parallel_requests} requests in "
                    "flight. Raise max_parallel_requests (or the rpm/tpm it is derived from) for this deployment"
                ),
                llm_provider="",
                model=self.model_group,
                category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
                rate_limit_type=RateLimitType.CONCURRENT_REQUESTS,
            )
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1


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
            limit: Final = MaxParallelRequestsLimit(
                max_parallel_requests=calculated_max_parallel_requests,
                model_id=model_id,
                model_group=model.get("model_name", ""),
            )
            cache_key: Final = f"{model_id}_max_parallel_requests_client"
            litellm_router_instance.cache.set_cache(
                key=cache_key,
                value=limit,
                local_only=True,
            )
