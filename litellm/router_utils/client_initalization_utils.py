from types import TracebackType
from typing import Final

from litellm.exceptions import RateLimitError, RateLimitErrorCategory, RateLimitType
from litellm.types.router import RouterErrors


class MaxParallelRequestsLimit:
    """A deployment's max_parallel_requests slots. A caller arriving while every slot is in use gets a 429 instead
    of waiting for one to free up."""

    def __init__(self, max_parallel_requests: int, model_id: str, model_group: str) -> None:
        self.max_parallel_requests: int = max_parallel_requests
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
