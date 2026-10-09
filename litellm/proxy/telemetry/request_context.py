import asyncio
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Final

from litellm.telemetry.records import AttemptRecord, BlockCounts, TokenCounts


@dataclass(frozen=True, slots=True)
class AttemptObservation:
    attempt: AttemptRecord
    succeeded: bool
    tokens: TokenCounts = field(default_factory=TokenCounts)
    litellm_cache_hit: bool = False
    blocks: BlockCounts | None = None


class RequestAccumulator:
    """Collects the provider attempts logged while one proxy request is in flight"""

    def __init__(self) -> None:
        self._observations: tuple[AttemptObservation, ...] = ()
        self._succeeded: Final = asyncio.Event()

    @property
    def observations(self) -> tuple[AttemptObservation, ...]:
        return self._observations

    def add(self, observation: AttemptObservation) -> None:
        self._observations = (*self._observations, observation)
        if observation.succeeded:
            self._succeeded.set()

    async def wait_for_success(self, timeout_s: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._succeeded.wait(), timeout_s)


current_request: Final[ContextVar[RequestAccumulator | None]] = ContextVar("litellm_telemetry_request", default=None)
