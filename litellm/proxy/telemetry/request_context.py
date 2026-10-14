import asyncio
import contextlib
from collections.abc import Callable
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


class RequestAccumulator:
    """Collects the provider attempts logged for one proxy request's own ``litellm_call_id``"""

    def __init__(self) -> None:
        self._observations: tuple[AttemptObservation, ...] = ()
        self._succeeded: Final = asyncio.Event()
        self._call_id: str | None = None
        self._blocks: BlockCounts | None = None
        self._blocks_counted: bool = False

    def bind(self, call_id: str) -> None:
        if self._call_id is None:
            self._call_id = call_id

    def owns(self, call_id: object) -> bool:
        return self._call_id is not None and call_id == self._call_id

    @property
    def blocks(self) -> BlockCounts | None:
        return self._blocks

    def count_blocks_once(self, count: Callable[[], BlockCounts | None]) -> None:
        if not self._blocks_counted:
            self._blocks_counted = True
            self._blocks = count()

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


def bind_call_id(call_id: str) -> None:
    request: Final = current_request.get()
    if request is not None:
        request.bind(call_id)
