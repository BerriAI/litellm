"""The router's side of ``dynamic_queue``: the ``router_settings.dynamic_queue`` block, the queue time each request
owns across its attempts, per-attempt admission, and the public errors a refused wait raises.

Enabled, every async non-streaming chat completion attempt on a deployment the router owns and picked as it always
does waits for that deployment's turn, and its outcome is reported back. Only a failure whose provider answered 429 or
529 congests a deployment. A full queue or a spent queue time ends the request with no retry, fallback or cooldown.
When the queue's Redis cannot be reached, the request goes on as it would without the queue.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator
from typing_extensions import Self, assert_never

from litellm.exceptions import ServiceUnavailableError, Timeout
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.total_deadline import TotalDeadline, current_total_deadline
from litellm.router_utils.dynamic_queue import (
    Admitted,
    Attempt,
    CoordinationUnavailable,
    DynamicQueue,
    MatchingFailure,
    MonotonicTimer,
    OtherFailure,
    Outcome,
    QueueBackend,
    QueueFull,
    QueueSettings,
    QueueTimeout,
    Timer,
    WaitRequest,
)
from litellm.router_utils.dynamic_queue_redis import RedisQueueBackend, ScriptHost

QUEUE_FULL_CODE: Final = "queue_full"
REDIS_NOT_CONFIGURED: Final = "redis_not_configured"
MAX_WAITING_LIMIT: Final = 64
TRIGGER_STATUS_CODES: Final = frozenset({429, 529})


class DynamicQueueSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    queue_timeout: float = Field(default=30.0, gt=0)
    max_waiting: StrictInt | None = Field(default=None, gt=0, le=MAX_WAITING_LIMIT)

    @model_validator(mode="after")
    def _an_enabled_queue_names_its_size(self) -> Self:
        """Every admission scans its deployment's whole queue in one Redis call, so its size is the operator's
        explicit choice, never a default."""
        if self.enabled and self.max_waiting is None:
            raise ValueError(
                "router_settings.dynamic_queue.enabled requires max_waiting, "
                f"a whole number from 1 to {MAX_WAITING_LIMIT}"
            )
        return self

    def queue_settings(self) -> QueueSettings:
        return QueueSettings(max_waiting=self.max_waiting or 0)


_SETTINGS: Final[TypeAdapter[DynamicQueueSettings | None]] = TypeAdapter(DynamicQueueSettings | None)


def parse_dynamic_queue(value: object, default_priority: int | None) -> DynamicQueueSettings | None:
    settings: Final = _SETTINGS.validate_python(value)
    if settings is None or not settings.enabled:
        return None
    if default_priority is None:
        raise ValueError("router_settings.dynamic_queue.enabled requires router_settings.default_priority")
    return settings


class QueueRejectedError(Exception):
    """Marks the public errors below so the router ends the request without a retry, fallback or cooldown."""


class DynamicQueueTimeoutError(Timeout, QueueRejectedError):
    pass


class DynamicQueueFullError(ServiceUnavailableError, QueueRejectedError):
    def __init__(self, model: str) -> None:
        super().__init__(message="the deployment's dynamic queue is full", llm_provider="", model=model)
        self.code = QUEUE_FULL_CODE


def rejection_error(rejection: QueueTimeout | QueueFull, model: str) -> QueueRejectedError:
    match rejection:
        case QueueTimeout():
            return DynamicQueueTimeoutError(
                message="the request ran out of dynamic queue time", model=model, llm_provider=""
            )
        case QueueFull():
            return DynamicQueueFullError(model)
        case _:
            assert_never(rejection)


_T: Final = TypeVar("_T")


async def _within(timer: Timer, seconds: float, call: Awaitable[_T]) -> _T | None:
    work: Final = asyncio.ensure_future(call)
    guard: Final = asyncio.create_task(timer.sleep(seconds))
    try:
        _ = await asyncio.wait((work, guard), return_when=asyncio.FIRST_COMPLETED)
        return work.result() if work.done() else None
    finally:
        for task in (work, guard):
            _ = task.cancel()
        _ = await asyncio.gather(work, guard, return_exceptions=True)


@dataclass(frozen=True, slots=True)
class Admission:
    """An admitted attempt, held until its provider call answers."""

    attempt: Attempt
    held: contextlib.AsyncExitStack
    owner: "RequestQueue"

    @property
    def router(self) -> "DynamicQueueRouter":
        return self.owner.router

    async def finish(self, outcome: Outcome | None) -> bool:
        """Whether the queue learned the outcome. When it could not, the rest of the request goes on without it."""
        reported: Final = None if outcome is None else await self.attempt.report(outcome)
        await self.held.aclose()
        if isinstance(reported, CoordinationUnavailable):
            self.owner.lose_coordination()
        return reported is None


@dataclass(frozen=True, slots=True)
class WithoutQueue:
    """The attempt goes on without the queue because its coordination failed. ``still_cooling`` when the deployment's
    own cooldown is confirmed active, so the attempt must not go out on it."""

    still_cooling: bool


class RequestQueue:
    """The queue state one request owns across its attempts. Only time spent waiting is
    taken from its budget. Once the queue's coordination fails for the request, its later attempts skip the queue,
    so it never half-waits on a Redis that may come back; the next request tries the queue again."""

    def __init__(self, router: "DynamicQueueRouter", priority: int, request_id: str) -> None:
        self.router = router
        self._priority = priority
        self._request_id = request_id
        self._deadline: Final[TotalDeadline | None] = current_total_deadline()
        self._remaining_s = router.queue_timeout_s
        self._waits = 0
        self.coordinated = True

    def lose_coordination(self) -> None:
        self.coordinated = False

    def _budget_s(self) -> float:
        deadline: Final = self._deadline or current_total_deadline()
        left: Final = None if deadline is None else deadline.remaining()
        return self._remaining_s if left is None else min(self._remaining_s, left)

    async def admit(
        self, capacity: str, model: str, cooldown_left_s: Callable[[], Awaitable[float]]
    ) -> Admission | WithoutQueue:
        """The attempt's turn, or WithoutQueue when coordination failed. Raises the public error of a full queue or
        spent queue time. ``cooldown_left_s`` keeps the wait queued, in priority order, until the deployment's own
        cooldown ends."""
        cooldown: Final = _ObservedCooldown(self.router.timer, cooldown_left_s)
        if not self.coordinated:
            return await self._without_queue(cooldown)
        queue: Final = self.router.queue()
        if isinstance(queue, CoordinationUnavailable):
            self.lose_coordination()
            return await self._without_queue(cooldown)
        self._waits += 1
        request: Final = WaitRequest(capacity, f"{self._request_id}:{self._waits}", self._priority)
        started: Final = self.router.timer.now()
        admitted: Final = await queue.admit(request, self._budget_s(), cooldown.left_s)
        self._remaining_s = max(self._remaining_s - (self.router.timer.now() - started), 0.0)
        if isinstance(admitted, CoordinationUnavailable):
            self.lose_coordination()
            return await self._without_queue(cooldown)
        if not isinstance(admitted, Admitted):
            raise rejection_error(admitted, model)
        held: Final = contextlib.AsyncExitStack()
        attempt: Final = await held.enter_async_context(queue.attempt(request, admitted))
        return Admission(attempt, held, self)

    async def _without_queue(self, cooldown: "_ObservedCooldown") -> WithoutQueue:
        """A lookup that fails or does not answer within the queue's operation limit keeps the cooldown this admission
        last saw, until it ends; one it never saw is none."""
        bound_s: Final = self.router.queue_settings.operation_timeout_ms / 1000
        try:
            left: Final = await _within(self.router.timer, bound_s, cooldown.left_s())
        except Exception:  # noqa: BLE001  # an unreadable cooldown falls back to the one this admission saw
            return cooldown.without_queue()
        return cooldown.without_queue() if left is None else cooldown.without_queue(still_cooling=left > 0)


class _ObservedCooldown:
    """One admission's cooldown lookup, keeping the end of the last cooldown it answered on the queue's timer."""

    def __init__(self, timer: Timer, lookup: Callable[[], Awaitable[float]]) -> None:
        self._timer = timer
        self._lookup = lookup
        self._ends_at: float | None = None

    async def left_s(self) -> float:
        left: Final = await self._lookup()
        self._ends_at = self._timer.now() + left
        return left

    def active(self) -> bool:
        return self._ends_at is not None and self._timer.now() < self._ends_at

    def without_queue(self, *, still_cooling: bool | None = None) -> WithoutQueue:
        return WithoutQueue(still_cooling=self.active() if still_cooling is None else still_cooling)


_CURRENT: Final[ContextVar[RequestQueue | None]] = ContextVar("dynamic_queue_request", default=None)


def current_request_queue() -> RequestQueue | None:
    return _CURRENT.get()


@contextlib.contextmanager
def bound_request_queue(request_queue: RequestQueue | None) -> Generator[None, None, None]:
    token: Final = _CURRENT.set(request_queue)
    try:
        yield
    finally:
        _CURRENT.reset(token)


Backends: TypeAlias = Callable[[QueueSettings], QueueBackend | None]


def on_redis(redis: Callable[[], ScriptHost | None]) -> Backends:
    """The queue on the router's Redis, read lazily since the proxy attaches it after the router is built."""

    def backend(settings: QueueSettings) -> QueueBackend | None:
        host: Final = redis()
        return None if host is None else RedisQueueBackend(host, settings)

    return backend


class DynamicQueueRouter:
    """With no backend to coordinate on, every request goes on without the queue until one is available."""

    def __init__(self, settings: DynamicQueueSettings, backends: Backends, timer: Timer | None = None) -> None:
        self.settings = settings
        self.queue_settings: Final = settings.queue_settings()
        self.queue_timeout_s: Final = settings.queue_timeout
        self.timer: Final[Timer] = timer or MonotonicTimer()
        self._backends = backends
        self._built: DynamicQueue | None = None

    def queue(self) -> DynamicQueue | CoordinationUnavailable:
        if self._built is not None:
            return self._built
        backend: Final = self._backends(self.queue_settings)
        if backend is None:
            return CoordinationUnavailable(reason=REDIS_NOT_CONFIGURED)
        self._built = DynamicQueue(backend, self.timer)
        return self._built

    def request(self, priority: int, request_id: str) -> RequestQueue:
        return RequestQueue(self, priority, request_id)

    def outcome_of(self, error: BaseException, retry_after_s: float | None) -> Outcome:
        from litellm.litellm_core_utils.exception_mapping_utils import provider_origin_status

        if provider_origin_status(error) not in TRIGGER_STATUS_CODES:
            return OtherFailure()
        return MatchingFailure(retry_after_ms=int((retry_after_s or 0.0) * 1000))


def _deployment_id(deployment: Mapping[str, object]) -> str | None:
    deployment_id: Final = (as_str_mapping(deployment.get("model_info")) or {}).get("id")
    return deployment_id if isinstance(deployment_id, str) and deployment_id else None


def capacity_of(deployment: Mapping[str, object]) -> str | None:
    """Each configured deployment waits on its own queue, keyed by its id alone."""
    deployment_id: Final = _deployment_id(deployment)
    return None if deployment_id is None else f"deployment:{deployment_id}"


__all__ = [
    "MAX_WAITING_LIMIT",
    "TRIGGER_STATUS_CODES",
    "Admission",
    "Backends",
    "DynamicQueueFullError",
    "DynamicQueueRouter",
    "DynamicQueueSettings",
    "DynamicQueueTimeoutError",
    "QueueRejectedError",
    "RequestQueue",
    "WithoutQueue",
    "bound_request_queue",
    "capacity_of",
    "current_request_queue",
    "on_redis",
    "parse_dynamic_queue",
    "rejection_error",
]
