import asyncio
import functools
import inspect
import itertools
import logging
import threading
import time
import weakref
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import FrameType, MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar, cast  # noqa: TID251  # legacy Redis exception cast for old stubs

from litellm._internal_context import current_service_caller
from litellm._logging import verbose_logger
from litellm.constants import REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION, REDIS_TIMEOUT_LOG_INTERVAL
from litellm.redis.transport.contract import RedisFailure

if TYPE_CHECKING:
    from litellm._service_logger import ServiceLogging

__all__ = (
    "RedisCircuitBreaker",
    "RedisCircuitBreakerOpenError",
    "_BreakerAdmission",
    "_RedisTimeoutLogThrottle",
    "_caller_frames",
    "_ends_call_stack",
    "_enter_circuit_breaker",
    "_exit_circuit_breaker",
    "_explicit_causes",
    "_get_call_stack_info",
    "_is_generic_caller_frame",
    "_is_redis_health_failure",
    "_log_throttling_timeouts",
    "_record_swallowed_redis_failure",
    "_redis_health_error_types",
    "_redis_timeout_error_types",
    "_run_under_circuit_breaker",
    "_run_under_circuit_breaker_sync",
    "is_redis_error",
    "is_redis_timeout_failure",
    "log_redis_failure",
    "log_redis_failure_value",
    "redis_breaker_state_counts",
)


_GENERIC_CALLER_MODULES: Final = frozenset(
    {
        "caching.redis_cache",
        "caching.redis_batch",
        "caching.dual_cache",
        "caching.caching",
        "rust_bridge.execution",
        "contextlib",
    }
)
_GENERIC_CALLER_FRAME_NAMES: Final = frozenset(
    {
        "<lambda>",
        "wrapper",
        "_run_under_circuit_breaker",
        "_run_under_circuit_breaker_sync",
        "run_alone",
        "_settle_alone",
        "get_cache",
        "set_cache",
        "async_get_cache",
        "async_set_cache",
        "async_batch_get_cache",
        "async_batch_get_cache_shared",
        "async_set_cache_pipeline",
        "async_increment_cache",
        "async_delete_cache",
    }
)
_CALL_STACK_END_MODULES: Final = ("asyncio", "concurrent", "threading")


def _is_generic_caller_frame(frame: FrameType) -> bool:
    module: Final = frame.f_globals.get("__name__")
    return (
        (isinstance(module, str) and module.removeprefix("litellm.") in _GENERIC_CALLER_MODULES)
        or (isinstance(module, str) and module.startswith("litellm.redis."))
        or frame.f_code.co_name in _GENERIC_CALLER_FRAME_NAMES
    )


def _ends_call_stack(frame: FrameType) -> bool:
    module: Final = frame.f_globals.get("__name__")
    return isinstance(module, str) and module.startswith(_CALL_STACK_END_MODULES)


def _caller_frames(first: FrameType) -> Iterator[FrameType]:
    frame: FrameType | None = first
    while frame is not None and not _ends_call_stack(frame):
        yield frame
        frame = frame.f_back


def _get_call_stack_info(num_frames: int = 2) -> str:
    """
    Get the function names of the nearest meaningful callers of the cache method.

    Frames that merely forward the call (this module's circuit-breaker guards, the
    cache facades, the batch pipeline's retry path, generic cache verbs) are
    skipped, and the walk stops at the event loop, so the chain names the litellm
    code that wanted the call. When nothing but forwarding frames is found (the call
    runs in a task of its own, like a batch op retried on the flush) the chain the
    declaring code threaded through ``service_caller`` is reported, else ``unknown``.

    Args:
        num_frames: Number of previous frames to include (default: 2)

    Returns:
        A string with format "caller_function [<- grandparent_function]"
    """
    try:
        current_frame: Final = inspect.currentframe()
        if current_frame is None:
            return "unknown"

        # Skip this function and the immediate caller (which sets call_type)
        f_back: Final = current_frame.f_back
        if f_back is None:
            return "unknown"
        first: Final = f_back.f_back
        if first is None:
            return "unknown"
        frames: Final = _caller_frames(first)
        leading: Final = tuple(itertools.islice(frames, num_frames))
        leading_names: Final = tuple(frame.f_code.co_name for frame in leading if not _is_generic_caller_frame(frame))
        further_names: Final = tuple(
            itertools.islice(
                (frame.f_code.co_name for frame in frames if not _is_generic_caller_frame(frame)),
                num_frames - len(leading_names),
            )
        )
        function_names: Final = leading_names + further_names

        if function_names:
            return " <- ".join(function_names)
        return current_service_caller() or "unknown"
    except Exception:
        return "unknown"


class RedisCircuitBreaker:
    """
    Tracks Redis health for a RedisCache instance.

    States:
      CLOSED    - normal, Redis is called
      OPEN      - Redis is down, raise immediately (no network call)
      HALF_OPEN - recovery probe: allow one request through

    Transitions:
      CLOSED    -> OPEN      after failure_threshold consecutive hard connectivity
                             failures, or after an unbroken run of timeout failures
                             (no success or hard failure in between) that reaches
                             failure_threshold and spans timeout_min_duration seconds
      OPEN      -> HALF_OPEN after recovery_timeout seconds
      HALF_OPEN -> CLOSED    on success
      HALF_OPEN -> OPEN      on failure (resets timer)

    Timeouts are accounted separately from hard connectivity failures because the async
    Redis timeout includes time waiting for the worker event loop to resume: one loop
    stall makes every in-flight operation time out together, which satisfies a purely
    consecutive threshold instantly even though Redis is healthy. Requiring a
    timeout-only streak to also span timeout_min_duration filters such bursts while a
    real outage that surfaces as timeouts still opens the breaker after that duration.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: int,
        enabled: bool = True,
        timeout_min_duration: float = REDIS_CIRCUIT_BREAKER_TIMEOUT_MIN_DURATION,
        service_logger: "ServiceLogging | None" = None,
    ) -> None:
        self._service_logger = service_logger
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.enabled = enabled
        self.timeout_min_duration = timeout_min_duration
        self._failure_count = 0
        self._hard_failure_count = 0
        self._timeout_count = 0
        self._timeout_streak_started_at: float | None = None
        self._opened_at: float | None = None
        self._state = self.CLOSED
        self._generation = 0
        _LIVE_BREAKERS.add(self)
        self._report(transitions_to=None, failure_class=None)

    @property
    def state(self) -> str:
        return self._state

    @property
    def generation(self) -> int:
        """Counts state transitions, so a call can tell whether the breaker moved while it ran."""
        return self._generation

    def is_open(self) -> bool:
        """Returns True if Redis calls should be skipped."""
        if not self.enabled:
            return False
        if self._state == self.HALF_OPEN:
            # Probe already in flight — fast-fail all concurrent requests.
            # Only the one call that caused the OPEN→HALF_OPEN transition
            # (which returned False) is the designated probe.
            return True
        if self._state == self.OPEN:
            if time.time() - (self._opened_at or 0) > self.recovery_timeout:
                self._set_state(self.HALF_OPEN)
                return False  # this caller is the designated probe
            return True
        return False

    def _should_open(self, now: float) -> bool:
        if self._state == self.HALF_OPEN:
            return True
        if self._hard_failure_count >= self.failure_threshold:
            return True
        if self._timeout_count < self.failure_threshold:
            return False
        return now - (self._timeout_streak_started_at or now) >= self.timeout_min_duration

    def record_failure(self, is_timeout: bool = False) -> None:
        if not self.enabled:
            return
        now: Final = time.time()
        self._failure_count += 1
        if is_timeout:
            self._timeout_count += 1
            if self._timeout_streak_started_at is None:
                self._timeout_streak_started_at = now
        else:
            self._hard_failure_count += 1
            self._timeout_count = 0
            self._timeout_streak_started_at = None
        self._opened_at = now
        self._report(transitions_to=None, failure_class="timeout" if is_timeout else "connectivity")
        if self._should_open(now):
            if self._state != self.OPEN:
                verbose_logger.warning(
                    "Redis circuit breaker OPENED after %d consecutive failures"
                    " (%d hard connectivity) — fast-failing Redis calls for %ds",
                    self._failure_count,
                    self._hard_failure_count,
                    self.recovery_timeout,
                )
            self._set_state(self.OPEN)

    def record_success(self) -> None:
        if not self.enabled or self._state == self.OPEN:
            return
        if self._state == self.HALF_OPEN:
            verbose_logger.info("Redis circuit breaker CLOSED — Redis recovered")
        self._failure_count = 0
        self._hard_failure_count = 0
        self._timeout_count = 0
        self._timeout_streak_started_at = None
        self._set_state(self.CLOSED)

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._generation += 1
        self._report(transitions_to=state, failure_class=None)

    def _report(self, transitions_to: str | None, failure_class: str | None) -> None:
        if self._service_logger is not None:
            self._service_logger.redis_breaker_hook(transitions_to=transitions_to, failure_class=failure_class)


_RedisCallResult = TypeVar("_RedisCallResult")


_swallowed_redis_failures: Final[ContextVar[int]] = ContextVar("litellm_swallowed_redis_failures", default=0)


@functools.lru_cache(maxsize=1)
def _redis_health_error_types() -> tuple[type, ...]:
    """Exception types that mean the Redis backend itself is unhealthy.

    Command and data errors say nothing about connectivity: an INCR against a non-numeric
    value or an undecodable cached entry is a request problem, and counting those would let
    a caller trip the shared breaker on demand, dropping rate limiting to per-process
    counters that spreading traffic across replicas can outrun.

    Imported lazily because this module is reachable from a base ``import litellm`` while
    redis is not a base dependency.
    """
    from redis.exceptions import (  # noqa: TID251  # shared Redis exception classification
        BusyLoadingError,
        ClusterDownError,
    )
    from redis.exceptions import (  # noqa: TID251  # shared Redis exception classification
        ConnectionError as RedisConnectionError,
    )
    from redis.exceptions import (  # noqa: TID251  # shared Redis exception classification
        TimeoutError as RedisTimeoutError,
    )

    return (RedisConnectionError, RedisTimeoutError, BusyLoadingError, ClusterDownError, OSError, asyncio.TimeoutError)


def _is_redis_health_failure(exc: BaseException) -> bool:
    """True when ``exc`` indicates Redis is unreachable rather than the request being bad."""
    try:
        return isinstance(exc, _redis_health_error_types())
    except ImportError:
        return True


@functools.lru_cache(maxsize=1)
def _redis_timeout_error_types() -> tuple[type, ...]:
    """Health failures that are timeouts rather than unambiguous connectivity errors.

    ``builtins.TimeoutError`` covers ``socket.timeout`` (an alias since py3.10) and, from
    py3.11, ``asyncio.TimeoutError``; on py3.10 ``asyncio.TimeoutError`` is still its own
    class, so it is listed explicitly. ``redis.exceptions.TimeoutError`` subclasses neither.
    ``MaxConnectionsError`` is the async cluster client's pool-exhausted signal (its nodes have
    no blocking pool, so it is raised instantly instead of after a wait): a busy pool, not a
    dead Redis, like the standalone pool's chained wait timeout.
    """
    try:
        from redis import exceptions as redis_exceptions  # noqa: TID251  # shared Redis exception classification
    except ImportError:
        return (TimeoutError, asyncio.TimeoutError)
    max_connections_error: Final = cast(  # cast-ok: types-redis 4.6 stubs predate MaxConnectionsError
        "type[Exception]",
        redis_exceptions.MaxConnectionsError,  # pyright: ignore[reportAttributeAccessIssue]  # types-redis 4.6 stubs predate it
    )
    return (redis_exceptions.TimeoutError, TimeoutError, asyncio.TimeoutError, max_connections_error)


_MAX_EXCEPTION_CAUSE_DEPTH: Final = 20


def _explicit_causes(exc: BaseException) -> Iterator[BaseException]:
    current = exc  # rebind-ok: advances one link per iteration of the bounded walk
    for _ in range(_MAX_EXCEPTION_CAUSE_DEPTH):
        yield current
        if current.__cause__ is None:
            return
        current = current.__cause__


def is_redis_timeout_failure(exc: BaseException) -> bool:
    """True when ``exc`` or any exception it was explicitly raised ``from`` is a timeout.

    redis-py's blocking pool reports a pool wait timeout as ``ConnectionError`` chained from
    ``asyncio.TimeoutError``, which is a busy pool rather than an unreachable Redis.
    """
    timeout_types: Final = _redis_timeout_error_types()
    return any(isinstance(link, timeout_types) for link in _explicit_causes(exc))


def _record_swallowed_redis_failure(breaker: RedisCircuitBreaker, exc: BaseException) -> None:
    """Record a Redis failure that the calling method is about to swallow.

    The marker is a ContextVar rather than a counter on the breaker because breakers are
    shared by every concurrent caller. A plain shared counter cannot tell "my call failed"
    from "some other in-flight call failed", so a success overlapping someone else's
    failure would be discarded and a Redis that is answering would still be evicted.
    asyncio gives each task its own copy of the context, so this is per-call.
    """
    if not _is_redis_health_failure(exc):
        return
    breaker.record_failure(is_timeout=is_redis_timeout_failure(exc))
    _swallowed_redis_failures.set(_swallowed_redis_failures.get() + 1)


_LIVE_BREAKERS: Final[weakref.WeakSet[RedisCircuitBreaker]] = weakref.WeakSet()


def redis_breaker_state_counts() -> Mapping[str, int]:
    """How many live Redis circuit breakers sit in each state, for the Prometheus services logger."""
    return MappingProxyType(dict(Counter(breaker.state for breaker in _LIVE_BREAKERS)))


class RedisCircuitBreakerOpenError(Exception):
    pass


class _RedisTimeoutLogThrottle:
    """Admits one Redis timeout log line per interval and counts the timeouts it suppressed in between."""

    def __init__(self, interval: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._last_logged_at: float | None = None
        self._suppressed = 0

    def admit(self) -> int | None:
        """Return the number of timeouts suppressed since the last admitted line, or None to suppress this one."""
        with self._lock:
            now: Final = self._clock()
            if self._last_logged_at is not None and now - self._last_logged_at < self.interval:
                self._suppressed += 1
                return None
            suppressed: Final = self._suppressed
            self._suppressed = 0
            self._last_logged_at = now
            return suppressed


_redis_timeout_log_throttle: Final = _RedisTimeoutLogThrottle(REDIS_TIMEOUT_LOG_INTERVAL)


def log_redis_failure(
    logger: logging.Logger, level: int, message: str, exc: BaseException, with_traceback: bool = False
) -> None:
    if isinstance(exc, RedisCircuitBreakerOpenError):
        logger.debug("%s: %s", message, exc, stacklevel=2)
        return
    _log_throttling_timeouts(
        logger, level, message, exc, timeout=is_redis_timeout_failure(exc), exc_info=exc if with_traceback else None
    )


def log_redis_failure_value(logger: logging.Logger, level: int, message: str, failure: RedisFailure) -> None:
    if failure.kind == "breaker_open":
        logger.debug("%s: %s", message, failure.message, stacklevel=2)
        return
    _log_throttling_timeouts(logger, level, message, failure.message, timeout=failure.timeout, exc_info=None)


def _log_throttling_timeouts(
    logger: logging.Logger,
    level: int,
    message: str,
    detail: object,
    *,
    timeout: bool,
    exc_info: BaseException | None,
) -> None:
    if not timeout:
        logger.log(level, "%s: %s", message, detail, exc_info=exc_info, stacklevel=3)
        return
    suppressed: Final = _redis_timeout_log_throttle.admit()
    if suppressed is None:
        logger.debug("%s: %s", message, detail, stacklevel=3)
        return
    if suppressed == 0:
        logger.log(level, "%s: %s", message, detail, exc_info=exc_info, stacklevel=3)
        return
    logger.log(
        level,
        "%s: %s (%d more Redis timeouts since the previous Redis timeout line were logged at DEBUG)",
        message,
        detail,
        suppressed,
        exc_info=exc_info,
        stacklevel=3,
    )


@dataclass(frozen=True, slots=True)
class _BreakerAdmission:
    swallowed_before: int
    generation: int


def _enter_circuit_breaker(breaker: RedisCircuitBreaker, name: str) -> _BreakerAdmission:
    """Reject the call if the breaker is open, else record what its success may later prove."""
    if breaker.is_open():
        raise RedisCircuitBreakerOpenError(f"Redis circuit breaker is open — skipping {name}")
    return _BreakerAdmission(swallowed_before=_swallowed_redis_failures.get(), generation=breaker.generation)


def _exit_circuit_breaker(breaker: RedisCircuitBreaker, admission: _BreakerAdmission) -> None:
    """Record success only when nothing failed while the call ran and the breaker has not moved since.

    Several Redis methods catch their own connection errors and return a default, so a
    method that returned is not on its own proof of a healthy Redis. A success also vouches
    only for the breaker state that admitted the call: a call admitted before the breaker
    opened, or a probe admitted before a later failure reopened it, finishes knowing nothing
    about whether Redis has recovered since, so only the current probe may close the breaker.
    """
    if _swallowed_redis_failures.get() != admission.swallowed_before:
        return
    if breaker.generation != admission.generation:
        return
    breaker.record_success()


async def _run_under_circuit_breaker(
    breaker: RedisCircuitBreaker,
    name: str,
    call: Callable[[], Awaitable[_RedisCallResult]],
) -> _RedisCallResult:
    """Run one Redis coroutine under a circuit breaker.

    Shared by the method decorator and the Lua script executor so both feed the same
    health signal.
    """
    admission: Final = _enter_circuit_breaker(breaker, name)
    try:
        result: Final = await call()
    except Exception as e:
        if _is_redis_health_failure(e):
            breaker.record_failure(is_timeout=is_redis_timeout_failure(e))
        raise
    _exit_circuit_breaker(breaker, admission)
    return result


def is_redis_error(exc: BaseException) -> bool:
    from redis.exceptions import RedisError  # noqa: TID251  # shared Redis exception classification

    return isinstance(exc, RedisError)


def _run_under_circuit_breaker_sync(
    breaker: RedisCircuitBreaker,
    name: str,
    call: Callable[[], _RedisCallResult],
) -> _RedisCallResult:
    """Run one blocking Redis call under a circuit breaker, feeding the same health signal as the async path."""
    admission: Final = _enter_circuit_breaker(breaker, name)
    try:
        result: Final = call()
    except Exception as e:
        if _is_redis_health_failure(e):
            breaker.record_failure(is_timeout=is_redis_timeout_failure(e))
        raise
    _exit_circuit_breaker(breaker, admission)
    return result
