import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from typing import Final

from litellm._logging import verbose_proxy_logger

DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT: Final = 100

FIRE_AND_FORGET_POST_TIMEOUT_SECONDS: Final = 30.0

FIRE_AND_FORGET_DISPATCHED_REASON: Final = "fire_and_forget dispatched, verdict not read"

FIRE_AND_FORGET_DROPPED_REASON: Final = "fire_and_forget_max_inflight reached, call dropped"

FIRE_AND_FORGET_NOT_DISPATCHED_REASON: Final = "fire_and_forget payload could not be built, call not dispatched"

_DROP_LOG_INTERVAL: Final = 100


def resolve_max_inflight(value: object) -> int:
    if value is None:
        return DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"fire_and_forget_max_inflight must be an int, got {value!r}")
    return value


class BackgroundDispatcher:
    """Runs calls as detached tasks, dropping (and counting) calls once ``max_inflight`` are outstanding."""

    def __init__(self, *, guardrail_name: str | None, max_inflight: int) -> None:
        if max_inflight < 1:
            raise ValueError(f"fire_and_forget_max_inflight must be >= 1 (got {max_inflight})")
        self._guardrail_name: Final = guardrail_name
        self._max_inflight: Final = max_inflight
        self._pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs, asyncio keeps only weak ones
        self._dropped: int = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def dispatch(self, run: Callable[[], Awaitable[None]], *, context: str) -> bool:
        if len(self._pending) >= self._max_inflight:
            self._dropped += 1
            if self._dropped % _DROP_LOG_INTERVAL == 1:
                verbose_proxy_logger.warning(
                    "Generic Guardrail API (%s, fire_and_forget): dropped %d call(s) so far, "
                    "%d already in flight (fire_and_forget_max_inflight=%d). %s",
                    self._guardrail_name,
                    self._dropped,
                    len(self._pending),
                    self._max_inflight,
                    context,
                )
            return False

        task: Final = contextvars.Context().run(asyncio.create_task, self._run_logging_failures(run, context=context))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return True

    async def _run_logging_failures(self, run: Callable[[], Awaitable[None]], *, context: str) -> None:
        try:
            await run()
        except Exception as e:  # noqa: BLE001  # a detached task has no caller to raise into
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s, fire_and_forget) call failed. %s: %s",
                self._guardrail_name,
                context,
                e,
            )

    async def wait_for_pending(self) -> None:
        pending: Final = tuple(self._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
