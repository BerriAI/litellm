"""Off-critical-path dispatch for the Generic Guardrail API (``fire_and_forget``).

The event loop only holds a weak reference to a task created with
``asyncio.create_task``, so a task with no other referrer can be garbage
collected mid-flight and silently cancelled. Every dispatched task is therefore
kept in a strong-ref set until it completes.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Final

from litellm._logging import verbose_proxy_logger

DEFAULT_FIRE_AND_FORGET_MAX_INFLIGHT: Final = 100

# Log one warning per this many drops, so a saturated endpoint cannot turn into a
# log flood of its own.
_DROP_LOG_INTERVAL: Final = 100


class BackgroundDispatcher:
    """Runs guardrail calls detached from the request, with a bounded queue.

    Async dispatch decouples the request rate from the endpoint's throughput, so
    a slow endpoint would otherwise pile up tasks without limit. Once
    ``max_inflight`` tasks are outstanding, further calls are dropped and
    counted rather than queued.
    """

    def __init__(self, *, guardrail_name: str | None, max_inflight: int) -> None:
        if max_inflight < 1:
            raise ValueError(f"fire_and_forget_max_inflight must be >= 1 (got {max_inflight})")
        self._guardrail_name: Final = guardrail_name
        self._max_inflight: Final = max_inflight
        # Strong refs to in-flight tasks; entries are discarded on completion.
        self._pending: set[asyncio.Task[None]] = set()  # mutable-ok: tasks are added/removed as they complete
        self._dropped: int = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def dispatch(self, run: Callable[[], Awaitable[None]], *, context: str) -> bool:
        """Start ``run`` detached. Returns False when dropped for backpressure."""
        if len(self._pending) >= self._max_inflight:
            self._dropped += 1
            if self._dropped % _DROP_LOG_INTERVAL == 1:
                verbose_proxy_logger.warning(
                    "Generic Guardrail API (%s, fire_and_forget): dropped %d call(s) so far; "
                    "%d already in flight (fire_and_forget_max_inflight=%d). %s",
                    self._guardrail_name,
                    self._dropped,
                    len(self._pending),
                    self._max_inflight,
                    context,
                )
            return False

        task: Final = asyncio.create_task(self._guarded(run, context=context))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return True

    async def _guarded(self, run: Callable[[], Awaitable[None]], *, context: str) -> None:
        """Never raise: a detached task's exception has no frame to surface in."""
        try:
            await run()
        except Exception as e:  # noqa: BLE001  # a detached task has no frame to raise into
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s, fire_and_forget) call failed. %s: %s",
                self._guardrail_name,
                context,
                e,
            )

    async def wait_for_pending(self) -> None:
        """Await every in-flight task. Used by tests and best-effort teardown."""
        pending: Final = tuple(self._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
