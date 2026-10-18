"""One request's async logging run paired with the sync run that also covers it, so a CustomLogger with both
hooks gets one of them per request: the async hook when the async run reaches the logger, the sync hook otherwise."""

import contextvars
import functools
import threading
from collections.abc import Callable, Mapping
from typing import Final

from litellm._logging import verbose_logger


def run_held_hook_in_executor(sync_hook: Callable[[], None]) -> None:
    """Runs a held sync hook off the event loop, the way the sync run would have, or here once the executor is shut down."""
    from litellm.litellm_core_utils.thread_pool_executor import executor

    try:
        executor.submit(sync_hook)
    except RuntimeError:
        sync_hook()


def run_held_hook_now(sync_hook: Callable[[], None]) -> None:
    sync_hook()


class HookPairing:
    """The sync run holds back the sync hook of each logger the async run is expected to reach, and the async run marks
    each logger it hands its async hook. Once the async run ends, however it ends (finished, returned early, raised,
    cancelled by the logging worker's timeout), every held hook whose logger it never reached runs after all."""

    def __init__(self, run_held: Callable[[Callable[[], None]], None]) -> None:
        self._run_held: Final = run_held
        self._lock: Final = threading.Lock()
        self._reached: frozenset[int] = frozenset()
        self._held: Mapping[int, Callable[[], None]] = {}
        self._ended = False

    def reach(self, logger: object) -> None:
        """The async run is handing ``logger`` its async hook."""
        with self._lock:
            self._reached = self._reached | {id(logger)}
            self._held = {key: hook for key, hook in self._held.items() if key != id(logger)}

    def hold(self, logger: object, sync_hook: Callable[[], None]) -> None:
        """Runs ``sync_hook`` unless the async run reaches ``logger``: now when the async run already ended without
        reaching it, else once it ends."""
        context_bound: Final = functools.partial(contextvars.copy_context().run, _non_blocking(logger, sync_hook))
        with self._lock:
            if id(logger) in self._reached:
                return
            if not self._ended:
                self._held = {**self._held, id(logger): context_bound}
                return
        context_bound()

    def end(self) -> None:
        with self._lock:
            self._ended = True
            unreached: Final = tuple(self._held.values())
            self._held = {}
        for sync_hook in unreached:
            self._run_held(sync_hook)


def _non_blocking(logger: object, sync_hook: Callable[[], None]) -> Callable[[], None]:
    def run() -> None:
        try:
            sync_hook()
        except Exception as e:  # noqa: BLE001  # a logger's own error never stops the request's other logging
            verbose_logger.exception(
                "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging %s\nCallback=%s", e, logger
            )

    return run
