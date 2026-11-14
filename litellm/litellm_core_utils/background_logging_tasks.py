"""Strong-referenced scheduling for fire-and-forget logging tasks.

The event loop keeps only a weak reference to tasks, so a detached
``asyncio.create_task`` can be garbage collected before it finishes and its
spend log row is silently never written. Scheduling through :func:`spawn_logging_task`
keeps a strong reference until the task completes, logs its failures, and lets
proxy shutdown wait for stragglers through :func:`drain_logging_tasks`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Final

from litellm._logging import verbose_logger

_PENDING_LOGGING_TASKS: Final[set[asyncio.Task[object]]] = set()  # mutable-ok: strong refs to pending logging tasks


def log_background_task_failure(task: asyncio.Task[object], *, task_name: str) -> None:
    if task.cancelled():
        return
    exception: Final = task.exception()
    if exception is not None:
        verbose_logger.error("%s failed: %s", task_name, exception)


def spawn_logging_task(
    running_loop: asyncio.AbstractEventLoop, coroutine: Coroutine[object, object, object], *, task_name: str
) -> asyncio.Task[object]:
    task: Final = running_loop.create_task(coroutine)
    _PENDING_LOGGING_TASKS.add(task)
    task.add_done_callback(_PENDING_LOGGING_TASKS.discard)
    task.add_done_callback(lambda done: log_background_task_failure(done, task_name=task_name))
    return task


async def drain_logging_tasks(timeout: float = 5.0) -> None:
    # A task that finished as its loop closed never ran its discard callback, and a
    # task from another loop cannot be awaited here: forget both, wait on the rest.
    running_loop: Final = asyncio.get_running_loop()
    stale: Final = {task for task in _PENDING_LOGGING_TASKS if task.done() or task.get_loop() is not running_loop}
    _PENDING_LOGGING_TASKS.difference_update(stale)
    if not _PENDING_LOGGING_TASKS:
        return
    _, pending = await asyncio.wait(tuple(_PENDING_LOGGING_TASKS), timeout=timeout)
    if pending:
        verbose_logger.warning("Timed out draining logging tasks on shutdown: %d tasks still alive", len(pending))
