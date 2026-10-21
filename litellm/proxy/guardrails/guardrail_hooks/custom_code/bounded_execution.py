"""Wall-clock bounds for sandboxed guardrail code.

Sync guardrail code runs on a dedicated daemon thread so a runaway loop never stalls the event loop; async
guardrail code is awaited as its own task. Either way the code's deadline is published through a context
variable, and the sandbox compiler routes every ``while`` test, ``for`` iteration and comprehension through
:func:`budget_ok`, which raises ``ExecutionInterrupted`` once that deadline has passed, whatever the code
catches around the loop body. As a backstop, a worker thread still running at the deadline has
``ExecutionInterrupted`` injected with ``PyThreadState_SetAsyncExc`` and a task still running is cancelled
repeatedly. A long-running C call (a catastrophic regex, for one) only sees any of this once it returns, so
the caller still gets its timeout on schedule while the worker keeps burning CPU until that call ends.
"""

import asyncio
import concurrent.futures
import contextvars
import ctypes
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

from litellm._logging import verbose_proxy_logger

T: Final = TypeVar("T")

_INTERRUPT_GRACE_SECONDS: Final = 1.0
_INTERRUPT_POLL_SECONDS: Final = 0.05
_deadline: Final[contextvars.ContextVar[float | None]] = contextvars.ContextVar("guardrail_code_deadline", default=None)


class ExecutionInterrupted(BaseException):
    """Raised inside guardrail code once its budget is spent; a BaseException so sandboxed
    ``except Exception`` clauses cannot swallow it."""


class ExecutionTimeoutError(Exception):
    """The guardrail code did not finish within its wall-clock budget."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"exceeded the {timeout:g}s execution timeout")
        self.timeout: Final = timeout


class SandboxExit(Exception):
    """Sandboxed code raised something outside the ``Exception`` tree (``SystemExit``, ``KeyboardInterrupt``,
    a bare ``BaseException``). It is delivered as an ordinary exception so it can neither stop the event loop
    nor pass for a timeout."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")


def _past_deadline() -> bool:
    deadline: Final = _deadline.get()
    return deadline is not None and time.monotonic() > deadline


def budget_ok() -> bool:
    """Bound to ``_budget_ok_`` in the sandbox, where every ``while`` test starts with a call to it."""
    if _past_deadline():
        raise ExecutionInterrupted
    return True


def budgeted_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Bound to ``_getiter_`` in the sandbox, so every ``for`` loop and comprehension checks the budget per item."""
    for item in iterable:
        budget_ok()
        yield item


class _InterruptGate:
    """Aims the interrupt at the worker thread only while it is inside the sandboxed call, so a thread id the
    OS recycles after the worker exits is never hit."""

    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._thread_id: int | None = None

    def open(self) -> None:
        self._thread_id = threading.get_ident()

    def close(self) -> None:
        with self._lock:
            self._thread_id = None

    def is_open(self) -> bool:
        return self._thread_id is not None

    def interrupt(self) -> bool:
        with self._lock:
            if self._thread_id is None:
                return False
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._thread_id), ctypes.py_object(ExecutionInterrupted)
            )
            return True


@dataclass(frozen=True, slots=True)
class _Worker(Generic[T]):
    thread: threading.Thread
    outcome: concurrent.futures.Future[T]
    gate: _InterruptGate


def _run(fn: Callable[[], T], timeout: float, gate: _InterruptGate) -> tuple[T | None, Exception | None]:
    _deadline.set(time.monotonic() + timeout)
    gate.open()
    try:
        result: Final = fn()
    except Exception as e:  # noqa: BLE001  # every failure is handed to the waiting caller through the future
        return None, e
    except ExecutionInterrupted:
        return None, ExecutionTimeoutError(timeout)
    except BaseException as e:  # noqa: BLE001  # a SystemExit must reach the caller as a failure, not end the worker silently
        return None, SandboxExit(e)
    finally:
        gate.close()
    if _past_deadline():
        return None, ExecutionTimeoutError(timeout)
    return result, None


def _deliver(fn: Callable[[], T], timeout: float, outcome: concurrent.futures.Future[T], gate: _InterruptGate) -> None:
    try:
        _settle(outcome, *_run(fn, timeout, gate))
    except ExecutionInterrupted:
        _settle(outcome, exception=ExecutionTimeoutError(timeout))


def _settle(outcome: concurrent.futures.Future[T], result: T | None = None, exception: Exception | None = None) -> None:
    try:
        if exception is not None:
            outcome.set_exception(exception)
        else:
            outcome.set_result(result)  # pyright: ignore[reportArgumentType]  # result is T whenever exception is None
    except concurrent.futures.InvalidStateError:
        return


def _start_worker(fn: Callable[[], T], timeout: float, label: str) -> _Worker[T]:
    outcome: Final[concurrent.futures.Future[T]] = concurrent.futures.Future()
    gate: Final = _InterruptGate()
    thread: Final = threading.Thread(
        target=_deliver, args=(fn, timeout, outcome, gate), name=f"guardrail-code:{label}", daemon=True
    )
    thread.start()
    return _Worker(thread, outcome, gate)


def _interrupt(worker: _Worker[T]) -> None:
    deadline: Final = time.monotonic() + _INTERRUPT_GRACE_SECONDS
    while worker.gate.interrupt() and time.monotonic() < deadline:
        worker.thread.join(_INTERRUPT_POLL_SECONDS)
    if worker.gate.is_open():
        verbose_proxy_logger.error(
            "%s is still running after its timeout; it is stuck in a call Python cannot interrupt", worker.thread.name
        )


def call_with_timeout(fn: Callable[[], T], timeout: float, label: str) -> T:
    """Run ``fn`` on a worker thread and wait for it, from sync code."""
    worker: Final = _start_worker(fn, timeout, label)
    try:
        return worker.outcome.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        worker.outcome.cancel()
        _interrupt(worker)
        raise ExecutionTimeoutError(timeout) from None


async def call_off_loop_with_timeout(fn: Callable[[], T], timeout: float, label: str) -> T:
    """Run ``fn`` on a worker thread and await it without blocking the event loop."""
    worker: Final = _start_worker(fn, timeout, label)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(worker.outcome), timeout)
    except asyncio.TimeoutError:
        await asyncio.to_thread(_interrupt, worker)
        raise ExecutionTimeoutError(timeout) from None
    except asyncio.CancelledError:
        threading.Thread(target=_interrupt, args=(worker,), name=f"guardrail-interrupt:{label}", daemon=True).start()
        raise


def _discard_outcome(task: asyncio.Future[T]) -> None:
    if not task.cancelled():
        task.exception()


async def _cancel(task: asyncio.Task[T], label: str) -> None:
    deadline: Final = time.monotonic() + _INTERRUPT_GRACE_SECONDS
    while not task.done() and time.monotonic() < deadline:
        task.cancel()
        await asyncio.wait((task,), timeout=_INTERRUPT_POLL_SECONDS)
    if not task.done():
        verbose_proxy_logger.error(
            "guardrail-code:%s is still running after its timeout; it keeps swallowing cancellation", label
        )


async def _contain(pending: Awaitable[T], timeout: float) -> T:
    try:
        result: Final = await pending
    except (Exception, asyncio.CancelledError):
        raise
    except ExecutionInterrupted:
        raise ExecutionTimeoutError(timeout) from None
    except BaseException as e:  # noqa: BLE001  # a SystemExit escaping a task stops the whole event loop
        raise SandboxExit(e) from e
    if _past_deadline():
        raise ExecutionTimeoutError(timeout)
    return result


async def await_with_timeout(pending: Awaitable[object], timeout: float, label: str) -> object:
    """Await ``pending`` on the event loop and give it up at the deadline, even if it swallows cancellation."""
    context: Final = contextvars.copy_context()
    context.run(_deadline.set, time.monotonic() + timeout)
    task: Final = context.run(asyncio.ensure_future, _contain(pending, timeout))
    task.add_done_callback(_discard_outcome)
    try:
        await asyncio.wait((task,), timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if task.done():
        return task.result()
    await _cancel(task, label)
    raise ExecutionTimeoutError(timeout)
