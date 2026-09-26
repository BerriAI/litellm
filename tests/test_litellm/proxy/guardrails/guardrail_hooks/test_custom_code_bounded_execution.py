import asyncio
import threading
import time

import pytest

from litellm.proxy.guardrails.guardrail_hooks.custom_code.bounded_execution import (
    ExecutionTimeoutError,
    SandboxExit,
    await_with_timeout,
    call_off_loop_with_timeout,
    call_with_timeout,
)


def _worker_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("guardrail-code:")]


def _spin_forever() -> None:
    n = 0
    while True:
        n += 1


def _spin_swallowing_exceptions() -> None:
    while True:
        try:
            _spin_forever()
        except Exception:
            continue


async def _swallow_cancellations_for(seconds: float) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            await asyncio.sleep(deadline - time.monotonic())
        except asyncio.CancelledError:
            continue
    return "survived"


def _exit_now() -> None:
    raise SystemExit("bye")


def test_call_with_timeout_returns_the_result_and_reraises_failures():
    assert call_with_timeout(lambda: 42, 1.0, label="ok") == 42
    with pytest.raises(ZeroDivisionError):
        call_with_timeout(lambda: 1 // 0, 1.0, label="boom")


def test_call_with_timeout_delivers_a_system_exit_at_once():
    started = time.monotonic()

    with pytest.raises(SandboxExit, match="SystemExit: bye"):
        call_with_timeout(_exit_now, 5.0, label="exit")

    assert time.monotonic() - started < 1.0


async def _exit_later() -> None:
    await asyncio.sleep(0)
    raise SystemExit("bye")


@pytest.mark.asyncio
async def test_await_with_timeout_contains_a_system_exit_instead_of_stopping_the_loop():
    with pytest.raises(SandboxExit, match="SystemExit: bye"):
        await await_with_timeout(_exit_later(), 1.0, label="exit")

    assert await asyncio.sleep(0, result="loop still running") == "loop still running"


@pytest.mark.parametrize("fn", [_spin_forever, _spin_swallowing_exceptions])
def test_call_with_timeout_interrupts_a_busy_loop_and_reclaims_the_thread(fn):
    started = time.monotonic()

    with pytest.raises(ExecutionTimeoutError, match=r"exceeded the 0\.2s execution timeout") as exc_info:
        call_with_timeout(fn, 0.2, label="spin")

    assert exc_info.value.timeout == 0.2
    assert time.monotonic() - started < 1.5
    time.sleep(0.2)
    assert _worker_threads() == []


@pytest.mark.asyncio
async def test_call_off_loop_with_timeout_keeps_the_loop_running_and_stops_the_worker():
    ticks = 0

    async def tick_forever() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    ticker = asyncio.create_task(tick_forever())
    try:
        assert await call_off_loop_with_timeout(lambda: "done", 1.0, label="ok") == "done"
        with pytest.raises(ExecutionTimeoutError):
            await call_off_loop_with_timeout(_spin_forever, 0.3, label="spin")
    finally:
        ticker.cancel()

    assert ticks >= 5
    await asyncio.sleep(0.2)
    assert _worker_threads() == []


def _stragglers() -> list[asyncio.Task[object]]:
    return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]


@pytest.mark.asyncio
async def test_await_with_timeout_keeps_cancelling_a_coroutine_that_swallows_cancellation():
    assert await await_with_timeout(_swallow_cancellations_for(0.0), 1.0, label="ok") == "survived"
    started = time.monotonic()

    with pytest.raises(ExecutionTimeoutError, match=r"exceeded the 0\.1s execution timeout"):
        await await_with_timeout(_swallow_cancellations_for(0.4), 0.1, label="stubborn")

    assert time.monotonic() - started < 1.0
    assert _stragglers() == []


@pytest.mark.asyncio
async def test_await_with_timeout_abandons_a_coroutine_that_never_stops_swallowing_cancellation():
    started = time.monotonic()

    with pytest.raises(ExecutionTimeoutError):
        await await_with_timeout(_swallow_cancellations_for(2.0), 0.1, label="stubborn")

    elapsed = time.monotonic() - started
    assert 1.0 <= elapsed < 1.8
    stragglers = _stragglers()
    assert len(stragglers) == 1
    with pytest.raises(ExecutionTimeoutError):
        await asyncio.gather(*stragglers)


@pytest.mark.asyncio
async def test_call_off_loop_with_timeout_stops_the_worker_when_the_caller_is_cancelled():
    waiting = asyncio.create_task(call_off_loop_with_timeout(_spin_forever, 30.0, label="spin"))
    await asyncio.sleep(0.1)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    await asyncio.sleep(0.3)
    assert _worker_threads() == []
