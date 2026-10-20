import asyncio
import re
import threading
import time
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.custom_code.bounded_execution import (
    ExecutionTimeoutError,
    SandboxExit,
    await_with_timeout,
    call_off_loop_with_timeout,
    call_with_timeout,
)
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    CustomCodeExecutionError,
    CustomCodeGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives import (
    regex_find_all,
    regex_match,
    regex_match_all,
    regex_replace,
)


def _worker_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("guardrail-code:")]


def _join_worker_threads() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith("guardrail-code:"):
            thread.join(timeout=5)


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
    _join_worker_threads()
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
    await asyncio.to_thread(_join_worker_threads)
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
    spinning = threading.Event()

    def signal_then_spin() -> None:
        spinning.set()
        _spin_forever()

    waiting = asyncio.create_task(call_off_loop_with_timeout(signal_then_spin, 30.0, label="spin"))
    assert await asyncio.to_thread(spinning.wait, 5)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    await asyncio.to_thread(_join_worker_threads)
    assert _worker_threads() == []


@pytest.mark.asyncio
async def test_await_with_timeout_cancels_the_code_when_the_caller_is_cancelled():
    started = asyncio.Event()
    interrupted = asyncio.Event()

    async def sleep_until_cancelled() -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            interrupted.set()
            raise

    waiting = asyncio.create_task(await_with_timeout(sleep_until_cancelled(), 30.0, label="sleep"))
    assert await asyncio.wait_for(started.wait(), timeout=5)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert await asyncio.wait_for(interrupted.wait(), timeout=5)


@pytest.mark.asyncio
async def test_regex_near_match_does_not_hold_the_execution_timeout() -> None:
    guardrail: Final = CustomCodeGuardrail(
        custom_code=(
            "def apply_guardrail(inputs, request_data, input_type):\n"
            '    regex_match(inputs["texts"][0], "(a+)+$")\n'
            "    return allow()\n"
        ),
        guardrail_name="regex-near-match",
        execution_timeout=0.1,
    )
    started: Final = time.monotonic()
    try:
        result: Final = await guardrail.apply_guardrail(
            inputs={"texts": ["a" * 26 + "!"]}, request_data={"model": "m"}, input_type="request"
        )
        assert result == {"texts": ["a" * 26 + "!"]}
    except CustomCodeExecutionError as error:
        assert "exceeded its 0.1s execution timeout" in str(error)
    await asyncio.to_thread(_join_worker_threads)
    assert time.monotonic() - started < 0.8
    assert _worker_threads() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        'regex_match(inputs["texts"][0], "(a|aa)+$")',
        'regex_match_all(inputs["texts"][0], "(a|aa)+$")',
        'regex_replace(inputs["texts"][0], "(a|aa)+$", "x")',
        'regex_find_all(inputs["texts"][0], "(a|aa)+$")',
    ],
)
async def test_regex_deadline_cannot_be_swallowed_as_a_failed_match(operation: str, asynchronous: bool) -> None:
    guardrail: Final = CustomCodeGuardrail(
        custom_code=(
            f"{'async ' if asynchronous else ''}def apply_guardrail(inputs, request_data, input_type):\n"
            "    try:\n"
            f"        {operation}\n"
            "    except Exception:\n"
            "        return allow()\n"
            "    return allow()\n"
        ),
        guardrail_name="regex-deadline",
        execution_timeout=0.1,
    )
    started: Final = time.monotonic()
    with pytest.raises(CustomCodeExecutionError, match=r"exceeded its 0\.1s execution timeout"):
        await guardrail.apply_guardrail(
            inputs={"texts": ["a" * 32 + "!"]}, request_data={"model": "m"}, input_type="request"
        )
    await asyncio.to_thread(_join_worker_threads)
    assert time.monotonic() - started < 0.8
    assert _worker_threads() == []


@pytest.mark.parametrize(
    ("text", "pattern", "flags"),
    [
        ("café", r"\w+", re.ASCII),
        ("café", r"\w+", re.UNICODE),
        ("HELLO", "hello", re.IGNORECASE),
        ("first\nsecond", "^second$", re.MULTILINE),
        ("first\nsecond", ".+", re.DOTALL),
        ("hello", "h e l l o", re.VERBOSE),
        ("hello", "(h)(ello)", 0),
        ("hello", "(", 0),
    ],
)
def test_regex_primitives_preserve_standard_matching_results(text: str, pattern: str, flags: int) -> None:
    try:
        expected_match: Final = bool(re.search(pattern, text, flags))
        expected_fullmatch: Final = bool(re.fullmatch(pattern, text, flags))
        expected_replacement: Final = re.sub(pattern, "x", text, flags=flags)
        expected_matches: Final = re.findall(pattern, text, flags)
    except re.error:
        assert (regex_match(text, pattern, flags), regex_match_all(text, pattern, flags)) == (False, False)
        assert (regex_replace(text, pattern, "x", flags), regex_find_all(text, pattern, flags)) == (text, [])
        return
    assert (regex_match(text, pattern, flags), regex_match_all(text, pattern, flags)) == (
        expected_match,
        expected_fullmatch,
    )
    assert (regex_replace(text, pattern, "x", flags), regex_find_all(text, pattern, flags)) == (
        expected_replacement,
        expected_matches,
    )
