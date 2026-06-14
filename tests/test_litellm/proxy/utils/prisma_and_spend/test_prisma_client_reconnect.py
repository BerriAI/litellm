"""Pin ``PrismaClient`` reconnect + watchdog symbols.

Symbols pinned here:
  - ``PrismaClient._run_reconnect_cycle``
  - ``PrismaClient._attempt_reconnect_inside_lock``
  - ``PrismaClient.attempt_db_reconnect``
  - ``PrismaClient.start_db_health_watchdog_task``
  - ``PrismaClient.stop_db_health_watchdog_task``
  - ``PrismaClient._db_health_watchdog_loop``
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
async def test_run_reconnect_cycle_direct_path_when_engine_alive(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = False
    prisma_client._engine_pid = 0
    prisma_client.db.recreate_prisma_client = AsyncMock()
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()

    writer = MagicMock()
    writer.query_raw = AsyncMock(return_value=[{"?column?": 1}])
    monkeypatch.setattr(
        PrismaClient,
        "writer_db",
        property(lambda self: writer),
    )

    await prisma_client._run_reconnect_cycle(timeout_seconds=5)
    pinned = {
        "recreate_called": prisma_client.db.recreate_prisma_client.await_count,
        "start_watcher_called": prisma_client._start_engine_watcher.await_count,
        "writer_smoke_test_called": writer.query_raw.await_count,
        "engine_confirmed_dead": prisma_client._engine_confirmed_dead,
    }
    assert pinned == {
        "recreate_called": 1,
        "start_watcher_called": 1,
        "writer_smoke_test_called": 1,
        "engine_confirmed_dead": False,
    }


@pytest.mark.asyncio
async def test_run_reconnect_cycle_heavy_path_when_engine_dead(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = True
    prisma_client._engine_pid = 1234
    prisma_client.db.recreate_prisma_client = AsyncMock()
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))

    await prisma_client._run_reconnect_cycle(timeout_seconds=5)
    pinned = {
        "recreate_called": prisma_client.db.recreate_prisma_client.await_count,
        "start_watcher_called": prisma_client._start_engine_watcher.await_count,
        "cleanup_called": prisma_client._cleanup_engine_watcher.call_count,
        "dead_flag_cleared": prisma_client._engine_confirmed_dead,
    }
    assert pinned == {
        "recreate_called": 1,
        "start_watcher_called": 1,
        "cleanup_called": 1,
        "dead_flag_cleared": False,
    }


@pytest.mark.asyncio
async def test_run_reconnect_cycle_raises_when_database_url_missing(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL not set"):
        await prisma_client._run_reconnect_cycle(timeout_seconds=1)


@pytest.mark.asyncio
async def test_attempt_reconnect_inside_lock_runs_cycle_and_resets_counter(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._consecutive_reconnect_failures = 2
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client._attempt_reconnect_inside_lock(
        force=True, reason="test", timeout_seconds=1
    )
    pinned = {
        "returned": ok,
        "cycle_called": prisma_client._run_reconnect_cycle.await_count,
        "failures_reset": prisma_client._consecutive_reconnect_failures,
    }
    assert pinned == {
        "returned": True,
        "cycle_called": 1,
        "failures_reset": 0,
    }


@pytest.mark.asyncio
async def test_attempt_reconnect_inside_lock_skips_when_in_cooldown(
    prisma_client: PrismaClient,
) -> None:
    import time

    prisma_client._db_reconnect_cooldown_seconds = 60
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client._attempt_reconnect_inside_lock(
        force=False, reason="test", timeout_seconds=1
    )
    assert ok is False
    assert prisma_client._run_reconnect_cycle.await_count == 0


@pytest.mark.asyncio
async def test_attempt_reconnect_inside_lock_increments_failure_counter_on_error(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._consecutive_reconnect_failures = 0
    prisma_client._run_reconnect_cycle = AsyncMock(side_effect=RuntimeError("boom"))

    ok = await prisma_client._attempt_reconnect_inside_lock(
        force=True, reason="failing_test", timeout_seconds=1
    )
    assert ok is False
    assert prisma_client._consecutive_reconnect_failures == 1


@pytest.mark.asyncio
async def test_attempt_db_reconnect_force_runs_under_lock(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._attempt_reconnect_inside_lock = AsyncMock(return_value=True)

    result = await prisma_client.attempt_db_reconnect(reason="explicit", force=True)
    args = prisma_client._attempt_reconnect_inside_lock.await_args
    pinned = {
        "returned": result,
        "calls": prisma_client._attempt_reconnect_inside_lock.await_count,
        "passed_force": args.args[0],
        "passed_reason": args.args[1],
        "passed_timeout": args.args[2],
    }
    assert pinned == {
        "returned": True,
        "calls": 1,
        "passed_force": True,
        "passed_reason": "explicit",
        "passed_timeout": None,
    }


@pytest.mark.asyncio
async def test_attempt_db_reconnect_lock_timeout_returns_false(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reconnect attempt that can't acquire the lock within
    ``lock_timeout_seconds`` returns False without running the cycle.

    The production code creates an inner task, races it against the
    timeout via ``asyncio.wait``, then cancels and awaits the loser.
    Under coverage instrumentation on Python 3.11 the CancelledError from
    a freshly-cancelled task can outrun the surrounding ``except`` block,
    so this test pre-completes the inner task (no cancellation happens)
    by replacing ``asyncio.wait`` with a callable that returns the loser
    task as still-pending after it's already been completed elsewhere.
    """
    completed_task: asyncio.Task[bool] = asyncio.get_running_loop().create_task(
        _no_op_returning_true()
    )
    # Ensure the inner task has finished before attempt_db_reconnect sees it.
    await completed_task

    async def _wait_returns_loser(_tasks: Any, **kwargs: Any) -> Any:
        return set(), {completed_task}

    monkeypatch.setattr("asyncio.wait", _wait_returns_loser)
    monkeypatch.setattr(
        asyncio,
        "create_task",
        lambda coro, *a, **kw: (coro.close() or completed_task),
    )

    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._attempt_reconnect_inside_lock = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(
        reason="lock_busy",
        lock_timeout_seconds=0.0,
    )
    assert ok is False
    assert prisma_client._attempt_reconnect_inside_lock.await_count == 0


async def _no_op_returning_true() -> bool:
    return True


@pytest.mark.asyncio
async def test_attempt_db_reconnect_skips_in_cooldown_returns_false(
    prisma_client: PrismaClient,
) -> None:
    import time

    prisma_client._db_reconnect_cooldown_seconds = 60
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    ok = await prisma_client.attempt_db_reconnect(reason="cooled_down")
    assert ok is False


@pytest.mark.asyncio
async def test_start_db_health_watchdog_task_creates_loop_task(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._db_health_watchdog_enabled = True
    prisma_client._db_health_watchdog_task = None
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._db_health_watchdog_loop = AsyncMock(return_value=None)

    await prisma_client.start_db_health_watchdog_task()
    task = prisma_client._db_health_watchdog_task
    # Yield control so the just-scheduled task actually invokes the loop mock.
    await asyncio.sleep(0)
    pinned = {
        "task_type": type(task).__name__,
        "watcher_started": prisma_client._start_engine_watcher.await_count,
        "loop_invoked": prisma_client._db_health_watchdog_loop.await_count,
    }
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert pinned == {
        "task_type": "Task",
        "watcher_started": 1,
        "loop_invoked": 1,
    }


@pytest.mark.asyncio
async def test_start_db_health_watchdog_task_disabled_short_circuits(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._db_health_watchdog_enabled = False
    prisma_client._start_engine_watcher = AsyncMock()
    await prisma_client.start_db_health_watchdog_task()
    assert prisma_client._db_health_watchdog_task is None
    assert prisma_client._start_engine_watcher.await_count == 0


@pytest.mark.asyncio
async def test_stop_db_health_watchdog_task_cancels_and_clears(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._stop_engine_watcher = MagicMock()

    cancel_called = {"n": 0}

    class _FakeTask:
        def cancel(self) -> None:
            cancel_called["n"] += 1

        def __await__(self):
            return iter([])

    prisma_client._db_health_watchdog_task = _FakeTask()  # type: ignore[assignment]

    await prisma_client.stop_db_health_watchdog_task()
    pinned = {
        "task_cleared": prisma_client._db_health_watchdog_task,
        "engine_stop_called": prisma_client._stop_engine_watcher.call_count,
        "cancel_called": cancel_called["n"],
        "no_failure": True,
    }
    assert pinned == {
        "task_cleared": None,
        "engine_stop_called": 1,
        "cancel_called": 1,
        "no_failure": True,
    }


@pytest.mark.asyncio
async def test_stop_db_health_watchdog_task_noop_when_no_task(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._db_health_watchdog_task = None
    prisma_client._stop_engine_watcher = MagicMock(side_effect=RuntimeError("err"))
    with pytest.raises(RuntimeError, match="err"):
        await prisma_client.stop_db_health_watchdog_task()


@pytest.mark.asyncio
async def test_db_health_watchdog_loop_triggers_reconnect_on_timeout(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The watchdog loop reconnects when ``wait_for`` raises TimeoutError
    or a recognized DB connection error.
    """
    prisma_client._db_health_watchdog_interval_seconds = 0
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)

    call_count = {"n": 0}

    async def _timeout_then_cancel(*args: Any, **kwargs: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise asyncio.CancelledError()
        raise asyncio.TimeoutError()

    monkeypatch.setattr("asyncio.wait_for", _timeout_then_cancel)
    await prisma_client._db_health_watchdog_loop()
    pinned = {
        "reconnect_called": prisma_client.attempt_db_reconnect.await_count,
        "reconnect_reason": prisma_client.attempt_db_reconnect.await_args.kwargs[
            "reason"
        ],
        "wait_for_calls": call_count["n"],
        "loop_exited_clean": True,
    }
    assert pinned == {
        "reconnect_called": 1,
        "reconnect_reason": "db_health_watchdog_connection_error",
        "wait_for_calls": 2,
        "loop_exited_clean": True,
    }


@pytest.mark.asyncio
async def test_db_health_watchdog_loop_swallows_non_db_errors(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-DB error during the probe should NOT trigger reconnect; the
    loop logs and continues until cancellation.
    """
    prisma_client._db_health_watchdog_interval_seconds = 0
    prisma_client.attempt_db_reconnect = AsyncMock()

    call_count = {"n": 0}

    async def _raise_then_cancel(*args: Any, **kwargs: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise asyncio.CancelledError()
        raise ValueError("not a db error")

    monkeypatch.setattr("asyncio.wait_for", _raise_then_cancel)
    await prisma_client._db_health_watchdog_loop()
    assert prisma_client.attempt_db_reconnect.await_count == 0
