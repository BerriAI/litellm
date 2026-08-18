"""Pin ``PrismaClient`` reconnect + watchdog symbols.

Symbols pinned here:
  - ``PrismaClient._run_reconnect_cycle``
  - ``PrismaClient._attempt_reconnect_inside_lock``
  - ``PrismaClient.attempt_db_reconnect``
  - ``PrismaClient.start_db_health_watchdog_task``
  - ``PrismaClient.stop_db_health_watchdog_task``
  - ``PrismaClient._db_health_watchdog_loop``

Note on fixtures for the routing tests: the reader and the writer carry
independent generation counters, so a fixture that gives them far-apart values
reads clearly and proves nothing about identity, because comparing the numbers
alone already yields the right answer. Pick values so that ONLY the mechanism
under test can produce the expected result, which for identity means two
engines whose generations deliberately coincide.

Note on what to assert: pin the requirement, not the mechanism. An assertion
that restates what the implementation currently does can only ever agree with
it, including when it is wrong, so it ends up defending the defect from being
corrected. One here did exactly that, asserting that a declined heavy-path
recreate leaves the dead-engine flag set, which read as a faithful description
and was a reintroduction of #29176. "A later cycle must not kill a healthy
engine" would have failed against it whatever mechanism produced it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.utils import PrismaClient, _StaleReadEngine


@pytest.mark.asyncio
async def test_run_reconnect_cycle_direct_path_skips_recreate_when_probe_healthy(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct path probes the writer first: if SELECT 1 succeeds the
    connection is healthy (e.g. an IAM token refresh just replaced the
    engine) and recreating — killing the fresh engine — must be skipped.
    Part of the fix for https://github.com/BerriAI/litellm/issues/29176."""
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
        "writer_probe_called": writer.query_raw.await_count,
        "engine_confirmed_dead": prisma_client._engine_confirmed_dead,
    }
    assert pinned == {
        "recreate_called": 0,
        "start_watcher_called": 1,
        "writer_probe_called": 1,
        "engine_confirmed_dead": False,
    }


@pytest.mark.asyncio
async def test_run_reconnect_cycle_direct_path_recreates_when_probe_fails(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Genuine network blip: probe fails, so the client is recreated and the
    final SELECT 1 smoke test validates the new writer engine."""
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = False
    prisma_client._engine_pid = 0
    prisma_client.db.recreate_prisma_client = AsyncMock()
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()

    writer = MagicMock()
    writer.query_raw = AsyncMock(
        side_effect=[ConnectionError("probe failed"), [{"?column?": 1}]]
    )
    monkeypatch.setattr(
        PrismaClient,
        "writer_db",
        property(lambda self: writer),
    )

    await prisma_client._run_reconnect_cycle(timeout_seconds=5)
    pinned = {
        "recreate_called": prisma_client.db.recreate_prisma_client.await_count,
        "start_watcher_called": prisma_client._start_engine_watcher.await_count,
        "writer_query_raw_calls": writer.query_raw.await_count,
        "cleanup_called": prisma_client._cleanup_engine_watcher.call_count,
    }
    assert pinned == {
        "recreate_called": 1,
        "start_watcher_called": 1,
        "writer_query_raw_calls": 2,
        "cleanup_called": 1,
    }


@pytest.mark.asyncio
async def test_run_reconnect_cycle_force_recreate_skips_probe_and_recreates(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy writer must not veto the recreate when the caller already
    knows the session state is poisoned (stale prepared statements after a
    schema change). Regression for
    https://github.com/BerriAI/litellm/issues/36418."""
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = False
    prisma_client._engine_pid = 0
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()

    writer = prisma_client.db
    writer.recreate_prisma_client = AsyncMock()
    writer.query_raw = AsyncMock(return_value=[{"?column?": 1}])

    await prisma_client._run_reconnect_cycle(timeout_seconds=5, force_recreate=True)
    pinned = {
        "recreate_called": writer.recreate_prisma_client.await_count,
        "writer_query_raw_calls": writer.query_raw.await_count,
    }
    assert pinned == {"recreate_called": 1, "writer_query_raw_calls": 1}


@pytest.mark.asyncio
async def test_attempt_db_reconnect_forwards_force_recreate_to_cycle(
    prisma_client: PrismaClient,
) -> None:
    """Regression for https://github.com/BerriAI/litellm/issues/36418: the flag
    has to survive both hops (attempt_db_reconnect -> inside-lock -> cycle),
    otherwise the cached-plan caller silently gets a probe-gated reconnect."""
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(reason="explicit", force_recreate=True)

    assert ok is True
    assert prisma_client._run_reconnect_cycle.await_args.kwargs.get("force_recreate") is True


@pytest.mark.asyncio
async def test_run_reconnect_cycle_passes_writer_generation_to_recreate(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cycle snapshots the writer's engine generation at entry and passes
    it to recreate_prisma_client so a recreate that lost the race against a
    planned restart (IAM refresh) is skipped inside the wrapper."""
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = False
    prisma_client._engine_pid = 0
    prisma_client.db.recreate_prisma_client = AsyncMock()
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()

    writer = MagicMock()
    writer._engine_generation = 7
    writer.query_raw = AsyncMock(
        side_effect=[ConnectionError("probe failed"), [{"?column?": 1}]]
    )
    monkeypatch.setattr(
        PrismaClient,
        "writer_db",
        property(lambda self: writer),
    )

    await prisma_client._run_reconnect_cycle(timeout_seconds=5)
    recreate_kwargs = prisma_client.db.recreate_prisma_client.await_args.kwargs
    assert recreate_kwargs.get("expected_generation") == 7


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


@pytest.mark.asyncio
async def test_iam_refresh_racing_reconnect_recreates_engine_only_once(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration repro for https://github.com/BerriAI/litellm/issues/29176.

    An IAM token refresh (PrismaWrapper._safe_refresh_token) is mid-recreate
    when an in-flight transport error triggers attempt_db_reconnect. The
    reconnect must NOT recreate the Prisma client a second time (which would
    SIGTERM the engine the refresh just spawned).
    """
    import os
    import urllib.parse
    from datetime import datetime, timedelta

    import prisma as prisma_pkg

    from litellm.proxy.db.prisma_client import PrismaWrapper

    def token_db_url(created: datetime) -> str:
        token = (
            f"host/?X-Amz-Date={created.strftime('%Y%m%dT%H%M%SZ')}"
            f"&X-Amz-Expires=900&X-Amz-Signature=abc"
        )
        return f"postgresql://user:{urllib.parse.quote(token, safe='')}@host:5432/db"

    # Old engine (PID 111) carries an expired token; in-flight queries on it
    # fail with a transport error.
    expired_url = token_db_url(datetime.utcnow() - timedelta(seconds=1200))
    fresh_url = token_db_url(datetime.utcnow())
    monkeypatch.setenv("DATABASE_URL", expired_url)

    old_prisma = MagicMock(name="OldPrisma")
    old_prisma._engine = MagicMock()
    old_prisma._engine.process.pid = 111
    old_prisma.query_raw = AsyncMock(side_effect=ConnectionError("engine restarting"))

    wrapper = PrismaWrapper(original_prisma=old_prisma, iam_token_db_auth=True)
    prisma_client.db = wrapper
    prisma_client._engine_pid = 0
    prisma_client._engine_confirmed_dead = False
    prisma_client._start_engine_watcher = AsyncMock()

    # The refresh's recreate is held open at connect() so the reconnect path
    # races it deterministically.
    connect_started = asyncio.Event()
    release_connect = asyncio.Event()

    async def slow_connect(*args: Any, **kwargs: Any) -> None:
        connect_started.set()
        await release_connect.wait()

    new_prisma = MagicMock(name="NewPrisma")
    new_prisma.connect = AsyncMock(side_effect=slow_connect)
    new_prisma._engine = MagicMock()
    new_prisma._engine.process.pid = 222
    new_prisma.query_raw = AsyncMock(return_value=[{"?column?": 1}])

    prisma_factory = MagicMock(name="PrismaFactory", return_value=new_prisma)
    monkeypatch.setattr(prisma_pkg, "Prisma", prisma_factory, raising=False)

    def fake_get_token() -> str:
        os.environ["DATABASE_URL"] = fresh_url
        return fresh_url

    monkeypatch.setattr(wrapper, "get_rds_iam_token", fake_get_token)
    kill_mock = MagicMock()
    monkeypatch.setattr("os.kill", kill_mock)

    refresh_task = asyncio.create_task(wrapper._safe_refresh_token())
    await asyncio.wait_for(connect_started.wait(), timeout=5)

    # In-flight transport-error path fires while the refresh holds the
    # wrapper's reconnection lock mid-recreate.
    reconnect_task = asyncio.create_task(
        prisma_client.attempt_db_reconnect(
            reason="in_flight_transport_error", force=True
        )
    )
    await asyncio.sleep(0.05)
    release_connect.set()

    await asyncio.wait_for(refresh_task, timeout=5)
    reconnect_ok = await asyncio.wait_for(reconnect_task, timeout=5)

    # Drain any refresh task scheduled by PrismaWrapper.__getattr__ during
    # the probe (expired-token path) so it coalesces before we assert.
    for _ in range(3):
        await asyncio.sleep(0)

    killed_pids = [c.args[0] for c in kill_mock.call_args_list]
    pinned = {
        "prisma_constructed": prisma_factory.call_count,
        "fresh_engine_killed": 222 in killed_pids,
        "reconnect_ok": reconnect_ok,
        "wrapper_client_is_new": wrapper._original_prisma is new_prisma,
    }
    assert pinned == {
        "prisma_constructed": 1,
        "fresh_engine_killed": False,
        "reconnect_ok": True,
        "wrapper_client_is_new": True,
    }


@pytest.mark.asyncio
async def test_run_reconnect_cycle_heavy_path_forwards_entry_generation_to_recreate(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heavy (engine-dead) path must also forward an engine-generation
    snapshot to recreate_prisma_client, captured atomically at cycle entry.

    A concurrent IAM refresh that replaces the engine mid-cycle bumps the
    generation, so the guarded recreate becomes a no-op instead of killing the
    freshly-spawned engine (#29176). The snapshot must be taken before any
    await — `asyncio.wait_for(_do_heavy_reconnect())` yields, during which a
    refresh can slip in. A side effect that bumps the generation AFTER entry
    must NOT change the forwarded value (proves entry-snapshot, not in-closure).
    """
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = True
    prisma_client._engine_pid = 1234
    prisma_client.db.recreate_prisma_client = AsyncMock()
    prisma_client._start_engine_watcher = AsyncMock()
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))

    writer = MagicMock()
    writer._engine_generation = 4
    monkeypatch.setattr(PrismaClient, "writer_db", property(lambda self: writer))

    # Simulate a concurrent refresh bumping the generation after cycle entry:
    # _cleanup_engine_watcher runs between the entry snapshot and the recreate.
    def _bump_then_cleanup() -> None:
        writer._engine_generation = 5

    monkeypatch.setattr(prisma_client, "_cleanup_engine_watcher", _bump_then_cleanup)

    await prisma_client._run_reconnect_cycle(timeout_seconds=5)

    kwargs = prisma_client.db.recreate_prisma_client.await_args.kwargs
    assert kwargs.get("expected_generation") == 4


@pytest.mark.asyncio
async def test_attempt_db_reconnect_bypasses_cooldown_for_still_live_stale_engine(
    prisma_client: PrismaClient,
) -> None:
    """A schema change landing inside the cooldown of an earlier reconnect used
    to leave auth failing until the cooldown elapsed. While the engine the
    caller's failure came from is still the live one, the cooldown must not
    gate the recreate. Regression for
    https://github.com/BerriAI/litellm/issues/36418."""
    prisma_client.db.engine_generation = 7
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=_StaleReadEngine(wrapper=prisma_client.read_db, generation=7),
    )

    assert ok is True
    assert prisma_client._run_reconnect_cycle.await_count == 1


@pytest.mark.asyncio
async def test_attempt_db_reconnect_honors_cooldown_once_stale_engine_replaced(
    prisma_client: PrismaClient,
) -> None:
    """The bypass is scoped to the damaged engine: once a concurrent recreate
    has replaced it, the cooldown must still collapse the rest of the burst
    onto that recreate instead of killing the fresh engine."""
    prisma_client.db.engine_generation = 8
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=_StaleReadEngine(wrapper=prisma_client.read_db, generation=7),
    )

    assert ok is False
    prisma_client._run_reconnect_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_keeps_cooldown_for_callers_without_generation(
    prisma_client: PrismaClient,
) -> None:
    """Watchdog and transport-error callers name no generation, so they keep
    the plain cooldown behaviour."""
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(reason="watchdog_probe_failed")

    assert ok is False
    prisma_client._run_reconnect_cycle.assert_not_awaited()


def _routing_client(prisma_client: PrismaClient, reader_generation: int, writer_generation: int) -> tuple[Any, Any]:
    """Wire ``prisma_client.db`` to a routing wrapper with distinct engines.

    Returns the (writer, reader) mocks so a test can move either generation
    independently, which is the only way to tell the two counters apart.
    """
    writer = MagicMock(name="writer")
    writer.engine_generation = writer_generation
    reader = MagicMock(name="reader")
    reader.engine_generation = reader_generation
    prisma_client.db = RoutingPrismaWrapper(writer=writer, reader=reader)
    return writer, reader


@pytest.mark.asyncio
async def test_attempt_db_reconnect_reads_generation_from_the_reader_that_served_the_query(
    prisma_client: PrismaClient,
) -> None:
    """``query_first`` is a top-level read, so with a replica configured the
    stale prepared statements are on the READER. A writer reconnect that moved
    the writer generation must not re-arm the cooldown while the reader the
    query actually failed on is still the live, poisoned one. Regression for
    https://github.com/BerriAI/litellm/issues/36418."""
    _routing_client(prisma_client, reader_generation=7, writer_generation=99)
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=_StaleReadEngine(wrapper=prisma_client.read_db, generation=7),
    )

    assert ok is True
    assert prisma_client._run_reconnect_cycle.await_count == 1


@pytest.mark.asyncio
async def test_attempt_db_reconnect_honors_cooldown_once_the_reader_itself_was_replaced(
    prisma_client: PrismaClient,
) -> None:
    """The mirror of the above: once the reader has been replaced, the recreate
    the caller needed has already happened, so the cooldown collapses the rest
    of the burst even though the writer generation never moved."""
    _routing_client(prisma_client, reader_generation=8, writer_generation=99)
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=_StaleReadEngine(wrapper=prisma_client.read_db, generation=7),
    )

    assert ok is False
    prisma_client._run_reconnect_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_gates_when_reads_moved_to_an_engine_of_the_same_generation(
    prisma_client: PrismaClient,
) -> None:
    """The counters are per engine, so the reader and the writer can sit on the
    same number at the same time. Once the reader goes unavailable reads move to
    the writer, and the caller's poisoned reader is no longer serving anything,
    so the cooldown should gate it. Comparing generations alone cannot tell the
    two apart and would hand out the waiver here: the generations are equal on
    purpose, which is what makes this the case identity has to decide."""
    writer, reader = _routing_client(prisma_client, reader_generation=5, writer_generation=5)
    stale: Final = _StaleReadEngine(wrapper=reader, generation=5)
    prisma_client.db._reader_unavailable = True
    prisma_client._db_last_reconnect_attempt_ts = time.time()
    prisma_client._run_reconnect_cycle = AsyncMock()

    ok = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=stale,
    )

    pinned = {
        "reads_now_served_by_the_writer": prisma_client.read_db is writer,
        "generations_coincide": reader.engine_generation == writer.engine_generation,
        # `_cooldown_applies` gates on the failed-repair record OR on liveness,
        # and either alone produces this result. Pin that the record is empty,
        # or a stray entry would make this pass while testing the other gate.
        "no_failed_repair_recorded": dict(prisma_client._failed_recreate_generations) == {},
        "recovered": ok,
        "cycles_run": prisma_client._run_reconnect_cycle.await_count,
    }
    assert pinned == {
        "reads_now_served_by_the_writer": True,
        "generations_coincide": True,
        "no_failed_repair_recorded": True,
        "recovered": False,
        "cycles_run": 0,
    }


@pytest.mark.asyncio
async def test_failed_repair_of_one_engine_is_not_evicted_by_a_failure_on_the_other(
    prisma_client: PrismaClient,
) -> None:
    """The record is kept per engine. Held in a single slot, a failed writer
    repair would evict the reader's record, and the next caller naming the
    reader's still-unrepaired generation would get the waiver back and run its
    own redundant cycle, which is the burst the record exists to collapse."""
    writer, reader = _routing_client(prisma_client, reader_generation=5, writer_generation=3)
    stale_reader: Final = _StaleReadEngine(wrapper=reader, generation=5)
    stale_writer: Final = _StaleReadEngine(wrapper=writer, generation=3)
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._run_reconnect_cycle = AsyncMock(side_effect=RuntimeError("engine spawn failed"))

    await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error", force_recreate=True, stale_read_engine=stale_reader
    )
    prisma_client.db._reader_unavailable = True
    await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error", force_recreate=True, stale_read_engine=stale_writer
    )
    prisma_client.db._reader_unavailable = False
    cycles_before_the_reader_returns: Final = prisma_client._run_reconnect_cycle.await_count

    await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error", force_recreate=True, stale_read_engine=stale_reader
    )

    pinned = {
        "cycles_before": cycles_before_the_reader_returns,
        "cycles_after": prisma_client._run_reconnect_cycle.await_count,
    }
    assert pinned == {"cycles_before": 2, "cycles_after": 2}


@pytest.mark.asyncio
async def test_attempt_db_reconnect_withdraws_the_waiver_after_this_generation_failed_to_repair(
    prisma_client: PrismaClient,
) -> None:
    """A failed recreate leaves the generation where it was, so without a record
    of the failure every queued caller of the same burst would still see its own
    generation live and run its own full recreate serially instead of collapsing
    onto one attempt. Drives two callers rather than presetting the record, so
    the record has to actually be written by the failure."""
    prisma_client.db.engine_generation = 7
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._run_reconnect_cycle = AsyncMock(side_effect=RuntimeError("engine spawn failed"))

    first = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=_StaleReadEngine(wrapper=prisma_client.read_db, generation=7),
    )
    second = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=_StaleReadEngine(wrapper=prisma_client.read_db, generation=7),
    )

    pinned = {
        "first": first,
        "second": second,
        "cycles_run": prisma_client._run_reconnect_cycle.await_count,
    }
    assert pinned == {"first": False, "second": False, "cycles_run": 1}


@pytest.mark.asyncio
async def test_attempt_db_reconnect_keeps_the_waiver_after_an_unrelated_reconnect_failure(
    prisma_client: PrismaClient,
) -> None:
    """The failure record is scoped to the generation it was trying to repair.
    A watchdog or transport-error reconnect names no generation, so its failure
    says nothing about whether a stale read engine can be repaired and must not
    gate it: gating on a global failure count would 503 authentication for the
    length of the cooldown."""
    prisma_client.db.engine_generation = 7
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._run_reconnect_cycle = AsyncMock(side_effect=RuntimeError("watchdog reconnect failed"))

    unrelated = await prisma_client.attempt_db_reconnect(reason="watchdog_probe_failed")
    # Read before the second call: a global failure gate would be armed here,
    # and the recovering reconnect below resets the counter either way.
    failures_left_by_the_unrelated_reconnect: Final = prisma_client._consecutive_reconnect_failures

    prisma_client._run_reconnect_cycle = AsyncMock()
    cached_plan = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=_StaleReadEngine(wrapper=prisma_client.read_db, generation=7),
    )

    pinned = {
        "unrelated_failed": unrelated,
        "failures_left_by_the_unrelated_reconnect": failures_left_by_the_unrelated_reconnect,
        "cached_plan_recovered": cached_plan,
        "cycles_run_for_cached_plan": prisma_client._run_reconnect_cycle.await_count,
    }
    assert pinned == {
        "unrelated_failed": False,
        "failures_left_by_the_unrelated_reconnect": 1,
        "cached_plan_recovered": True,
        "cycles_run_for_cached_plan": 1,
    }


@pytest.mark.asyncio
async def test_forced_recreate_declined_by_the_generation_guard_is_not_reported_as_success(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``recreate_prisma_client`` declines when the writer generation moved
    since cycle entry, and the routing wrapper then leaves the reader untouched
    too. A forced caller asked for its engine to be replaced and it was not, so
    reporting success would reset the consecutive-failure count and log a repair
    that never happened. The declined attempt must equally not count as a
    failure, or the caller's own backoff would be gated on its next try."""
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = False
    prisma_client._engine_pid = 0
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._consecutive_reconnect_failures = 0

    writer = prisma_client.db
    writer.recreate_prisma_client = AsyncMock(return_value=False)
    writer.query_raw = AsyncMock(return_value=[{"?column?": 1}])

    ok = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
    )

    pinned = {
        "reported_success": ok,
        "recreate_attempted": writer.recreate_prisma_client.await_count,
        "consecutive_failures": prisma_client._consecutive_reconnect_failures,
    }
    assert pinned == {
        "reported_success": False,
        "recreate_attempted": 1,
        "consecutive_failures": 0,
    }


@pytest.mark.asyncio
async def test_unforced_recreate_declined_by_the_generation_guard_still_succeeds(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decline is only an error for a caller that forced the recreate. A
    transport-blip caller is happy to learn another path already replaced the
    engine, so its reconnect still reports success."""
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = False
    prisma_client._engine_pid = 0
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()
    prisma_client._db_last_reconnect_attempt_ts = 0.0

    writer = prisma_client.db
    writer.recreate_prisma_client = AsyncMock(return_value=False)
    # First call is the liveness probe, which must fail so the recreate is
    # reached at all; the second is the post-recreate smoke test.
    writer.query_raw = AsyncMock(side_effect=[Exception("probe fails"), [{"?column?": 1}]])

    ok = await prisma_client.attempt_db_reconnect(reason="transport_blip")

    pinned = {"reported_success": ok, "recreate_attempted": writer.recreate_prisma_client.await_count}
    assert pinned == {"reported_success": True, "recreate_attempted": 1}


@pytest.mark.asyncio
async def test_heavy_path_forced_recreate_declined_is_not_reported_as_success(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced caller reaches the heavy branch too: the escalation threshold
    flips ``_engine_confirmed_dead`` after repeated failures, and every cycle
    after that takes the dead-engine path. A decline there has to be treated
    exactly as it is on the direct path, or the escalation itself reintroduces
    the success that never happened."""
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_confirmed_dead = True
    prisma_client._engine_pid = 1234
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    prisma_client._consecutive_reconnect_failures = 0
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))

    prisma_client.db.recreate_prisma_client = AsyncMock(return_value=False)

    ok = await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
    )

    pinned = {
        "reported_success": ok,
        "recreate_attempted": prisma_client.db.recreate_prisma_client.await_count,
        "consecutive_failures": prisma_client._consecutive_reconnect_failures,
        # The dead-engine flag must be CLEARED. A raise normally skips the
        # clear, which is right for a failure and wrong here: the guard
        # declined because another path had already replaced the engine, so it
        # is alive. Leaving it set routes the next cycle back down this
        # probe-free branch, where the recreate would kill that healthy engine.
        "engine_still_confirmed_dead": prisma_client._engine_confirmed_dead,
    }
    assert pinned == {
        "reported_success": False,
        "recreate_attempted": 1,
        "consecutive_failures": 0,
        "engine_still_confirmed_dead": False,
    }


@pytest.mark.asyncio
async def test_declined_heavy_recreate_disarms_escalation_for_the_next_attempt(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing the dead-engine flag on a decline is not enough on its own. The
    escalation check re-arms that flag whenever the consecutive-failure count is
    still at the threshold, so a decline that left the count alone would send
    the very next attempt back down the probe-free heavy path and recreate over
    the healthy engine another path had just installed. Drives the SECOND
    attempt, because the first one alone cannot show this."""
    monkeypatch.setenv("DATABASE_URL", "postgres://x:y@h:5432/db")
    prisma_client._engine_pid = 1234
    prisma_client._start_engine_watcher = AsyncMock()
    prisma_client._cleanup_engine_watcher = MagicMock()
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))
    # Escalation already armed by earlier genuine failures.
    prisma_client._consecutive_reconnect_failures = prisma_client._reconnect_escalation_threshold
    prisma_client.db.recreate_prisma_client = AsyncMock(return_value=False)
    prisma_client.db.query_raw = AsyncMock(return_value=[{"?column?": 1}])

    armed: Final = prisma_client._engine_confirmed_dead is False and prisma_client._consecutive_reconnect_failures > 0

    await prisma_client.attempt_db_reconnect(reason="postgres_cached_plan_error", force_recreate=True)

    # Kept as its own assert, not folded into the judgement below. These are two
    # claims about two moments, the first being a precondition for the second
    # meaning anything, and a single combined comparison would hide which one
    # failed from both the traceback and a mutation report.
    assert {
        "escalation_was_armed_by_the_count": armed,
        "failures": prisma_client._consecutive_reconnect_failures,
        "engine_confirmed_dead": prisma_client._engine_confirmed_dead,
    } == {"escalation_was_armed_by_the_count": True, "failures": 0, "engine_confirmed_dead": False}

    prisma_client._db_last_reconnect_attempt_ts = 0.0
    await prisma_client.attempt_db_reconnect(reason="postgres_cached_plan_error", force_recreate=True)

    # The requirement: a later cycle must not reclassify the healthy replacement
    # as dead and restart it through the probe-free path.
    assert prisma_client._engine_confirmed_dead is False


@pytest.mark.asyncio
async def test_unrelated_reconnect_failure_does_not_erase_the_burst_record(
    prisma_client: PrismaClient,
) -> None:
    """The failure record names one engine, so a caller that names none must
    not overwrite it. Otherwise a watchdog failure landing between two callers
    of the same burst clears the record and the second caller runs its own full
    recreate against the engine the first one just failed to repair."""
    prisma_client.db.engine_generation = 7
    prisma_client._db_last_reconnect_attempt_ts = 0.0
    stale: Final = _StaleReadEngine(wrapper=prisma_client.read_db, generation=7)
    prisma_client._run_reconnect_cycle = AsyncMock(side_effect=RuntimeError("engine spawn failed"))

    await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=stale,
    )
    # force=True the way the engine-death callers do, so this one actually
    # reaches the failure branch instead of being skipped by the cooldown the
    # first caller just stamped.
    await prisma_client.attempt_db_reconnect(reason="engine_process_death", force=True)
    cycles_before_the_second_burst_caller: Final = prisma_client._run_reconnect_cycle.await_count

    await prisma_client.attempt_db_reconnect(
        reason="postgres_cached_plan_error",
        force_recreate=True,
        stale_read_engine=stale,
    )

    pinned = {
        "cycles_before": cycles_before_the_second_burst_caller,
        "cycles_after": prisma_client._run_reconnect_cycle.await_count,
    }
    assert pinned == {"cycles_before": 2, "cycles_after": 2}
