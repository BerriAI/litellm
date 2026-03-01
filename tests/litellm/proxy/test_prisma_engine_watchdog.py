"""
Tests for PrismaClient engine watchdog: death detection and automatic reconnect.

Covers:
- Engine PID discovery and liveness check
- Engine process gone (os.kill raises ProcessLookupError) → reconnect triggered
- PermissionError from os.kill → treated as alive (process exists but not ours)
- pidfd handler → schedules attempt_db_reconnect even when lock is held
- waitpid thread → instant cross-platform detection, triggers reconnect
- _run_reconnect_cycle branches: heavy path (engine dead) vs lightweight path (engine alive)
- _engine_confirmed_dead flag ensures heavy reconnect even after _engine_pid reset
- Successful heavy reconnect → watcher re-armed for new process
- Missing DATABASE_URL → graceful RuntimeError in reconnect cycle
- Shutdown → polling loop exits cleanly
"""

import asyncio
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    import sys

    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield


@pytest.fixture
def mock_proxy_logging():
    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    return proxy_logging


@pytest.fixture
def engine_client(mock_proxy_logging) -> PrismaClient:
    """
    Minimal PrismaClient fixture for engine watchdog tests.
    Uses the real constructor pattern from PR #21706 (database_url).
    """
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db = MagicMock()
    client.db.recreate_prisma_client = AsyncMock()
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])
    return client


# ---------------------------------------------------------------------------
# _is_engine_alive
# ---------------------------------------------------------------------------


def test_is_engine_alive_returns_true_when_pid_unknown(engine_client):
    """_is_engine_alive returns True when no engine PID is tracked."""
    engine_client._engine_pid = 0
    assert engine_client._is_engine_alive() is True


def test_is_engine_alive_returns_false_when_process_gone(engine_client):
    """_is_engine_alive returns False when os.kill raises ProcessLookupError."""
    engine_client._engine_pid = 9999
    with patch("os.kill", side_effect=ProcessLookupError):
        assert engine_client._is_engine_alive() is False


def test_is_engine_alive_returns_true_on_permission_error(engine_client):
    """_is_engine_alive returns True when os.kill raises PermissionError (process exists but not ours)."""
    engine_client._engine_pid = 1234
    with patch("os.kill", side_effect=PermissionError):
        assert engine_client._is_engine_alive() is True


def test_is_engine_alive_returns_true_for_running_process(engine_client):
    """_is_engine_alive returns True when os.kill succeeds (process running)."""
    engine_client._engine_pid = 1234
    with patch("os.kill"):
        assert engine_client._is_engine_alive() is True


# ---------------------------------------------------------------------------
# _poll_engine_proc — calls attempt_db_reconnect on death
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_missing_process_triggers_reconnect(engine_client) -> None:
    """Polling loop triggers attempt_db_reconnect when os.kill raises ProcessLookupError."""
    engine_client._engine_pid = 1234
    engine_client._watching_engine = True
    engine_client.attempt_db_reconnect = AsyncMock(return_value=True)

    with patch("os.kill", side_effect=ProcessLookupError):
        await engine_client._poll_engine_proc()

    engine_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="engine_process_death",
        force=True,
    )


@pytest.mark.asyncio
async def test_poll_permission_error_stops_polling(engine_client) -> None:
    """Polling loop stops cleanly when os.kill raises PermissionError (process not ours)."""
    engine_client._engine_pid = 1234
    engine_client._watching_engine = True
    engine_client.attempt_db_reconnect = AsyncMock(return_value=True)

    with patch("os.kill", side_effect=PermissionError):
        await engine_client._poll_engine_proc()

    # PermissionError means process exists but isn't ours — no reconnect, just stop polling
    engine_client.attempt_db_reconnect.assert_not_awaited()
    assert engine_client._watching_engine is False
    assert engine_client._engine_pid == 0


@pytest.mark.asyncio
async def test_stop_loop_halts_polling(engine_client) -> None:
    """Polling loop exits cleanly when _stop_engine_watcher is called."""
    engine_client._engine_pid = 1234
    engine_client._watching_engine = True

    async def stop_during_sleep(_duration: float) -> None:
        engine_client._stop_engine_watcher()

    with (
        patch("os.kill"),
        patch("asyncio.sleep", side_effect=stop_during_sleep),
    ):
        await engine_client._poll_engine_proc()

    assert engine_client._watching_engine is False
    assert engine_client._engine_pid == 0


# ---------------------------------------------------------------------------
# _on_pidfd_readable — calls attempt_db_reconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pidfd_readable_schedules_reconnect(engine_client) -> None:
    """pidfd handler schedules attempt_db_reconnect via asyncio.create_task."""
    engine_client._engine_pid = 1234
    engine_client.attempt_db_reconnect = AsyncMock(return_value=True)

    created_coros = []

    def capture_task(coro):
        created_coros.append(coro)
        return MagicMock()

    with patch("asyncio.create_task", side_effect=capture_task):
        engine_client._on_pidfd_readable()

    # Run the captured coroutine to completion
    assert len(created_coros) == 1
    await created_coros[0]

    engine_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="engine_process_death",
        force=True,
    )


@pytest.mark.asyncio
async def test_pidfd_schedules_reconnect_task_when_lock_held(engine_client) -> None:
    """pidfd handler schedules reconnect task even when _db_reconnect_lock is held."""
    engine_client._engine_pid = 1234

    created_coros = []

    def capture_task(coro):
        created_coros.append(coro)
        return MagicMock()

    async with engine_client._db_reconnect_lock:
        with patch("asyncio.create_task", side_effect=capture_task):
            engine_client._on_pidfd_readable()

    for coro in created_coros:
        coro.close()

    assert len(created_coros) == 1


# ---------------------------------------------------------------------------
# _run_reconnect_cycle — engine liveness branching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_reconnect_cycle_uses_heavy_path_when_engine_dead(
    engine_client,
) -> None:
    """_run_reconnect_cycle calls recreate_prisma_client when engine is dead."""
    engine_client._engine_pid = 1234
    engine_client._start_engine_watcher = AsyncMock()

    with (
        patch.object(engine_client, "_is_engine_alive", return_value=False),
        patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
        patch("os.waitpid", side_effect=ChildProcessError),
    ):
        await engine_client._run_reconnect_cycle(timeout_seconds=5.0)

    engine_client.db.recreate_prisma_client.assert_awaited_once_with("postgresql://test")
    engine_client._start_engine_watcher.assert_awaited_once()
    engine_client.db.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_reconnect_cycle_uses_heavy_path_when_confirmed_dead(
    engine_client,
) -> None:
    """_run_reconnect_cycle takes heavy path when _engine_confirmed_dead is set.

    This is the critical race-condition fix: SIGCHLD/pidfd handlers set
    _engine_confirmed_dead BEFORE _cleanup_engine_watcher resets _engine_pid
    to 0, so the heavy path executes even after cleanup.
    """
    engine_client._engine_pid = 0  # Already reset by cleanup!
    engine_client._engine_confirmed_dead = True  # But flag survives cleanup
    engine_client._start_engine_watcher = AsyncMock()

    with (
        patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
        patch("os.waitpid", side_effect=ChildProcessError),
    ):
        await engine_client._run_reconnect_cycle(timeout_seconds=5.0)

    engine_client.db.recreate_prisma_client.assert_awaited_once_with("postgresql://test")
    engine_client._start_engine_watcher.assert_awaited_once()
    engine_client.db.connect.assert_not_awaited()
    assert engine_client._engine_confirmed_dead is False  # Reset after use


@pytest.mark.asyncio
async def test_run_reconnect_cycle_uses_lightweight_path_when_engine_alive(
    engine_client,
) -> None:
    """_run_reconnect_cycle uses disconnect/connect when engine is alive."""
    engine_client._engine_pid = 1234

    with patch.object(engine_client, "_is_engine_alive", return_value=True):
        await engine_client._run_reconnect_cycle(timeout_seconds=5.0)

    engine_client.db.connect.assert_awaited_once()
    engine_client.db.query_raw.assert_awaited_once_with("SELECT 1")
    engine_client.db.recreate_prisma_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_reconnect_cycle_uses_lightweight_path_when_pid_unknown(
    engine_client,
) -> None:
    """_run_reconnect_cycle uses lightweight path when engine PID is not tracked."""
    engine_client._engine_pid = 0

    await engine_client._run_reconnect_cycle(timeout_seconds=5.0)

    engine_client.db.connect.assert_awaited_once()
    engine_client.db.query_raw.assert_awaited_once_with("SELECT 1")
    engine_client.db.recreate_prisma_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_reconnect_cycle_heavy_path_raises_without_database_url(
    engine_client,
) -> None:
    """Heavy reconnect raises RuntimeError when DATABASE_URL is not set."""
    engine_client._engine_pid = 1234

    with (
        patch.object(engine_client, "_is_engine_alive", return_value=False),
        patch.dict(os.environ, {}, clear=True),
        patch("os.waitpid", side_effect=ChildProcessError),
    ):
        with pytest.raises(RuntimeError, match="DATABASE_URL not set"):
            await engine_client._run_reconnect_cycle(timeout_seconds=5.0)

    engine_client.db.recreate_prisma_client.assert_not_awaited()


# ---------------------------------------------------------------------------
# start/stop lifecycle integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_watchdog_task_also_starts_engine_watcher(
    engine_client,
) -> None:
    """start_db_health_watchdog_task() also starts engine watcher."""
    engine_client._start_engine_watcher = AsyncMock()

    loop = asyncio.get_running_loop()
    dummy_task = loop.create_task(asyncio.sleep(3600))

    def fake_create_task(coro):
        coro.close()
        return dummy_task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        await engine_client.start_db_health_watchdog_task()

    engine_client._start_engine_watcher.assert_awaited_once()
    dummy_task.cancel()
    try:
        await dummy_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_stop_watchdog_task_also_stops_engine_watcher(
    engine_client,
) -> None:
    """stop_db_health_watchdog_task() also stops engine watcher."""
    engine_client._stop_engine_watcher = MagicMock()

    loop = asyncio.get_running_loop()
    dummy_task = loop.create_task(asyncio.sleep(3600))
    engine_client._db_health_watchdog_task = dummy_task

    await engine_client.stop_db_health_watchdog_task()

    engine_client._stop_engine_watcher.assert_called_once()
    assert engine_client._db_health_watchdog_task is None


# ---------------------------------------------------------------------------
# waitpid thread (cross-platform)
# ---------------------------------------------------------------------------


def test_try_waitpid_watch_returns_false_when_not_child(engine_client):
    """_try_waitpid_watch returns False when PID is not our child process."""
    engine_client._engine_pid = 9999
    with patch("os.waitpid", side_effect=ChildProcessError):
        assert engine_client._try_waitpid_watch(9999) is False
    assert engine_client._engine_wait_thread is None


def test_try_waitpid_watch_starts_thread_for_child(engine_client):
    """_try_waitpid_watch starts a daemon thread when PID is our child."""
    engine_client._engine_pid = 1234
    mock_thread = MagicMock()
    mock_loop = MagicMock()
    with (
        patch("os.waitpid", return_value=(0, 0)),
        patch("asyncio.get_running_loop", return_value=mock_loop),
        patch("threading.Thread", return_value=mock_thread) as mock_thread_cls,
    ):
        result = engine_client._try_waitpid_watch(1234)
    assert result is True
    mock_thread.start.assert_called_once()
    assert engine_client._engine_wait_thread is mock_thread


@pytest.mark.asyncio
async def test_try_waitpid_watch_handles_already_dead_engine(engine_client) -> None:
    """_try_waitpid_watch detects engine already dead at watch start."""
    engine_client._engine_pid = 1234
    engine_client.attempt_db_reconnect = AsyncMock(return_value=True)

    created_coros = []

    def capture_task(coro):
        created_coros.append(coro)
        return MagicMock()

    waitpid_calls = iter([(1234, 0)])

    def mock_waitpid(pid, flags):
        if pid == -1:
            raise ChildProcessError
        return next(waitpid_calls)

    with (
        patch("os.waitpid", side_effect=mock_waitpid),
        patch("asyncio.create_task", side_effect=capture_task),
    ):
        result = engine_client._try_waitpid_watch(1234)

    assert result is True
    assert engine_client._engine_confirmed_dead is True
    assert len(created_coros) == 1
    created_coros[0].close()


@pytest.mark.asyncio
async def test_on_engine_death_from_thread_triggers_reconnect(engine_client) -> None:
    """waitpid thread callback schedules attempt_db_reconnect."""
    engine_client._engine_pid = 1234
    engine_client.attempt_db_reconnect = AsyncMock(return_value=True)

    created_coros = []

    def capture_task(coro):
        created_coros.append(coro)
        return MagicMock()

    with patch("asyncio.create_task", side_effect=capture_task):
        engine_client._on_engine_death_from_thread(1234)

    assert len(created_coros) == 1
    await created_coros[0]

    engine_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="engine_process_death",
        force=True,
    )


def test_on_engine_death_from_thread_no_double_trigger(engine_client):
    """waitpid thread callback does not trigger reconnect if already confirmed dead."""
    engine_client._engine_pid = 1234
    engine_client._engine_confirmed_dead = True

    with patch("asyncio.create_task") as mock_create_task:
        engine_client._on_engine_death_from_thread(1234)

    mock_create_task.assert_not_called()


def test_on_engine_death_from_thread_ignores_stale_pid(engine_client):
    """waitpid thread callback ignores death notification for a stale PID."""
    engine_client._engine_pid = 5678

    with patch("asyncio.create_task") as mock_create_task:
        engine_client._on_engine_death_from_thread(1234)

    mock_create_task.assert_not_called()
