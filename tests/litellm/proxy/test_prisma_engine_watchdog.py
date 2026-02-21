"""
Tests for PrismaClient engine watchdog: death detection and automatic reconnect.

Covers:
- Engine PID discovery and liveness check
- Process disappears from /proc → reconnect triggered via attempt_db_reconnect
- Process becomes zombie in /proc → reconnect triggered via attempt_db_reconnect
- pidfd handler → schedules attempt_db_reconnect even when lock is held
- SIGCHLD handler → reaps all zombies, triggers reconnect if engine reaped
- _run_reconnect_cycle branches: heavy path (engine dead) vs lightweight path (engine alive)
- _engine_confirmed_dead flag ensures heavy reconnect even after _engine_pid reset
- Successful heavy reconnect → watcher re-armed for new process
- Missing DATABASE_URL → graceful RuntimeError in reconnect cycle
- Shutdown → polling loop exits cleanly, SIGCHLD handler removed
"""

import asyncio
import os
import signal
import time
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

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
    """_is_engine_alive returns False when /proc/<pid>/stat is missing."""
    engine_client._engine_pid = 9999
    with patch("builtins.open", side_effect=FileNotFoundError):
        assert engine_client._is_engine_alive() is False


def test_is_engine_alive_returns_false_for_zombie(engine_client):
    """_is_engine_alive returns False when process is in zombie state."""
    engine_client._engine_pid = 1234
    zombie_stat = "1234 (prisma-query-engine) Z 1\n"
    with patch("builtins.open", mock_open(read_data=zombie_stat)):
        assert engine_client._is_engine_alive() is False


def test_is_engine_alive_returns_true_for_running_process(engine_client):
    """_is_engine_alive returns True when process is in sleeping state."""
    engine_client._engine_pid = 1234
    alive_stat = "1234 (prisma-query-engine) S 1\n"
    with patch("builtins.open", mock_open(read_data=alive_stat)):
        assert engine_client._is_engine_alive() is True


# ---------------------------------------------------------------------------
# _poll_engine_proc — calls attempt_db_reconnect on death
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_missing_process_triggers_reconnect(engine_client) -> None:
    """Polling loop triggers attempt_db_reconnect when the engine process disappears."""
    engine_client._engine_pid = 1234
    engine_client._watching_engine = True
    engine_client.attempt_db_reconnect = AsyncMock(return_value=True)

    with patch("builtins.open", side_effect=FileNotFoundError):
        await engine_client._poll_engine_proc()

    engine_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="engine_process_death",
        force=True,
    )


@pytest.mark.asyncio
async def test_poll_zombie_process_triggers_reconnect(engine_client) -> None:
    """Polling loop triggers attempt_db_reconnect when the engine enters zombie state."""
    engine_client._engine_pid = 1234
    engine_client._watching_engine = True
    engine_client.attempt_db_reconnect = AsyncMock(return_value=True)

    zombie_stat = "1234 (prisma-query-engine) Z 1\n"
    with patch("builtins.open", mock_open(read_data=zombie_stat)):
        await engine_client._poll_engine_proc()

    engine_client.attempt_db_reconnect.assert_awaited_once_with(
        reason="engine_process_death",
        force=True,
    )


@pytest.mark.asyncio
async def test_stop_loop_halts_polling(engine_client) -> None:
    """Polling loop exits cleanly when _stop_engine_watcher is called."""
    engine_client._engine_pid = 1234
    engine_client._watching_engine = True

    alive_stat = "1234 (prisma-query-engine) S 1\n"

    async def stop_during_sleep(_duration: float) -> None:
        engine_client._stop_engine_watcher()

    with (
        patch("builtins.open", mock_open(read_data=alive_stat)),
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
    """stop_db_health_watchdog_task() also stops engine watcher and SIGCHLD handler."""
    engine_client._stop_engine_watcher = MagicMock()

    loop = asyncio.get_running_loop()
    dummy_task = loop.create_task(asyncio.sleep(3600))
    engine_client._db_health_watchdog_task = dummy_task

    await engine_client.stop_db_health_watchdog_task()

    engine_client._stop_engine_watcher.assert_called_once()
    assert engine_client._db_health_watchdog_task is None


# ---------------------------------------------------------------------------
# SIGCHLD handler
# ---------------------------------------------------------------------------


def test_sigchld_handler_reaps_engine_and_triggers_reconnect(engine_client):
    """SIGCHLD handler detects engine death, reaps zombies, triggers reconnect."""
    engine_client._engine_pid = 1234
    created_coros = []

    def capture_task(coro):
        created_coros.append(coro)
        return MagicMock()

    # Simulate waitpid returning the engine PID then raising ChildProcessError
    with (
        patch("os.waitpid", side_effect=[(1234, 0), ChildProcessError]),
        patch("asyncio.create_task", side_effect=capture_task),
    ):
        engine_client._on_sigchld()

    assert engine_client._engine_confirmed_dead is True
    assert engine_client._engine_pid == 0  # cleanup ran
    assert len(created_coros) == 1
    # Clean up the coroutine
    created_coros[0].close()


def test_sigchld_handler_ignores_non_engine_zombies(engine_client):
    """SIGCHLD handler reaps non-engine zombies without triggering reconnect."""
    engine_client._engine_pid = 1234

    # Simulate reaping PID 5555 (not the engine)
    with patch("os.waitpid", side_effect=[(5555, 0), ChildProcessError]):
        engine_client._on_sigchld()

    assert engine_client._engine_confirmed_dead is False
    assert engine_client._engine_pid == 1234  # unchanged


def test_sigchld_handler_no_double_trigger(engine_client):
    """SIGCHLD handler does not trigger reconnect if already confirmed dead."""
    engine_client._engine_pid = 1234
    engine_client._engine_confirmed_dead = True  # Already handled

    with (
        patch("os.waitpid", side_effect=[(1234, 0), ChildProcessError]),
        patch("asyncio.create_task") as mock_create_task,
    ):
        engine_client._on_sigchld()

    mock_create_task.assert_not_called()


def test_install_sigchld_handler_success(engine_client):
    """SIGCHLD handler installs on a running event loop."""
    mock_loop = MagicMock()
    with patch("asyncio.get_running_loop", return_value=mock_loop):
        assert engine_client._install_sigchld_handler() is True

    assert engine_client._sigchld_installed is True
    mock_loop.add_signal_handler.assert_called_once_with(
        signal.SIGCHLD, engine_client._on_sigchld
    )


def test_install_sigchld_handler_no_loop(engine_client):
    """SIGCHLD handler returns False when no event loop is running."""
    with patch("asyncio.get_running_loop", side_effect=RuntimeError):
        assert engine_client._install_sigchld_handler() is False

    assert engine_client._sigchld_installed is False


def test_remove_sigchld_handler(engine_client):
    """SIGCHLD handler is properly removed."""
    engine_client._sigchld_installed = True
    mock_loop = MagicMock()
    with patch("asyncio.get_running_loop", return_value=mock_loop):
        engine_client._remove_sigchld_handler()

    assert engine_client._sigchld_installed is False
    mock_loop.remove_signal_handler.assert_called_once_with(signal.SIGCHLD)
