"""Pin ``PrismaClient`` engine watcher methods.

Symbols pinned here:
  - ``PrismaClient._get_engine_pid``
  - ``PrismaClient._is_engine_alive``
  - ``PrismaClient._reap_all_zombies``
  - ``PrismaClient._try_waitpid_watch``
  - ``PrismaClient._waitpid_thread_func``
  - ``PrismaClient._on_engine_death_from_thread``
  - ``PrismaClient._try_pidfd_watch``
  - ``PrismaClient._on_pidfd_readable``
  - ``PrismaClient._poll_engine_proc``
  - ``PrismaClient._cleanup_engine_watcher``
  - ``PrismaClient._start_engine_watcher``
  - ``PrismaClient._stop_engine_watcher``

Linux-only tests are skipped on Windows; the production code uses
``waitpid``/``pidfd_open`` which are Unix-only.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import PrismaClient


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="engine watcher is Unix-only"
)


def test_get_engine_pid_extracts_process_pid(prisma_client: PrismaClient) -> None:
    fake_engine = MagicMock()
    fake_engine.process = MagicMock()
    fake_engine.process.pid = 4242
    prisma_client.db._original_prisma = MagicMock()
    prisma_client.db._original_prisma._engine = fake_engine
    actual = {
        "pid": prisma_client._get_engine_pid(),
        "engine_attr": prisma_client.db._original_prisma._engine is fake_engine,
        "process_pid": fake_engine.process.pid,
    }
    assert actual == {"pid": 4242, "engine_attr": True, "process_pid": 4242}


def test_get_engine_pid_returns_zero_when_engine_attr_missing(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db._original_prisma = MagicMock(spec=[])
    assert prisma_client._get_engine_pid() == 0


def test_is_engine_alive_true_when_pid_zero(prisma_client: PrismaClient) -> None:
    prisma_client._engine_pid = 0
    pinned = {
        "result": prisma_client._is_engine_alive(),
        "pid": prisma_client._engine_pid,
        "type": type(prisma_client._is_engine_alive()).__name__,
    }
    assert pinned == {"result": True, "pid": 0, "type": "bool"}


def test_is_engine_alive_false_when_process_lookup_fails(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prisma_client._engine_pid = 99999
    monkeypatch.setattr(
        "os.kill", MagicMock(side_effect=ProcessLookupError())
    )
    assert prisma_client._is_engine_alive() is False


def test_is_engine_alive_true_on_permission_error(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prisma_client._engine_pid = 1
    monkeypatch.setattr("os.kill", MagicMock(side_effect=PermissionError()))
    assert prisma_client._is_engine_alive() is True


def test_reap_all_zombies_returns_set_of_reaped_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = iter([(111, 0), (222, 0), (0, 0)])

    def fake_waitpid(pid: int, flags: int) -> Any:
        return next(calls)

    monkeypatch.setattr("os.waitpid", fake_waitpid)
    reaped = PrismaClient._reap_all_zombies()
    pinned = {
        "type": type(reaped).__name__,
        "size": len(reaped),
        "contains_111": 111 in reaped,
        "contains_222": 222 in reaped,
    }
    assert pinned == {"type": "set", "size": 2, "contains_111": True, "contains_222": True}


def test_reap_all_zombies_handles_no_children_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "os.waitpid", MagicMock(side_effect=ChildProcessError())
    )
    assert PrismaClient._reap_all_zombies() == set()


@pytest.mark.asyncio
async def test_try_waitpid_watch_starts_thread_for_live_child(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.waitpid", MagicMock(return_value=(0, 0)))

    threads: list[threading.Thread] = []

    real_thread_cls = threading.Thread

    def _capture_thread(*args: Any, **kwargs: Any) -> threading.Thread:
        t = real_thread_cls(*args, **kwargs)
        threads.append(t)
        # Replace start so we don't actually launch the thread.
        t.start = MagicMock()  # type: ignore[method-assign]
        return t

    monkeypatch.setattr("threading.Thread", _capture_thread)
    monkeypatch.setattr(prisma_client, "_waitpid_thread_func", MagicMock())

    result = prisma_client._try_waitpid_watch(7777)
    pinned = {
        "returned": result,
        "threads_made": len(threads),
        "wait_thread_set": prisma_client._engine_wait_thread is threads[0],
        "thread_name_prefix": threads[0].name.startswith("prisma-engine-waitpid-"),
    }
    assert pinned == {
        "returned": True,
        "threads_made": 1,
        "wait_thread_set": True,
        "thread_name_prefix": True,
    }


@pytest.mark.asyncio
async def test_try_waitpid_watch_returns_false_for_non_child(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "os.waitpid", MagicMock(side_effect=ChildProcessError())
    )
    assert prisma_client._try_waitpid_watch(123) is False


@pytest.mark.asyncio
async def test_try_waitpid_watch_handles_already_dead_pid(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the engine PID is already dead at watch start, _try_waitpid_watch
    returns True and schedules a reconnect.
    """
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    monkeypatch.setattr("os.waitpid", MagicMock(return_value=(8888, 0)))
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))
    monkeypatch.setattr(prisma_client, "_cleanup_engine_watcher", MagicMock())

    result = prisma_client._try_waitpid_watch(8888)
    # Drain pending tasks so attempt_db_reconnect is awaited and we don't leak.
    await asyncio.sleep(0)
    pinned = {
        "result": result,
        "engine_confirmed_dead": prisma_client._engine_confirmed_dead,
        "cleanup_called": prisma_client._cleanup_engine_watcher.call_count,
        "reconnect_scheduled": prisma_client.attempt_db_reconnect.await_count >= 1,
    }
    assert pinned == {
        "result": True,
        "engine_confirmed_dead": True,
        "cleanup_called": 1,
        "reconnect_scheduled": True,
    }


def test_waitpid_thread_func_swallows_child_process_error(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.waitpid", MagicMock(side_effect=ChildProcessError()))
    loop = MagicMock()
    loop.call_soon_threadsafe = MagicMock()
    prisma_client._waitpid_thread_func(123, loop)
    assert loop.call_soon_threadsafe.call_count == 1


def test_waitpid_thread_func_invokes_on_engine_death_on_normal_exit(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.waitpid", MagicMock(return_value=(123, 0)))
    loop = MagicMock()
    received: list[Any] = []
    loop.call_soon_threadsafe = lambda fn, pid: received.append((fn, pid))
    prisma_client._waitpid_thread_func(123, loop)
    pinned = {
        "callbacks_received": len(received),
        "callback_target": received[0][0] == prisma_client._on_engine_death_from_thread,
        "pid_arg": received[0][1],
        "first_tuple_size": len(received[0]),
    }
    assert pinned == {
        "callbacks_received": 1,
        "callback_target": True,
        "pid_arg": 123,
        "first_tuple_size": 2,
    }


def test_waitpid_thread_func_swallows_loop_runtime_error(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.waitpid", MagicMock(return_value=(123, 0)))
    loop = MagicMock()
    loop.call_soon_threadsafe = MagicMock(side_effect=RuntimeError("loop closed"))
    prisma_client._waitpid_thread_func(123, loop)


@pytest.mark.asyncio
async def test_on_engine_death_from_thread_schedules_reconnect(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prisma_client._engine_pid = 7777
    prisma_client._engine_confirmed_dead = False
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))
    monkeypatch.setattr(prisma_client, "_cleanup_engine_watcher", MagicMock())

    prisma_client._on_engine_death_from_thread(7777)
    await asyncio.sleep(0)
    pinned = {
        "confirmed_dead": prisma_client._engine_confirmed_dead,
        "cleanup_called": prisma_client._cleanup_engine_watcher.call_count,
        "reconnect_called": prisma_client.attempt_db_reconnect.await_count,
        "reconnect_reason": prisma_client.attempt_db_reconnect.await_args.kwargs["reason"],
    }
    assert pinned == {
        "confirmed_dead": True,
        "cleanup_called": 1,
        "reconnect_called": 1,
        "reconnect_reason": "engine_process_death",
    }


def test_on_engine_death_from_thread_ignores_wrong_pid_or_already_dead(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._engine_pid = 1111
    prisma_client._engine_confirmed_dead = True
    prisma_client._cleanup_engine_watcher = MagicMock()
    prisma_client._on_engine_death_from_thread(1111)
    assert prisma_client._cleanup_engine_watcher.call_count == 0


def test_on_engine_death_from_thread_wrong_pid_does_nothing(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._engine_pid = 1111
    prisma_client._engine_confirmed_dead = False
    prisma_client._cleanup_engine_watcher = MagicMock()
    prisma_client._on_engine_death_from_thread(2222)
    assert prisma_client._cleanup_engine_watcher.call_count == 0
    assert prisma_client._engine_confirmed_dead is False


@pytest.mark.asyncio
async def test_try_pidfd_watch_returns_false_when_pidfd_open_missing(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr("os.pidfd_open", raising=False)
    assert prisma_client._try_pidfd_watch(123) is False


@pytest.mark.asyncio
async def test_try_pidfd_watch_arms_reader_when_available(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_pidfd(pid: int, flags: int) -> int:
        return 42

    monkeypatch.setattr("os.pidfd_open", fake_pidfd, raising=False)
    loop = asyncio.get_running_loop()
    fake_add_reader = MagicMock()
    monkeypatch.setattr(loop, "add_reader", fake_add_reader)

    result = prisma_client._try_pidfd_watch(123)
    assert result is True
    assert prisma_client._engine_pidfd == 42
    assert fake_add_reader.call_args.args[0] == 42


@pytest.mark.asyncio
async def test_try_pidfd_watch_error_returns_false_and_cleans_up(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_pidfd(pid: int, flags: int) -> int:
        raise OSError("ENOSYS")

    monkeypatch.setattr("os.pidfd_open", fake_pidfd, raising=False)
    assert prisma_client._try_pidfd_watch(123) is False
    assert prisma_client._engine_pidfd == -1


@pytest.mark.asyncio
async def test_on_pidfd_readable_invokes_reconnect_path(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prisma_client._engine_pid = 4321
    prisma_client._engine_confirmed_dead = False
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))
    cleanup = MagicMock()
    prisma_client._cleanup_engine_watcher = cleanup

    prisma_client._on_pidfd_readable()
    await asyncio.sleep(0)
    pinned = {
        "confirmed_dead": prisma_client._engine_confirmed_dead,
        "cleanup_called": cleanup.call_count,
        "reconnect_called": prisma_client.attempt_db_reconnect.await_count,
        "force_kwarg": prisma_client.attempt_db_reconnect.await_args.kwargs["force"],
    }
    assert pinned == {
        "confirmed_dead": True,
        "cleanup_called": 1,
        "reconnect_called": 1,
        "force_kwarg": True,
    }


@pytest.mark.asyncio
async def test_on_pidfd_readable_noop_when_already_dead_closes_pidfd(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When _engine_confirmed_dead is already True, the reader handler should
    not schedule another reconnect and should release the pidfd resource.
    """
    closed: list[int] = []
    monkeypatch.setattr("os.close", lambda fd: closed.append(fd))
    loop = asyncio.get_running_loop()
    removed: list[int] = []
    monkeypatch.setattr(loop, "remove_reader", lambda fd: removed.append(fd))

    prisma_client._engine_confirmed_dead = True
    prisma_client._engine_pidfd = 99
    prisma_client.attempt_db_reconnect = AsyncMock()

    prisma_client._on_pidfd_readable()
    pinned = {
        "engine_pidfd": prisma_client._engine_pidfd,
        "closed": closed,
        "removed": removed,
        "reconnect_call_count": prisma_client.attempt_db_reconnect.await_count,
    }
    assert pinned == {
        "engine_pidfd": -1,
        "closed": [99],
        "removed": [99],
        "reconnect_call_count": 0,
    }


@pytest.mark.asyncio
async def test_poll_engine_proc_detects_death_and_reconnects(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prisma_client._engine_pid = 555
    prisma_client._watching_engine = True
    prisma_client.attempt_db_reconnect = AsyncMock()
    monkeypatch.setattr("os.kill", MagicMock(side_effect=ProcessLookupError()))
    monkeypatch.setattr(PrismaClient, "_reap_all_zombies", staticmethod(lambda: set()))
    prisma_client._cleanup_engine_watcher = MagicMock()

    await prisma_client._poll_engine_proc()
    pinned = {
        "reconnect_count": prisma_client.attempt_db_reconnect.await_count,
        "cleanup_count": prisma_client._cleanup_engine_watcher.call_count,
        "confirmed_dead": prisma_client._engine_confirmed_dead,
        "reason": prisma_client.attempt_db_reconnect.await_args.kwargs["reason"],
    }
    assert pinned == {
        "reconnect_count": 1,
        "cleanup_count": 1,
        "confirmed_dead": True,
        "reason": "engine_process_death",
    }


@pytest.mark.asyncio
async def test_poll_engine_proc_returns_on_permission_error(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prisma_client._engine_pid = 555
    prisma_client._watching_engine = True
    monkeypatch.setattr("os.kill", MagicMock(side_effect=PermissionError()))
    prisma_client._cleanup_engine_watcher = MagicMock()
    await prisma_client._poll_engine_proc()
    assert prisma_client._cleanup_engine_watcher.call_count == 1


@pytest.mark.asyncio
async def test_cleanup_engine_watcher_resets_state(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []
    monkeypatch.setattr("os.close", lambda fd: closed.append(fd))
    loop = asyncio.get_running_loop()
    removed: list[int] = []
    monkeypatch.setattr(loop, "remove_reader", lambda fd: removed.append(fd))

    prisma_client._engine_pidfd = 42
    prisma_client._engine_pid = 999
    prisma_client._engine_wait_thread = MagicMock()
    prisma_client._watching_engine = True

    prisma_client._cleanup_engine_watcher()
    pinned = {
        "engine_pidfd": prisma_client._engine_pidfd,
        "engine_pid": prisma_client._engine_pid,
        "wait_thread": prisma_client._engine_wait_thread,
        "watching": prisma_client._watching_engine,
        "closed": closed,
        "removed": removed,
    }
    assert pinned == {
        "engine_pidfd": -1,
        "engine_pid": 0,
        "wait_thread": None,
        "watching": False,
        "closed": [42],
        "removed": [42],
    }


@pytest.mark.asyncio
async def test_cleanup_engine_watcher_swallows_close_error(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.close", MagicMock(side_effect=OSError("bad fd")))
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "remove_reader", MagicMock(side_effect=Exception("boom")))
    prisma_client._engine_pidfd = 99
    prisma_client._cleanup_engine_watcher()
    assert prisma_client._engine_pidfd == -1


@pytest.mark.asyncio
async def test_start_engine_watcher_picks_waitpid_when_available(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prisma_client, "_get_engine_pid", MagicMock(return_value=12345))
    monkeypatch.setattr(prisma_client, "_try_waitpid_watch", MagicMock(return_value=True))
    pidfd_called = MagicMock(return_value=False)
    monkeypatch.setattr(prisma_client, "_try_pidfd_watch", pidfd_called)
    await prisma_client._start_engine_watcher()
    pinned = {
        "engine_pid": prisma_client._engine_pid,
        "confirmed_dead_reset": prisma_client._engine_confirmed_dead,
        "waitpid_called": prisma_client._try_waitpid_watch.call_count,
        "pidfd_skipped": pidfd_called.call_count,
    }
    assert pinned == {
        "engine_pid": 12345,
        "confirmed_dead_reset": False,
        "waitpid_called": 1,
        "pidfd_skipped": 0,
    }


@pytest.mark.asyncio
async def test_start_engine_watcher_returns_early_when_pid_unknown(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prisma_client, "_get_engine_pid", MagicMock(return_value=0))
    monkeypatch.setattr(prisma_client, "_try_waitpid_watch", MagicMock())
    await prisma_client._start_engine_watcher()
    assert prisma_client._try_waitpid_watch.call_count == 0


@pytest.mark.asyncio
async def test_start_engine_watcher_falls_back_to_polling_when_no_kernel_apis(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prisma_client, "_get_engine_pid", MagicMock(return_value=4242))
    monkeypatch.setattr(prisma_client, "_try_waitpid_watch", MagicMock(return_value=False))
    monkeypatch.setattr(prisma_client, "_try_pidfd_watch", MagicMock(return_value=False))
    monkeypatch.setattr(prisma_client, "_poll_engine_proc", AsyncMock())
    await prisma_client._start_engine_watcher()
    await asyncio.sleep(0)
    assert prisma_client._watching_engine is True


def test_stop_engine_watcher_clears_dead_flag(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._engine_confirmed_dead = True
    prisma_client._cleanup_engine_watcher = MagicMock()
    prisma_client._stop_engine_watcher()
    assert prisma_client._cleanup_engine_watcher.call_count == 1
    assert prisma_client._engine_confirmed_dead is False


def test_stop_engine_watcher_error_in_cleanup_propagates(
    prisma_client: PrismaClient,
) -> None:
    prisma_client._cleanup_engine_watcher = MagicMock(side_effect=RuntimeError("cleanup boom"))
    with pytest.raises(RuntimeError, match="cleanup boom"):
        prisma_client._stop_engine_watcher()
