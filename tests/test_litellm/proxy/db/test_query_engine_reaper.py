import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.db.query_engine_reaper import (
    REAPER_THREAD_NAME,
    _read_comm_and_ppid,
    _reaper_loop,
    _send_signal,
    _try_reap,
    list_orphaned_engine_pids,
    reap_orphaned_engines,
    set_child_subreaper,
    start_query_engine_reaper,
    terminate_and_reap,
    terminate_and_reap_all,
)


def _write_stat(proc_root, pid, comm, ppid):
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir()
    (pid_dir / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194304 100 0 0 0")


class TestReadCommAndPpid:
    def test_parses_comm_and_ppid(self, tmp_path):
        _write_stat(tmp_path, 137, "query-engine-de", 81)
        assert _read_comm_and_ppid(137, str(tmp_path)) == ("query-engine-de", 81)

    def test_comm_containing_parens_and_spaces(self, tmp_path):
        _write_stat(tmp_path, 42, "weird) (name", 1)
        assert _read_comm_and_ppid(42, str(tmp_path)) == ("weird) (name", 1)

    def test_missing_pid_returns_none(self, tmp_path):
        assert _read_comm_and_ppid(999, str(tmp_path)) is None

    def test_malformed_stat_returns_none(self, tmp_path):
        pid_dir = tmp_path / "55"
        pid_dir.mkdir()
        (pid_dir / "stat").write_text("garbage with no parens")
        assert _read_comm_and_ppid(55, str(tmp_path)) is None

    def test_truncated_fields_after_comm_returns_none(self, tmp_path):
        pid_dir = tmp_path / "56"
        pid_dir.mkdir()
        (pid_dir / "stat").write_text("56 (proc) S")
        assert _read_comm_and_ppid(56, str(tmp_path)) is None

    def test_non_numeric_ppid_returns_none(self, tmp_path):
        pid_dir = tmp_path / "57"
        pid_dir.mkdir()
        (pid_dir / "stat").write_text("57 (proc) S notanint 57")
        assert _read_comm_and_ppid(57, str(tmp_path)) is None


class TestListOrphanedEnginePids:
    def test_finds_only_engine_children_of_parent(self, tmp_path):
        _write_stat(tmp_path, 137, "query-engine-de", 1)
        _write_stat(tmp_path, 138, "query-engine-de", 1)
        _write_stat(tmp_path, 260, "python", 1)
        _write_stat(tmp_path, 285, "query-engine-de", 260)
        (tmp_path / "not-a-pid").mkdir()

        assert sorted(list_orphaned_engine_pids(1, proc_root=str(tmp_path))) == [137, 138]

    def test_no_matches_returns_empty(self, tmp_path):
        _write_stat(tmp_path, 260, "python", 1)
        assert list_orphaned_engine_pids(1, proc_root=str(tmp_path)) == ()

    def test_missing_proc_root_returns_empty(self, tmp_path):
        assert list_orphaned_engine_pids(1, proc_root=str(tmp_path / "absent")) == ()


class TestSetChildSubreaper:
    def test_matches_platform_capability(self):
        result = set_child_subreaper()
        if sys.platform.startswith("linux"):
            assert result is True
        else:
            assert result is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and waitpid")
class TestSignalHelpers:
    def test_try_reap_true_for_non_child_pid(self):
        assert _try_reap(1) is True

    def test_send_signal_swallows_missing_pid(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        _send_signal(child.pid, signal.SIGTERM)


class TestReaperLoop:
    def test_survives_scan_failure_and_continues(self):
        calls = []

        def flaky_scan(parent_pid, proc_root="/proc"):
            calls.append(parent_pid)
            if len(calls) == 1:
                raise RuntimeError("scan blew up")
            raise KeyboardInterrupt

        with (
            patch(
                "litellm.proxy.db.query_engine_reaper.reap_orphaned_engines",
                side_effect=flaky_scan,
            ),
            patch("litellm.proxy.db.query_engine_reaper.time.sleep"),
            pytest.raises(KeyboardInterrupt),
        ):
            _reaper_loop(1234)

        assert calls == [1234, 1234]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and waitpid")
class TestTerminateAndReap:
    def test_sigterm_terminates_and_reaps_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        terminate_and_reap(child.pid, grace_seconds=10.0)

        with pytest.raises((ChildProcessError, OSError)):
            os.waitpid(child.pid, os.WNOHANG)
        child.returncode = -signal.SIGTERM

    def test_escalates_to_sigkill_when_sigterm_ignored(self):
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
            ]
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            probe = subprocess.run(
                [sys.executable, "-c", f"import os, signal; os.kill({child.pid}, 0)"],
                capture_output=True,
            )
            if probe.returncode == 0:
                break
            time.sleep(0.05)
        time.sleep(0.3)

        terminate_and_reap(child.pid, grace_seconds=0.5)

        with pytest.raises((ChildProcessError, OSError)):
            os.waitpid(child.pid, os.WNOHANG)
        child.returncode = -signal.SIGKILL


class TestReapOrphanedEngines:
    def test_terminates_each_orphan(self, tmp_path):
        _write_stat(tmp_path, 137, "query-engine-de", 1)
        _write_stat(tmp_path, 138, "query-engine-de", 1)
        _write_stat(tmp_path, 285, "query-engine-de", 260)

        with patch("litellm.proxy.db.query_engine_reaper.terminate_and_reap_all") as mock_terminate:
            acted_on = reap_orphaned_engines(1, proc_root=str(tmp_path))

        assert sorted(acted_on) == [137, 138]
        assert sorted(mock_terminate.call_args.args[0]) == [137, 138]

    def test_no_orphans_no_kills(self, tmp_path):
        _write_stat(tmp_path, 285, "query-engine-de", 260)

        with patch("litellm.proxy.db.query_engine_reaper.terminate_and_reap_all") as mock_terminate:
            assert reap_orphaned_engines(1, proc_root=str(tmp_path)) == ()

        mock_terminate.assert_not_called()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and waitpid")
class TestTerminateAndReapAll:
    def test_batch_shares_one_grace_period(self):
        children = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
                ]
            )
            for _ in range(3)
        ]
        time.sleep(0.5)

        start = time.monotonic()
        terminate_and_reap_all(tuple(child.pid for child in children), grace_seconds=1.0)
        elapsed = time.monotonic() - start

        assert elapsed < 3.0
        for child in children:
            with pytest.raises((ChildProcessError, OSError)):
                os.waitpid(child.pid, os.WNOHANG)
            child.returncode = -signal.SIGKILL


class TestStartQueryEngineReaper:
    def test_noop_on_non_linux(self):
        with patch("litellm.proxy.db.query_engine_reaper.sys.platform", "darwin"):
            assert start_query_engine_reaper() is None

    def test_starts_daemon_thread_on_linux(self):
        with (
            patch("litellm.proxy.db.query_engine_reaper.sys.platform", "linux"),
            patch(
                "litellm.proxy.db.query_engine_reaper.threading.enumerate",
                return_value=[],
            ),
            patch("litellm.proxy.db.query_engine_reaper.set_child_subreaper") as mock_subreaper,
            patch("litellm.proxy.db.query_engine_reaper.threading.Thread") as mock_thread_cls,
        ):
            thread = start_query_engine_reaper()

        mock_subreaper.assert_called_once()
        mock_thread_cls.assert_called_once()
        assert mock_thread_cls.call_args.kwargs["daemon"] is True
        assert mock_thread_cls.call_args.kwargs["args"] == (os.getpid(),)
        mock_thread_cls.return_value.start.assert_called_once()
        assert thread is mock_thread_cls.return_value

    def test_second_call_returns_existing_thread(self):
        existing = MagicMock()
        existing.name = REAPER_THREAD_NAME
        with (
            patch("litellm.proxy.db.query_engine_reaper.sys.platform", "linux"),
            patch(
                "litellm.proxy.db.query_engine_reaper.threading.enumerate",
                return_value=[existing],
            ),
            patch("litellm.proxy.db.query_engine_reaper.threading.Thread") as mock_thread_cls,
        ):
            thread = start_query_engine_reaper()

        assert thread is existing
        mock_thread_cls.assert_not_called()
