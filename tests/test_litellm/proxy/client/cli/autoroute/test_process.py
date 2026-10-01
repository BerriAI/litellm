import os
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from litellm.proxy.client.cli.commands.autoroute import process as process_module
from litellm.proxy.client.cli.commands.autoroute.process import (
    PidRecord,
    ProcessLaunchError,
    UpError,
    clear_pid_record,
    is_port_available,
    is_running,
    launch_proxy,
    missing_proxy_runtime_modules,
    poll_liveliness,
    read_pid_record,
    write_pid_record,
)


class FakeProcess:
    def __init__(self, returncode: int | None = None):
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class TestIsPortAvailable:
    def test_true_for_a_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            free_port = sock.getsockname()[1]
        assert is_port_available(free_port) is True

    def test_false_while_another_socket_holds_the_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            held_port = sock.getsockname()[1]
            assert is_port_available(held_port) is False


class TestLaunchProxy:
    def test_binds_loopback_only_not_all_interfaces(self, tmp_path):
        """proxy_cli.py's own --host default is 0.0.0.0 -- without an explicit override here, the
        ephemeral proxy would be reachable from other hosts on the network despite base_url always
        being built from 127.0.0.1, exposing its unauthenticated-until-master-key-lands routes."""
        config_path = tmp_path / "config.yaml"
        log_path = tmp_path / "proxy.log"

        with patch.object(process_module.subprocess, "Popen") as mock_popen:
            launch_proxy(config_path, 12345, log_path)

        args = mock_popen.call_args[0][0]
        assert "--host" in args
        assert args[args.index("--host") + 1] == "127.0.0.1"


class TestPidRecordRoundTrip:
    def test_write_then_read_round_trips(self, tmp_path):
        path = tmp_path / "pid.json"
        record = PidRecord(pid=123, port=4000, config_path="/tmp/config.yaml", log_path="/tmp/proxy.log")

        write_pid_record(record, path)

        assert read_pid_record(path) == record

    def test_read_missing_file_returns_none(self, tmp_path):
        assert read_pid_record(tmp_path / "missing.json") is None

    def test_read_raises_clean_error_on_corrupt_content(self, tmp_path):
        path = tmp_path / "pid.json"
        path.write_text("not json at all {{{")

        with pytest.raises(UpError, match="invalid or unexpected JSON"):
            read_pid_record(path)

    def test_clear_removes_an_existing_record(self, tmp_path):
        path = tmp_path / "pid.json"
        write_pid_record(PidRecord(pid=1, port=1, config_path="a", log_path="b"), path)
        assert path.exists()

        clear_pid_record(path)

        assert not path.exists()

    def test_clear_missing_file_is_a_no_op(self, tmp_path):
        clear_pid_record(tmp_path / "missing.json")

    def test_write_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "pid.json"

        write_pid_record(PidRecord(pid=1, port=1, config_path="a", log_path="b"), path)

        assert path.exists()


class TestIsRunning:
    def test_current_process_is_running(self):
        assert is_running(os.getpid()) is True

    def test_huge_unlikely_pid_is_not_running(self):
        assert is_running(2**30) is False

    def test_permission_error_from_kill_is_treated_as_running(self, monkeypatch):
        # The os.kill probe only runs on non-Windows platforms; on Windows liveness is
        # checked via OpenProcess/GetExitCodeProcess instead.
        monkeypatch.setattr(process_module.sys, "platform", "linux")

        def fake_kill(pid: int, sig: int) -> None:
            raise PermissionError("not permitted to signal this pid")

        monkeypatch.setattr(process_module.os, "kill", fake_kill)

        assert is_running(999) is True

    def test_windows_uses_handle_probe_instead_of_kill(self, monkeypatch):
        """os.kill(pid, 0) is not a usable liveness probe on Windows (it would kill the
        process before Python 3.14, and raises a bare OSError for dead pids on 3.14+),
        so is_running must not call os.kill at all on that platform."""
        calls = []

        def fake_kill(pid: int, sig: int) -> None:
            calls.append((pid, sig))

        monkeypatch.setattr(process_module.sys, "platform", "win32")
        monkeypatch.setattr(process_module.os, "kill", fake_kill)
        monkeypatch.setattr(process_module, "_is_running_win32", lambda pid: True)

        assert is_running(1234) is True
        assert calls == []

    def test_windows_probe_reports_dead_pid(self, monkeypatch):
        monkeypatch.setattr(process_module.sys, "platform", "win32")
        monkeypatch.setattr(process_module, "_is_running_win32", lambda pid: False)

        assert is_running(1234) is False


class _FakeKernel32:
    """Minimal kernel32 stand-in so the Win32 probe can be exercised functionally on any OS."""

    def __init__(self, handle=1, exit_code=259, get_exit_ok=True):
        self._handle = handle
        self._exit_code = exit_code
        self._get_exit_ok = get_exit_ok
        self.closed_handle = None

    def OpenProcess(self, access, inherit, pid):
        return self._handle

    def GetExitCodeProcess(self, handle, ref):
        if not self._get_exit_ok:
            return 0
        ref._obj.value = self._exit_code
        return 1

    def CloseHandle(self, handle):
        self.closed_handle = handle


class TestIsRunningWin32Probe:
    """Functional tests for _is_running_win32 with a mocked kernel32 (CI runs on Linux,
    so the real OpenProcess never executes there)."""

    def _install(self, monkeypatch, kernel32, last_error=0):
        import ctypes

        monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
        monkeypatch.setattr(ctypes, "GetLastError", lambda: last_error, raising=False)

    def test_still_active_process_is_running(self, monkeypatch):
        self._install(monkeypatch, _FakeKernel32(exit_code=259))  # STILL_ACTIVE

        assert process_module._is_running_win32(1234) is True

    def test_exited_process_is_not_running(self, monkeypatch):
        self._install(monkeypatch, _FakeKernel32(exit_code=0))

        assert process_module._is_running_win32(1234) is False

    def test_unopenable_pid_with_access_denied_is_running(self, monkeypatch):
        # The pid exists but we lack permission to query it.
        self._install(monkeypatch, _FakeKernel32(handle=0), last_error=5)  # ERROR_ACCESS_DENIED

        assert process_module._is_running_win32(1234) is True

    def test_unopenable_pid_with_other_error_is_not_running(self, monkeypatch):
        self._install(monkeypatch, _FakeKernel32(handle=0), last_error=87)  # ERROR_INVALID_PARAMETER

        assert process_module._is_running_win32(1234) is False

    def test_failed_exit_code_query_is_not_running(self, monkeypatch):
        self._install(monkeypatch, _FakeKernel32(get_exit_ok=False))

        assert process_module._is_running_win32(1234) is False

    def test_handle_is_closed(self, monkeypatch):
        kernel32 = _FakeKernel32()
        self._install(monkeypatch, kernel32)

        process_module._is_running_win32(1234)

        assert kernel32.closed_handle == 1


class TestTerminate:
    def test_skips_sigkill_escalation_when_signal_is_unavailable(self, monkeypatch):
        """Windows has no signal.SIGKILL; escalating there must not crash with AttributeError
        (os.kill with SIGTERM already force-terminates via TerminateProcess)."""
        sent = []

        def fake_kill(pid: int, sig: int) -> None:
            sent.append(sig)

        monkeypatch.setattr(process_module.os, "kill", fake_kill)
        monkeypatch.setattr(process_module, "is_running", lambda pid: True)
        monkeypatch.setattr(process_module.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(process_module.signal, "SIGKILL", None, raising=False)

        process_module.terminate(1234, grace_period=0.0)

        assert sent == [process_module.signal.SIGTERM]

    def test_escalates_to_sigkill_when_available(self, monkeypatch):
        sent = []

        def fake_kill(pid: int, sig: int) -> None:
            sent.append(sig)

        monkeypatch.setattr(process_module.os, "kill", fake_kill)
        monkeypatch.setattr(process_module, "is_running", lambda pid: True)
        monkeypatch.setattr(process_module.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(process_module.signal, "SIGKILL", 9, raising=False)

        process_module.terminate(1234, grace_period=0.0)

        assert sent == [process_module.signal.SIGTERM, 9]


class TestPollLiveliness:
    def test_succeeds_when_health_check_returns_200_quickly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(process_module.requests, "get", lambda url, timeout: FakeResponse(200))

        poll_liveliness("http://127.0.0.1:4000", tmp_path / "proxy.log", FakeProcess(), timeout=5.0)

    def test_raises_with_log_tail_when_timeout_elapses(self, monkeypatch, tmp_path):
        log_path = tmp_path / "proxy.log"
        log_path.write_text("line one\nline two\nline three\n")
        monkeypatch.setattr(process_module.requests, "get", lambda url, timeout: FakeResponse(500))
        monkeypatch.setattr(process_module.time, "sleep", lambda seconds: None)

        with pytest.raises(ProcessLaunchError) as exc_info:
            poll_liveliness("http://127.0.0.1:4000", log_path, FakeProcess(), timeout=0.05)

        assert "never became healthy" in str(exc_info.value)
        assert "line three" in str(exc_info.value)

    def test_raises_immediately_when_process_already_exited(self, tmp_path):
        log_path = tmp_path / "proxy.log"
        log_path.write_text("crash log line")

        with pytest.raises(ProcessLaunchError) as exc_info:
            poll_liveliness("http://127.0.0.1:4000", log_path, FakeProcess(returncode=1), timeout=5.0)

        assert "exited early" in str(exc_info.value)
        assert "crash log line" in str(exc_info.value)


class TestMissingProxyRuntimeModules:
    def test_flags_absent_modules_only(self, monkeypatch):
        """A thin litellm[cli] install lacks the proxy runtime; the missing ones must be reported
        (by name, for an actionable error) while modules that are importable are not."""
        monkeypatch.setattr(
            process_module,
            "_PROXY_RUNTIME_MODULES",
            ("os", "litellm_autoroute_definitely_absent_pkg", "socket"),
        )

        assert missing_proxy_runtime_modules() == ("litellm_autoroute_definitely_absent_pkg",)

    def test_empty_when_all_present(self, monkeypatch):
        monkeypatch.setattr(process_module, "_PROXY_RUNTIME_MODULES", ("os", "socket"))

        assert missing_proxy_runtime_modules() == ()
