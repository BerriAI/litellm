import os
import signal
import socket
import subprocess
import sys
import time
from typing import Final, Optional
from unittest.mock import Mock, patch

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
    terminate,
    write_pid_record,
)


class FakeProcess:
    def __init__(self, returncode: Optional[int] = None):
        self.returncode = returncode

    def poll(self) -> Optional[int]:
        return self.returncode


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def _signals_sent(fake_kill: Mock) -> tuple[int, ...]:
    """Signal numbers ``os.kill`` was called with, in call order."""
    return tuple(call.args[1] for call in fake_kill.call_args_list)


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

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX branch only; win32 answers with a query-only process handle, not os.kill",
    )
    def test_permission_error_from_kill_is_treated_as_running(self, monkeypatch):
        def fake_kill(pid: int, sig: int) -> None:
            raise PermissionError("not permitted to signal this pid")

        monkeypatch.setattr(process_module.os, "kill", fake_kill)

        assert is_running(999) is True


class TestIsRunningDoesNotSignal:
    """``os.kill(pid, 0)`` is not a liveness probe on Windows.

    ``signal.CTRL_C_EVENT`` is 0, so the call reaches
    ``GenerateConsoleCtrlEvent``, whose second argument is a process *group*.
    ``launch_proxy`` spawns the proxy without ``CREATE_NEW_PROCESS_GROUP``, so
    the child shares this console's group and the Ctrl-C hits the child, the
    caller, and anything else attached to the console.

    Measured on Windows 11, Python 3.12.10: probing a sleeping child returns
    ``True`` with no exception, the child then exits 3221225786
    (``0xC000013A`` / ``STATUS_CONTROL_C_EXIT``), and a ``KeyboardInterrupt``
    lands in the caller afterwards -- a ``BaseException``, so the handlers in
    ``is_running`` never see it.
    """

    def test_probing_a_child_does_not_terminate_it(self):
        """The behavioural check, with nothing mocked."""
        child: Final = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(30)"))
        try:
            assert is_running(child.pid) is True
            time.sleep(1.0)
            assert child.poll() is None, (
                f"the liveness probe terminated the process it was asked about "
                f"(exit code {child.returncode})"
            )
            assert is_running(child.pid) is True
        finally:
            child.kill()
            child.wait(timeout=10)

    def test_probing_our_own_pid_leaves_this_process_alone(self):
        """What ``TestIsRunning.test_current_process_is_running`` above relies on.

        The assertion there passes either way; it is the interpreter surviving
        the next statement that separates a probe from a signal.
        """
        assert is_running(os.getpid()) is True
        time.sleep(0.5)
        assert is_running(os.getpid()) is True

    def test_win32_answers_without_calling_os_kill(self, monkeypatch):
        fake_kill: Final = Mock()
        monkeypatch.setattr(process_module.sys, "platform", "win32")
        monkeypatch.setattr(process_module.os, "kill", fake_kill)
        monkeypatch.setattr(process_module, "_windows_pid_exists", lambda pid: True)

        assert is_running(1234) is True
        assert _signals_sent(fake_kill) == (), "os.kill reached on win32; signal 0 is CTRL_C_EVENT there"


class TestTerminateHardKillSignal:
    def test_escalation_resolves_a_signal_when_sigkill_is_absent(self, monkeypatch):
        """``signal.SIGKILL`` does not exist on Windows.

        Referencing it raises ``AttributeError``, and ``terminate`` wraps the
        call in ``contextlib.suppress(ProcessLookupError)``, which does not
        catch that -- so the branch that exists for a proxy ignoring SIGTERM
        crashed instead of hard-killing it.  ``os.kill`` with any signal other
        than 0 or 1 routes to ``TerminateProcess`` on Windows, so SIGTERM is a
        real kill there.
        """
        fake_kill: Final = Mock()
        monkeypatch.setattr(process_module.os, "kill", fake_kill)
        monkeypatch.setattr(process_module, "is_running", lambda pid: True)
        monkeypatch.setattr(process_module.time, "sleep", lambda seconds: None)
        monkeypatch.delattr(process_module.signal, "SIGKILL", raising=False)

        terminate(4242, grace_period=0.0)

        assert _signals_sent(fake_kill) == (signal.SIGTERM, signal.SIGTERM)

    @pytest.mark.skipif(
        not hasattr(signal, "SIGKILL"),
        reason="POSIX escalation path; SIGKILL is absent on this platform",
    )
    def test_escalation_still_uses_sigkill_where_it_exists(self, monkeypatch):
        fake_kill: Final = Mock()
        monkeypatch.setattr(process_module.os, "kill", fake_kill)
        monkeypatch.setattr(process_module, "is_running", lambda pid: True)
        monkeypatch.setattr(process_module.time, "sleep", lambda seconds: None)

        terminate(4242, grace_period=0.0)

        assert _signals_sent(fake_kill) == (signal.SIGTERM, signal.SIGKILL)

    @pytest.mark.parametrize(
        "raised",
        (
            PermissionError(5, "Access is denied"),
            OSError(22, "Invalid argument"),
            ProcessLookupError(3, "No such process"),
        ),
        ids=("permission-denied", "oserror", "already-gone"),
    )
    def test_a_kill_that_cannot_land_does_not_escape(self, monkeypatch, raised):
        """A process that exited between the probe and the kill must not crash ``down``.

        On Windows ``os.kill`` routes to ``TerminateProcess``, and a process
        that has already exited answers ``ERROR_ACCESS_DENIED`` -- so the call
        raises ``PermissionError``, not ``ProcessLookupError``.  Measured with
        the real CLI on Windows 11: ``lite autoroute down`` ended in
        ``PermissionError: [WinError 5]`` out of ``terminate``.
        ``litellm/proxy/db/prisma_client.py`` already tolerates all three for
        this same "already dead or inaccessible" case.
        """

        def fake_kill(pid: int, sig: int) -> None:
            raise raised

        monkeypatch.setattr(process_module.os, "kill", fake_kill)
        monkeypatch.setattr(process_module, "is_running", lambda pid: True)
        monkeypatch.setattr(process_module.time, "sleep", lambda seconds: None)

        terminate(4242, grace_period=0.0)


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
