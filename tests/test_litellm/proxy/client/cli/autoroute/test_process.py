import os
import socket
from typing import Optional
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement

from litellm.proxy.client.cli.commands.autoroute import process as process_module
from litellm.proxy.client.cli.commands.autoroute.process import (
    CommandResult,
    PidRecord,
    ProcessLaunchError,
    ProxyRuntimeInstallError,
    UpError,
    clear_pid_record,
    find_uv,
    install_proxy_runtime,
    is_port_available,
    is_running,
    launch_proxy,
    missing_proxy_runtime_modules,
    poll_liveliness,
    proxy_extra_requirements,
    read_pid_record,
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
        def fake_kill(pid: int, sig: int) -> None:
            raise PermissionError("not permitted to signal this pid")

        monkeypatch.setattr(process_module.os, "kill", fake_kill)

        assert is_running(999) is True


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


class TestProxyExtraRequirements:
    def test_returns_pinned_proxy_deps_without_litellm_or_markers(self):
        """`up` installs these on demand, so they must be litellm's real proxy dependencies with
        their version pins intact, must never include litellm itself (that would swap the running
        install), and must carry no environment markers (they are dropped so the win32-guarded
        entries still install on the macOS/Linux hosts the CLI supports)."""
        reqs = proxy_extra_requirements()

        names = {Requirement(r).name for r in reqs}
        assert {"fastapi", "uvicorn", "apscheduler", "backoff", "orjson", "websockets"} <= names
        assert "litellm" not in names
        assert all(str(Requirement(r).specifier) for r in reqs)
        assert all(";" not in r for r in reqs)


class TestFindUv:
    def test_prefers_uv_on_path(self, monkeypatch):
        monkeypatch.setattr(process_module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        assert find_uv() == "/usr/bin/uv"

    def test_falls_back_to_the_installers_local_bin_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(process_module.shutil, "which", lambda name: None)
        uv_path = tmp_path / ".local" / "bin" / "uv"
        uv_path.parent.mkdir(parents=True)
        uv_path.write_text("#!/bin/sh\n")
        uv_path.chmod(0o755)
        monkeypatch.setattr(process_module.Path, "home", lambda: tmp_path)

        assert find_uv() == str(uv_path)

    def test_returns_none_when_uv_absent_everywhere(self, monkeypatch, tmp_path):
        monkeypatch.setattr(process_module.shutil, "which", lambda name: None)
        monkeypatch.setattr(process_module.Path, "home", lambda: tmp_path)

        assert find_uv() is None


class TestRunUvInstall:
    def test_targets_the_current_interpreter_and_never_reinstalls_litellm(self, monkeypatch):
        """The install must land in the interpreter running `lite` (so the proxy subprocess sees it)
        and pass only the extra's deps -- passing `litellm` would let uv resolve a different version
        over the installed (possibly branch/QA) one."""
        captured = {}

        class _Completed:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def _fake_run(argv, capture_output, text):
            captured["argv"] = argv
            return _Completed()

        monkeypatch.setattr(process_module.subprocess, "run", _fake_run)

        result = process_module._run_uv_install("/bin/uv", ("fastapi>=1", "uvicorn>=1"))

        assert result == CommandResult(returncode=0, output="ok")
        assert captured["argv"] == [
            "/bin/uv",
            "pip",
            "install",
            "--python",
            process_module.sys.executable,
            "fastapi>=1",
            "uvicorn>=1",
        ]
        assert "litellm" not in captured["argv"][5:]

    def test_prefers_stderr_when_the_command_fails(self, monkeypatch):
        class _Completed:
            returncode = 1
            stdout = "some stdout"
            stderr = "the real error"

        monkeypatch.setattr(process_module.subprocess, "run", lambda argv, capture_output, text: _Completed())

        assert process_module._run_uv_install("/bin/uv", ("fastapi>=1",)) == CommandResult(
            returncode=1, output="the real error"
        )


class TestInstallProxyRuntime:
    def test_raises_when_uv_is_missing(self):
        with pytest.raises(ProxyRuntimeInstallError) as excinfo:
            install_proxy_runtime(find_uv_bin=lambda: None)

        assert "litellm[proxy]" in str(excinfo.value)

    def test_raises_when_no_proxy_requirements_are_found(self):
        with pytest.raises(ProxyRuntimeInstallError) as excinfo:
            install_proxy_runtime(find_uv_bin=lambda: "/bin/uv", requirements=lambda: ())

        assert "metadata" in str(excinfo.value)

    def test_raises_with_the_command_output_when_install_fails(self):
        def _run(uv, reqs):
            return CommandResult(returncode=1, output="network is unreachable")

        with pytest.raises(ProxyRuntimeInstallError) as excinfo:
            install_proxy_runtime(
                find_uv_bin=lambda: "/bin/uv",
                requirements=lambda: ("fastapi>=1",),
                run_install=_run,
            )

        assert "network is unreachable" in str(excinfo.value)

    def test_installs_the_discovered_requirements_with_the_found_uv(self):
        calls = []

        def _run(uv, reqs):
            calls.append((uv, reqs))
            return CommandResult(returncode=0, output="")

        install_proxy_runtime(
            find_uv_bin=lambda: "/bin/uv",
            requirements=lambda: ("fastapi>=1", "uvicorn>=1"),
            run_install=_run,
        )

        assert calls == [("/bin/uv", ("fastapi>=1", "uvicorn>=1"))]
