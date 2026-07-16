import os
import socket
from typing import Optional
from unittest.mock import patch

import pytest

from litellm.proxy.client.cli.commands.autoroute import process as process_module
from litellm.proxy.client.cli.commands.autoroute.process import (
    PidRecord,
    ProcessLaunchError,
    UpError,
    allocate_free_port,
    clear_pid_record,
    is_running,
    launch_proxy,
    poll_liveliness,
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


def test_allocate_free_port_returns_a_bindable_port():
    port = allocate_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


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
