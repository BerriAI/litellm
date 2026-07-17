import json
import stat
from typing import Optional

import yaml
from click.testing import CliRunner

from litellm.proxy.client.cli.commands.autoroute import commands as commands_module
from litellm.proxy.client.cli.commands.autoroute import process as process_module
from litellm.proxy.client.cli.commands.autoroute.commands import down, up
from litellm.proxy.client.cli.commands.autoroute.process import PidRecord, ProcessLaunchError, write_pid_record
from litellm.proxy.client.cli.commands.up import BackupRecord as ClaudeBackupRecord
from litellm.proxy.client.cli.commands.up import write_backup


class FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        return self.returncode


def _patch_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "proxy.log"
    claude_settings_path = tmp_path / "claude_settings.json"
    backup_path = tmp_path / "backup.json"
    pid_record_path = tmp_path / "pid.json"

    monkeypatch.setattr(commands_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(commands_module, "LOG_PATH", log_path)
    monkeypatch.setattr(commands_module, "CLAUDE_SETTINGS_PATH", claude_settings_path)
    monkeypatch.setattr(commands_module, "AUTOROUTE_BACKUP_PATH", backup_path)
    monkeypatch.setattr(process_module, "PID_RECORD_PATH", pid_record_path)

    return config_path, log_path, claude_settings_path, backup_path, pid_record_path


def _silence_signal_handling(monkeypatch):
    monkeypatch.setattr(commands_module.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(commands_module.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(commands_module, "stream_log", lambda *a, **k: None)


class TestUpCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_refuses_when_never_configured(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)

        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert "lite autoroute configure" in result.output

    def test_surfaces_clean_error_on_empty_config_file(self, monkeypatch, tmp_path):
        """A `configure` killed between secure_create's O_TRUNC and the write completing leaves an
        empty config.yaml on disk -- yaml.safe_load(empty) returns None, and validating None as the
        generated-config model raises a raw pydantic.ValidationError if uncaught."""
        config_path, _log_path, _settings_path, _backup_path, _pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text("")

        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "lite autoroute configure" in result.output

    def test_refuses_with_actionable_error_when_proxy_runtime_missing(self, monkeypatch, tmp_path):
        """`up` launches a real litellm proxy, which the thin `litellm[cli]` install cannot run.
        It must fail fast with an actionable message pointing at the proxy install, before it ever
        tries to launch the doomed subprocess (which would otherwise die with a bare ImportError)."""
        config_path, _log_path, _settings_path, _backup_path, _pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        monkeypatch.setattr(commands_module, "missing_proxy_runtime_modules", lambda: ("fastapi", "websockets"))

        def _fail_if_launched(*args, **kwargs):
            raise AssertionError("launch_proxy must not run when the proxy runtime is missing")

        monkeypatch.setattr(commands_module, "launch_proxy", _fail_if_launched)

        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert "fastapi, websockets" in result.output
        assert "litellm[proxy]" in result.output

    def test_refuses_when_pid_record_exists_and_process_still_running(self, monkeypatch, tmp_path):
        config_path, _log_path, _settings_path, _backup_path, pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        write_pid_record(
            PidRecord(pid=123, port=4000, config_path=str(config_path), log_path="/tmp/proxy.log"), pid_record_path
        )
        monkeypatch.setattr(commands_module, "is_running", lambda pid: True)

        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert "already running" in result.output
        assert "lite autoroute down" in result.output
        assert config_path.read_text() == yaml.safe_dump({"model_list": []})

    def test_refuses_when_backup_exists_after_an_unclean_crash(self, monkeypatch, tmp_path):
        """A prior `up` that was SIGKILL'd leaves no live pid but does leave a stale backup file.

        Without this guard, a fresh `up` would overwrite that backup with the currently-patched
        (not original) Claude settings, so `down`/Ctrl-C would restore the wrong content forever.
        """
        config_path, _log_path, claude_settings_path, backup_path, _pid_record_path = _patch_paths(
            monkeypatch, tmp_path
        )
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        claude_settings_path.write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "stale-patched-token"}}))
        write_backup(ClaudeBackupRecord(existed=True, content={"theme": "dark"}), backup_path)

        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert "already exists" in result.output
        assert "lite autoroute down" in result.output
        assert json.loads(backup_path.read_text())["content"] == {"theme": "dark"}

    def test_happy_path_patches_settings_then_restores_everything_on_stop(self, monkeypatch, tmp_path):
        config_path, log_path, claude_settings_path, backup_path, pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        original_settings = {"theme": "dark"}
        claude_settings_path.write_text(json.dumps(original_settings))
        _silence_signal_handling(monkeypatch)

        fake_process = FakeProcess(pid=99999)
        terminate_calls = []
        monkeypatch.setattr(commands_module, "launch_proxy", lambda *a, **k: fake_process)
        monkeypatch.setattr(commands_module, "poll_liveliness", lambda *a, **k: None)
        monkeypatch.setattr(commands_module, "allocate_free_port", lambda: 54321)
        monkeypatch.setattr(commands_module, "terminate", lambda pid, **k: terminate_calls.append(pid))
        captured = {}

        def fake_wait(self, timeout=None):
            captured["settings"] = json.loads(claude_settings_path.read_text())
            captured["backup_existed"] = backup_path.exists()
            captured["settings_mode"] = stat.S_IMODE(claude_settings_path.stat().st_mode)
            return True

        monkeypatch.setattr("threading.Event.wait", fake_wait)

        result = self.runner.invoke(up)

        assert result.exit_code == 0, result.output
        assert captured["backup_existed"] is True
        assert captured["settings"]["theme"] == "dark"
        assert captured["settings"]["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
        assert captured["settings"]["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-1234"
        assert "apiKeyHelper" not in captured["settings"]
        assert captured["settings_mode"] == 0o600

        assert terminate_calls == [99999]
        assert not pid_record_path.exists()
        assert not backup_path.exists()
        assert json.loads(claude_settings_path.read_text()) == original_settings

        written_config = yaml.safe_load(config_path.read_text())
        assert written_config["general_settings"]["master_key"] == "sk-1234"
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
        assert f"Config: {config_path}" in result.output
        assert f"Log: {log_path}" in result.output
        assert "Port: 54321" in result.output
        assert "Master key: sk-1234" in result.output

    def test_passes_debug_flags_to_proxy(self, monkeypatch, tmp_path):
        config_path, _log_path, claude_settings_path, _backup_path, _pid_record_path = _patch_paths(
            monkeypatch, tmp_path
        )
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        claude_settings_path.write_text(json.dumps({"theme": "dark"}))
        _silence_signal_handling(monkeypatch)

        fake_process = FakeProcess(pid=99999)
        launch_kwargs = {}
        monkeypatch.setattr(
            commands_module,
            "launch_proxy",
            lambda *args, **kwargs: launch_kwargs.update(kwargs) or fake_process,
        )
        monkeypatch.setattr(commands_module, "poll_liveliness", lambda *a, **k: None)
        monkeypatch.setattr(commands_module, "allocate_free_port", lambda: 54321)
        monkeypatch.setattr(commands_module, "terminate", lambda pid, **k: None)
        monkeypatch.setattr("threading.Event.wait", lambda self, timeout=None: True)

        result = self.runner.invoke(up, ["--debug", "--detailed-debug"])

        assert result.exit_code == 0, result.output
        assert launch_kwargs == {"debug": True, "detailed_debug": True}

    def test_teardown_reports_clean_error_when_backup_is_corrupt(self, monkeypatch, tmp_path):
        """A corrupt backup at teardown time (e.g. a concurrent process wrote garbage to it) must
        not crash the whole command -- _restore_once in up.py handles the identical case in
        lite up the same way, echoing the error instead of propagating it."""
        config_path, _log_path, claude_settings_path, backup_path, pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        claude_settings_path.write_text(json.dumps({"theme": "dark"}))
        _silence_signal_handling(monkeypatch)

        fake_process = FakeProcess(pid=11111)
        monkeypatch.setattr(commands_module, "launch_proxy", lambda *a, **k: fake_process)
        monkeypatch.setattr(commands_module, "poll_liveliness", lambda *a, **k: None)
        monkeypatch.setattr(commands_module, "allocate_free_port", lambda: 65432)
        monkeypatch.setattr(commands_module, "terminate", lambda pid, **k: None)

        def fake_wait(self, timeout=None):
            backup_path.write_text("not json at all {{{")
            return True

        monkeypatch.setattr("threading.Event.wait", fake_wait)

        result = self.runner.invoke(up)

        assert result.exit_code == 0, result.output
        assert "invalid or unexpected JSON" in result.output
        assert not pid_record_path.exists()

    def test_surfaces_clean_error_and_cleans_up_when_health_check_fails(self, monkeypatch, tmp_path):
        config_path, _log_path, claude_settings_path, backup_path, pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        original_settings = {"theme": "dark"}
        claude_settings_path.write_text(json.dumps(original_settings))

        fake_process = FakeProcess(pid=555)
        terminate_calls = []

        def _raise_launch_error(*args, **kwargs):
            raise ProcessLaunchError("boom: proxy never became healthy")

        monkeypatch.setattr(commands_module, "launch_proxy", lambda *a, **k: fake_process)
        monkeypatch.setattr(commands_module, "poll_liveliness", _raise_launch_error)
        monkeypatch.setattr(commands_module, "allocate_free_port", lambda: 12345)
        monkeypatch.setattr(commands_module, "terminate", lambda pid, **k: terminate_calls.append(pid))
        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert "boom" in result.output
        assert terminate_calls == [555]
        assert not pid_record_path.exists()
        assert not backup_path.exists()
        assert json.loads(claude_settings_path.read_text()) == original_settings

    def test_terminates_ephemeral_proxy_when_claude_settings_is_corrupt(self, monkeypatch, tmp_path):
        """The health check can pass and the proxy can come up fine, but if
        ~/.claude/settings.json turns out to be corrupt, the just-started proxy must not be left
        running with no pid record -- exactly the leak `lite autoroute down` exists to clean up."""
        config_path, _log_path, claude_settings_path, backup_path, pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        claude_settings_path.write_text("not json at all {{{")

        fake_process = FakeProcess(pid=777)
        terminate_calls = []
        monkeypatch.setattr(commands_module, "launch_proxy", lambda *a, **k: fake_process)
        monkeypatch.setattr(commands_module, "poll_liveliness", lambda *a, **k: None)
        monkeypatch.setattr(commands_module, "allocate_free_port", lambda: 23456)
        monkeypatch.setattr(commands_module, "terminate", lambda pid, **k: terminate_calls.append(pid))
        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert "invalid JSON" in result.output
        assert terminate_calls == [777]
        assert not pid_record_path.exists()
        assert not backup_path.exists()


class TestDownCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_restores_settings_and_terminates_when_process_still_running(self, monkeypatch, tmp_path):
        _config_path, _log_path, claude_settings_path, backup_path, pid_record_path = _patch_paths(
            monkeypatch, tmp_path
        )
        original_settings = {"theme": "dark"}
        write_backup(ClaudeBackupRecord(existed=True, content=original_settings), backup_path)
        claude_settings_path.write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "fixed-master-key"}}))
        write_pid_record(PidRecord(pid=777, port=1234, config_path="c", log_path="l"), pid_record_path)

        terminate_calls = []
        monkeypatch.setattr(commands_module, "is_running", lambda pid: True)
        monkeypatch.setattr(commands_module, "terminate", lambda pid, **k: terminate_calls.append(pid))

        result = self.runner.invoke(down)

        assert result.exit_code == 0, result.output
        assert "Stopped leftover ephemeral proxy" in result.output
        assert "Restored" in result.output
        assert terminate_calls == [777]
        assert not pid_record_path.exists()
        assert not backup_path.exists()
        assert json.loads(claude_settings_path.read_text()) == original_settings

    def test_is_a_clean_no_op_when_nothing_is_running_and_no_backup_exists(self, monkeypatch, tmp_path):
        _config_path, _log_path, claude_settings_path, _backup_path, _pid_record_path = _patch_paths(
            monkeypatch, tmp_path
        )

        result = self.runner.invoke(down)

        assert result.exit_code == 0, result.output
        assert "Nothing to restore." in result.output
        assert not claude_settings_path.exists()

    def test_clears_a_corrupt_pid_record_and_still_restores_settings(self, monkeypatch, tmp_path):
        """down is specifically the crash-recovery path -- a pid file truncated by a mid-write
        crash must not block it from clearing the record and restoring Claude settings anyway."""
        _config_path, _log_path, claude_settings_path, backup_path, pid_record_path = _patch_paths(
            monkeypatch, tmp_path
        )
        pid_record_path.parent.mkdir(parents=True, exist_ok=True)
        pid_record_path.write_text("not json at all {{{")
        original_settings = {"theme": "dark"}
        write_backup(ClaudeBackupRecord(existed=True, content=original_settings), backup_path)
        claude_settings_path.write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "fixed-master-key"}}))

        result = self.runner.invoke(down)

        assert result.exit_code == 0, result.output
        assert "invalid or unexpected JSON" in result.output
        assert "Restored" in result.output
        assert not pid_record_path.exists()
        assert not backup_path.exists()
        assert json.loads(claude_settings_path.read_text()) == original_settings

    def test_surfaces_clean_error_when_backup_is_corrupt(self, monkeypatch, tmp_path):
        _config_path, _log_path, _claude_settings_path, backup_path, _pid_record_path = _patch_paths(
            monkeypatch, tmp_path
        )
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text("not json at all {{{")

        result = self.runner.invoke(down)

        assert result.exit_code != 0
        assert "invalid or unexpected JSON" in result.output
