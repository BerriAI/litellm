import json
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
        monkeypatch.setattr(commands_module.secrets, "token_urlsafe", lambda n: "fixed-master-key")

        captured = {}

        def fake_wait(self, timeout=None):
            captured["settings"] = json.loads(claude_settings_path.read_text())
            captured["backup_existed"] = backup_path.exists()
            return True

        monkeypatch.setattr("threading.Event.wait", fake_wait)

        result = self.runner.invoke(up)

        assert result.exit_code == 0, result.output
        assert captured["backup_existed"] is True
        assert captured["settings"]["theme"] == "dark"
        assert captured["settings"]["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
        assert captured["settings"]["env"]["ANTHROPIC_AUTH_TOKEN"] == "fixed-master-key"
        assert "apiKeyHelper" not in captured["settings"]

        assert terminate_calls == [99999]
        assert not pid_record_path.exists()
        assert not backup_path.exists()
        assert json.loads(claude_settings_path.read_text()) == original_settings

        written_config = yaml.safe_load(config_path.read_text())
        assert written_config["general_settings"]["master_key"] == "fixed-master-key"

    def test_surfaces_clean_error_and_cleans_up_when_health_check_fails(self, monkeypatch, tmp_path):
        config_path, _log_path, claude_settings_path, backup_path, pid_record_path = _patch_paths(monkeypatch, tmp_path)
        config_path.write_text(yaml.safe_dump({"model_list": []}))
        original_settings = {"theme": "dark"}
        claude_settings_path.write_text(json.dumps(original_settings))

        fake_process = FakeProcess(pid=555)

        def _raise_launch_error(*args, **kwargs):
            raise ProcessLaunchError("boom: proxy never became healthy")

        monkeypatch.setattr(commands_module, "launch_proxy", lambda *a, **k: fake_process)
        monkeypatch.setattr(commands_module, "poll_liveliness", _raise_launch_error)
        monkeypatch.setattr(commands_module, "allocate_free_port", lambda: 12345)
        monkeypatch.setattr(commands_module.secrets, "token_urlsafe", lambda n: "fixed-master-key")

        result = self.runner.invoke(up)

        assert result.exit_code != 0
        assert "boom" in result.output
        assert not pid_record_path.exists()
        assert not backup_path.exists()
        assert json.loads(claude_settings_path.read_text()) == original_settings


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
