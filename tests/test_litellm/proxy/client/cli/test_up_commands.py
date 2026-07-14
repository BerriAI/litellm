import json
import shutil
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli.commands import up as up_module
from litellm.proxy.client.cli.commands.agents import AgentRunError
from litellm.proxy.client.cli.commands.up import (
    BackupRecord,
    UpError,
    down,
    merge_claude_settings,
    read_backup,
    resolve_api_key_helper,
    restore_claude_settings,
    up,
    write_backup,
)

UP_MODULE = "litellm.proxy.client.cli.commands.up"


def _patch_paths(monkeypatch, tmp_path):
    settings_path = tmp_path / "claude_settings.json"
    backup_path = tmp_path / "backup.json"
    monkeypatch.setattr(up_module, "CLAUDE_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(up_module, "BACKUP_PATH", backup_path)
    return settings_path, backup_path


class TestMergeClaudeSettings:
    def test_preserves_unrelated_top_level_keys(self):
        merged = merge_claude_settings({"theme": "dark"}, "http://localhost:4000", "helper")
        assert merged["theme"] == "dark"

    def test_preserves_unrelated_env_keys(self):
        settings = {"env": {"SOME_OTHER_VAR": "value"}}
        merged = merge_claude_settings(settings, "http://localhost:4000", "helper")
        assert merged["env"]["SOME_OTHER_VAR"] == "value"

    def test_overrides_base_url_and_helper(self):
        settings = {
            "env": {"ANTHROPIC_BASE_URL": "https://old.example.com"},
            "apiKeyHelper": "old-helper",
        }
        merged = merge_claude_settings(settings, "http://localhost:4000/", "new-helper")
        assert merged["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert merged["apiKeyHelper"] == "new-helper"

    def test_drops_stray_api_key(self):
        settings = {"env": {"ANTHROPIC_API_KEY": "leaked-key"}}
        merged = merge_claude_settings(settings, "http://localhost:4000", "helper")
        assert "ANTHROPIC_API_KEY" not in merged["env"]

    def test_works_from_empty_settings(self):
        merged = merge_claude_settings({}, "http://localhost:4000", "helper")
        assert merged["env"] == {"ANTHROPIC_BASE_URL": "http://localhost:4000"}
        assert merged["apiKeyHelper"] == "helper"

    def test_does_not_mutate_input(self):
        settings = {"env": {"FOO": "bar"}}
        merge_claude_settings(settings, "http://localhost:4000", "helper")
        assert settings == {"env": {"FOO": "bar"}}


class TestBackupRoundTrip:
    def test_restores_original_content_when_file_existed(self, monkeypatch, tmp_path):
        settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        original = {"apiKeyHelper": "old-helper", "theme": "dark"}
        settings_path.write_text(json.dumps(original))

        write_backup(BackupRecord(existed=True, content=original))
        settings_path.write_text(json.dumps({"apiKeyHelper": "lite-helper"}))

        restored = restore_claude_settings()

        assert restored is not None
        assert restored.existed is True
        assert json.loads(settings_path.read_text()) == original
        assert not backup_path.exists()

    def test_deletes_settings_file_when_it_did_not_exist_before(self, monkeypatch, tmp_path):
        settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        write_backup(BackupRecord(existed=False, content=None))
        settings_path.write_text(json.dumps({"apiKeyHelper": "lite-helper"}))

        restored = restore_claude_settings()

        assert restored is not None
        assert restored.existed is False
        assert not settings_path.exists()
        assert not backup_path.exists()

    def test_no_backup_is_a_no_op_returning_none(self, monkeypatch, tmp_path):
        settings_path, _backup_path = _patch_paths(monkeypatch, tmp_path)
        assert restore_claude_settings() is None
        assert not settings_path.exists()

    def test_read_backup_round_trips_write_backup(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)
        write_backup(BackupRecord(existed=True, content={"a": 1}))
        assert read_backup() == BackupRecord(existed=True, content={"a": 1})

    def test_read_backup_missing_file_returns_none(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)
        assert read_backup() is None

    def test_backup_file_always_removed_after_restore(self, monkeypatch, tmp_path):
        _settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        write_backup(BackupRecord(existed=False, content=None))
        assert backup_path.exists()

        restore_claude_settings()

        assert not backup_path.exists()


class TestResolveApiKeyHelper:
    def test_returns_helper_command_when_lite_found(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/lite")
        helper = resolve_api_key_helper()
        assert helper == "/usr/local/bin/lite auth print-token"

    def test_raises_when_lite_not_on_path(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(UpError, match="Could not find `lite`"):
            resolve_api_key_helper()


class TestUpCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_refuses_double_start_without_touching_settings_file(self, monkeypatch, tmp_path):
        settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        existing_backup = {"existed": False, "content": None}
        backup_path.write_text(json.dumps(existing_backup))

        with (
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh"}),
            patch(f"{UP_MODULE}.is_cli_token_fresh", return_value=True),
            patch(f"{UP_MODULE}.resolve_api_key", return_value="sk-fresh"),
            patch(f"{UP_MODULE}.verify_proxy_key"),
        ):
            result = self.runner.invoke(up, obj={"base_url": "http://localhost:4000"})

        assert result.exit_code != 0
        assert "already" in result.output
        assert "lite down" in result.output
        assert not settings_path.exists()
        assert json.loads(backup_path.read_text()) == existing_backup

    def test_no_fresh_login_non_interactive_fails_cleanly(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        with patch(f"{UP_MODULE}.load_token", return_value=None):
            result = self.runner.invoke(up, obj={"base_url": "http://localhost:4000"})

        assert result.exit_code != 0
        assert "lite login" in result.output

    def test_unreachable_proxy_fails_cleanly(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)

        with (
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh"}),
            patch(f"{UP_MODULE}.is_cli_token_fresh", return_value=True),
            patch(f"{UP_MODULE}.resolve_api_key", return_value="sk-fresh"),
            patch(
                f"{UP_MODULE}.verify_proxy_key",
                side_effect=AgentRunError("Could not reach the LiteLLM proxy at http://localhost:4000"),
            ),
        ):
            result = self.runner.invoke(up, obj={"base_url": "http://localhost:4000"})

        assert result.exit_code != 0
        assert "Could not reach the LiteLLM proxy" in result.output

    def test_happy_path_writes_settings_and_backup_then_restores_on_stop(self, monkeypatch, tmp_path):
        settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        original = {"theme": "dark"}
        settings_path.write_text(json.dumps(original))

        captured = {}

        def fake_wait(self, timeout=None):
            captured["settings"] = json.loads(settings_path.read_text())
            captured["backup_existed"] = backup_path.exists()
            return True

        with (
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh"}),
            patch(f"{UP_MODULE}.is_cli_token_fresh", return_value=True),
            patch(f"{UP_MODULE}.resolve_api_key", return_value="sk-fresh"),
            patch(f"{UP_MODULE}.verify_proxy_key"),
            patch(
                f"{UP_MODULE}.resolve_api_key_helper",
                return_value="/usr/local/bin/lite auth print-token",
            ),
            patch(f"{UP_MODULE}.signal.signal"),
            patch(f"{UP_MODULE}.atexit.register"),
            patch("threading.Event.wait", new=fake_wait),
        ):
            result = self.runner.invoke(up, obj={"base_url": "http://localhost:4000"})

        assert result.exit_code == 0, result.output
        assert captured["backup_existed"] is True
        assert captured["settings"]["theme"] == "dark"
        assert captured["settings"]["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
        assert captured["settings"]["apiKeyHelper"] == "/usr/local/bin/lite auth print-token"
        assert json.loads(settings_path.read_text()) == original
        assert not backup_path.exists()


class TestDownCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_restores_when_backup_exists(self, monkeypatch, tmp_path):
        settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        original = {"apiKeyHelper": "old-helper"}
        write_backup(BackupRecord(existed=True, content=original))
        settings_path.write_text(json.dumps({"apiKeyHelper": "lite-helper"}))

        result = self.runner.invoke(down)

        assert result.exit_code == 0, result.output
        assert "Restored" in result.output
        assert json.loads(settings_path.read_text()) == original
        assert not backup_path.exists()

    def test_removes_settings_file_when_it_did_not_exist_before(self, monkeypatch, tmp_path):
        settings_path, _backup_path = _patch_paths(monkeypatch, tmp_path)
        write_backup(BackupRecord(existed=False, content=None))
        settings_path.write_text(json.dumps({"apiKeyHelper": "lite-helper"}))

        result = self.runner.invoke(down)

        assert result.exit_code == 0, result.output
        assert "Removed" in result.output
        assert not settings_path.exists()

    def test_prints_nothing_to_restore_when_no_backup(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)

        result = self.runner.invoke(down)

        assert result.exit_code == 0, result.output
        assert "Nothing to restore." in result.output
