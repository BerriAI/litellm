import json
import shutil
import stat
import sys
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli.commands import up as up_module
from litellm.proxy.client.cli.commands.agents import AgentRunError
from litellm.proxy.client.cli.commands.up import (
    BackupRecord,
    UpError,
    _ensure_fresh_login,
    down,
    load_json_or_empty,
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


class TestLoadJsonOrEmpty:
    def test_returns_empty_dict_when_file_does_not_exist(self, tmp_path):
        assert load_json_or_empty(tmp_path / "missing.json") == {}

    def test_returns_empty_dict_when_file_is_empty(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("")
        assert load_json_or_empty(path) == {}

    def test_returns_empty_dict_when_file_is_whitespace_only(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("   \n")
        assert load_json_or_empty(path) == {}

    def test_parses_real_content(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"theme": "dark"}))
        assert load_json_or_empty(path) == {"theme": "dark"}

    def test_raises_clean_error_on_invalid_json(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("not json at all {{{")
        with pytest.raises(UpError, match="invalid JSON"):
            load_json_or_empty(path)

    def test_raises_clean_error_on_non_object_root(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(UpError, match="invalid JSON"):
            load_json_or_empty(path)


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

    def test_recreates_claude_dir_if_it_was_deleted_while_up_was_running(self, monkeypatch, tmp_path):
        """If ~/.claude/ is removed while `lite up` holds it open, restoring must recreate the
        directory rather than crash with FileNotFoundError and strand the backup file, which
        would otherwise permanently break every future `lite down`."""
        claude_dir = tmp_path / "claude_dir"
        settings_path = claude_dir / "settings.json"
        backup_path = tmp_path / "backup.json"
        monkeypatch.setattr(up_module, "CLAUDE_SETTINGS_PATH", settings_path)
        monkeypatch.setattr(up_module, "BACKUP_PATH", backup_path)
        original = {"theme": "dark"}
        claude_dir.mkdir(parents=True)
        write_backup(BackupRecord(existed=True, content=original))
        shutil.rmtree(claude_dir)

        restored = restore_claude_settings()

        assert restored is not None
        assert json.loads(settings_path.read_text()) == original
        assert not backup_path.exists()

    def test_read_backup_round_trips_write_backup(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)
        write_backup(BackupRecord(existed=True, content={"a": 1}))
        assert read_backup() == BackupRecord(existed=True, content={"a": 1})

    def test_read_backup_missing_file_returns_none(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)
        assert read_backup() is None

    def test_read_backup_raises_clean_error_on_corrupt_content(self, monkeypatch, tmp_path):
        _settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text("not json at all {{{")

        with pytest.raises(UpError, match="invalid or unexpected JSON"):
            read_backup()

    def test_write_backup_restricts_permissions_for_a_new_file(self, monkeypatch, tmp_path):
        _settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        write_backup(BackupRecord(existed=True, content={"a": 1}))
        assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600

    def test_write_backup_restricts_permissions_of_a_preexisting_permissive_file(self, monkeypatch, tmp_path):
        _settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text("{}")
        backup_path.chmod(0o644)

        write_backup(BackupRecord(existed=True, content={"a": 1}))

        assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600

    def test_backup_file_always_removed_after_restore(self, monkeypatch, tmp_path):
        _settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        write_backup(BackupRecord(existed=False, content=None))
        assert backup_path.exists()

        restore_claude_settings()

        assert not backup_path.exists()


class TestResolveApiKeyHelper:
    def test_returns_helper_command_bound_to_the_selected_proxy(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/lite")
        helper = resolve_api_key_helper("http://localhost:4000")
        assert helper == "/usr/local/bin/lite auth print-token --base-url http://localhost:4000"

    def test_quotes_a_base_url_containing_shell_metacharacters(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/lite")
        helper = resolve_api_key_helper("http://example.com/path; rm -rf /")
        assert helper == "/usr/local/bin/lite auth print-token --base-url 'http://example.com/path; rm -rf /'"

    def test_raises_when_lite_not_on_path(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(UpError, match="Could not find `lite`"):
            resolve_api_key_helper("http://localhost:4000")


def _make_ctx(base_url):
    return click.Context(click.Command("test"), obj={"base_url": base_url})


class TestEnsureFreshLogin:
    """A token that is fresh but was issued for a *different* proxy must not be trusted: without
    this check, a user logged into proxy A who runs `up --base-url proxy-b` would silently get an
    apiKeyHelper wired up around proxy A's real token, which print-token would then hand to proxy B."""

    def test_reuses_a_fresh_token_issued_for_the_same_proxy(self, monkeypatch):
        monkeypatch.setattr(up_module, "load_token", lambda: {"key": "sk-a", "base_url": "http://proxy-a:4000"})
        monkeypatch.setattr(up_module, "is_cli_token_fresh", lambda token_data: True)
        login_calls = []
        monkeypatch.setattr(up_module, "login", lambda ctx: login_calls.append(ctx))

        _ensure_fresh_login(_make_ctx("http://proxy-a:4000"))

        assert login_calls == []

    def test_forces_a_fresh_login_when_the_cached_token_is_for_a_different_proxy(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: True)
        tokens = iter(
            [
                {"key": "sk-a", "base_url": "http://proxy-a:4000"},
                {"key": "sk-b", "base_url": "http://proxy-b:4000"},
            ]
        )
        monkeypatch.setattr(up_module, "load_token", lambda: next(tokens))
        monkeypatch.setattr(up_module, "is_cli_token_fresh", lambda token_data: True)
        login_calls = []

        @click.pass_context
        def fake_login(ctx):
            login_calls.append(ctx.obj["base_url"])

        monkeypatch.setattr(up_module, "login", fake_login)

        _ensure_fresh_login(_make_ctx("http://proxy-b:4000"))

        assert login_calls == ["http://proxy-b:4000"]

    def test_fails_cleanly_non_interactively_when_only_a_different_proxys_token_is_cached(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(up_module, "load_token", lambda: {"key": "sk-a", "base_url": "http://proxy-a:4000"})
        monkeypatch.setattr(up_module, "is_cli_token_fresh", lambda token_data: True)

        with pytest.raises(UpError, match="lite login"):
            _ensure_fresh_login(_make_ctx("http://proxy-b:4000"))


class TestUpCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_refuses_double_start_without_touching_settings_file(self, monkeypatch, tmp_path):
        settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        existing_backup = {"existed": False, "content": None}
        backup_path.write_text(json.dumps(existing_backup))

        with (
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh", "base_url": "http://localhost:4000"}),
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
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh", "base_url": "http://localhost:4000"}),
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
            captured["settings_mode"] = stat.S_IMODE(settings_path.stat().st_mode)
            return True

        with (
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh", "base_url": "http://localhost:4000"}),
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
        assert captured["settings_mode"] == 0o600
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

    def test_surfaces_clean_error_on_a_corrupt_backup_file(self, monkeypatch, tmp_path):
        _settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text("not json at all {{{")

        result = self.runner.invoke(down)

        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "invalid or unexpected JSON" in result.output
