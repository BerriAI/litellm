import json
import shutil
import stat
import sys
import time
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli.commands import up as up_module
from litellm.proxy.client.cli.commands.agents import AgentRunError
from litellm.proxy.client.cli.commands.claude_settings import ClaudeSettingsError
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
AUTH_MODULE = "litellm.proxy.client.cli.commands.auth"


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
        with pytest.raises(ClaudeSettingsError, match="invalid JSON"):
            load_json_or_empty(path)

    def test_raises_clean_error_on_non_object_root(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ClaudeSettingsError, match="invalid JSON"):
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
        assert helper == "/usr/local/bin/lite --base-url http://localhost:4000 auth print-token"

    def test_quotes_a_base_url_containing_shell_metacharacters(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/lite")
        helper = resolve_api_key_helper("http://example.com/path; rm -rf /")
        assert helper == "/usr/local/bin/lite --base-url 'http://example.com/path; rm -rf /' auth print-token"

    def test_raises_when_lite_not_on_path(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(ClaudeSettingsError, match="Could not find `lite`"):
            resolve_api_key_helper("http://localhost:4000")


def _make_ctx(base_url):
    return click.Context(click.Command("test"), obj={"base_url": base_url})


def _make_group_ctx(base_url, api_key):
    """The context the `lite` group hands down after it already read the token file once."""
    return click.Context(
        click.Command("test"), obj={"base_url": base_url, "api_key": api_key, "api_key_from_token_file": True}
    )


def _pkce_record(base_url, seconds_left):
    return {
        "key": "sk-pkce",
        "base_url": base_url,
        "expires_at": time.time() + seconds_left,
        "refresh_token": "llm_srefresh_old",
    }


class _FakeTokenStore:
    """Stands in for the token file: `get_stored_api_key` answers per proxy and may rotate the
    record that `load_token` returns afterwards, exactly the side effect a refresh has on disk."""

    def __init__(self, monkeypatch, record, keys_by_base_url, rotated_to=None):
        self.record = record
        self.keys_by_base_url = keys_by_base_url
        self.rotated_to = rotated_to
        self.key_requests = []
        monkeypatch.setattr(up_module, "load_token", lambda **_: self.record)
        monkeypatch.setattr(up_module, "get_stored_api_key", self.get_stored_api_key)

    def get_stored_api_key(self, expected_base_url=None, **_):
        self.key_requests.append(expected_base_url)
        key = self.keys_by_base_url.get(expected_base_url)
        if key is not None and self.rotated_to is not None:
            self.record = self.rotated_to
        return key

    def log_in(self, record, key):
        self.record = record
        self.keys_by_base_url = {record["base_url"]: key}


def _capture_login(monkeypatch, on_login=lambda: None):
    login_calls = []

    @click.pass_context
    def fake_login(ctx, pkce=False):
        login_calls.append((ctx.obj["base_url"], pkce))
        on_login()

    monkeypatch.setattr(up_module, "login", fake_login)
    return login_calls


class TestEnsureFreshLogin:
    """A token that is fresh but was issued for a *different* proxy must not be trusted: without
    this check, a user logged into proxy A who runs `up --base-url proxy-b` would silently get an
    apiKeyHelper wired up around proxy A's real token, which print-token would then hand to proxy B."""

    def test_reuses_a_fresh_token_issued_for_the_same_proxy(self, monkeypatch):
        _FakeTokenStore(
            monkeypatch, {"key": "sk-a", "base_url": "http://proxy-a:4000"}, {"http://proxy-a:4000": "sk-a"}
        )
        monkeypatch.setattr(up_module, "is_cli_token_fresh", lambda token_data: True)
        login_calls = _capture_login(monkeypatch)

        _ensure_fresh_login(_make_ctx("http://proxy-a:4000"))

        assert login_calls == []

    def test_forces_a_fresh_login_when_the_cached_token_is_for_a_different_proxy(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: True)
        store = _FakeTokenStore(monkeypatch, {"key": "sk-a", "base_url": "http://proxy-a:4000"}, {})
        monkeypatch.setattr(up_module, "is_cli_token_fresh", lambda token_data: True)
        login_calls = _capture_login(
            monkeypatch, on_login=lambda: store.log_in({"key": "sk-b", "base_url": "http://proxy-b:4000"}, "sk-b")
        )

        _ensure_fresh_login(_make_ctx("http://proxy-b:4000"))

        assert login_calls == [("http://proxy-b:4000", False)]
        assert store.key_requests == ["http://proxy-b:4000", "http://proxy-b:4000"]

    def test_forces_a_fresh_login_when_the_cached_token_has_no_readable_key(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: True)
        store = _FakeTokenStore(monkeypatch, {"base_url": "http://proxy-a:4000"}, {})
        monkeypatch.setattr(up_module, "is_cli_token_fresh", lambda token_data: True)
        login_calls = _capture_login(
            monkeypatch,
            on_login=lambda: store.log_in({"key": "sk-a", "base_url": "http://proxy-a:4000"}, "sk-a"),
        )

        _ensure_fresh_login(_make_ctx("http://proxy-a:4000"))

        assert login_calls == [("http://proxy-a:4000", False)]

    def test_fails_cleanly_non_interactively_when_only_a_different_proxys_token_is_cached(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: False)
        _FakeTokenStore(monkeypatch, {"key": "sk-a", "base_url": "http://proxy-a:4000"}, {})
        monkeypatch.setattr(up_module, "is_cli_token_fresh", lambda token_data: True)

        with pytest.raises(UpError, match="Run `lite login` first"):
            _ensure_fresh_login(_make_ctx("http://proxy-b:4000"))

    def test_trusts_a_pkce_credential_that_was_renewed_on_the_way_in(self, monkeypatch):
        """A --pkce key inside its freshness buffer is renewed by `get_stored_api_key`, so `lite up`
        must judge the rotated record rather than refuse the one it started with."""
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: False)
        store = _FakeTokenStore(
            monkeypatch,
            _pkce_record("http://proxy-a:4000", seconds_left=200),
            {"http://proxy-a:4000": "sk-pkce-renewed"},
            rotated_to=_pkce_record("http://proxy-a:4000", seconds_left=86_400),
        )
        login_calls = _capture_login(monkeypatch)

        _ensure_fresh_login(_make_ctx("http://proxy-a:4000"))

        assert login_calls == []
        assert store.key_requests == ["http://proxy-a:4000"]

    def test_logs_in_again_with_pkce_when_the_stale_credential_came_from_pkce(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: True)
        store = _FakeTokenStore(monkeypatch, _pkce_record("http://proxy-a:4000", seconds_left=-10), {})
        login_calls = _capture_login(
            monkeypatch,
            on_login=lambda: store.log_in(_pkce_record("http://proxy-a:4000", seconds_left=86_400), "sk-pkce-fresh"),
        )

        _ensure_fresh_login(_make_ctx("http://proxy-a:4000"))

        assert login_calls == [("http://proxy-a:4000", True)]

    def test_names_the_pkce_login_in_the_non_interactive_error_for_a_pkce_credential(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: False)
        _FakeTokenStore(monkeypatch, _pkce_record("http://proxy-a:4000", seconds_left=-10), {})

        with pytest.raises(UpError, match="Run `lite login --pkce` first"):
            _ensure_fresh_login(_make_ctx("http://proxy-a:4000"))

    def test_trusts_the_key_the_cli_group_already_resolved_instead_of_reading_the_token_file_again(
        self, monkeypatch
    ):
        """The `lite` group renews a --pkce key on the way in, so a second renewal here would hit
        the proxy twice and, once the refresh token is burned, print the refusal twice."""
        store = _FakeTokenStore(monkeypatch, _pkce_record("http://proxy-a:4000", seconds_left=86_400), {})
        login_calls = _capture_login(monkeypatch)

        _ensure_fresh_login(_make_group_ctx("http://proxy-a:4000", api_key="sk-pkce-renewed-by-the-group"))

        assert login_calls == []
        assert store.key_requests == []

    def test_fails_non_interactively_without_a_second_renewal_when_the_group_could_not_renew(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: False)
        store = _FakeTokenStore(
            monkeypatch,
            _pkce_record("http://proxy-a:4000", seconds_left=-10),
            {"http://proxy-a:4000": "sk-a-second-renewal-would-mint-this"},
        )

        with pytest.raises(UpError, match="Run `lite login --pkce` first"):
            _ensure_fresh_login(_make_group_ctx("http://proxy-a:4000", api_key=None))

        assert store.key_requests == []

    def test_re_reads_the_token_file_only_after_the_interactive_login_it_started(self, monkeypatch):
        monkeypatch.setattr(up_module.sys.stdin, "isatty", lambda: True)
        store = _FakeTokenStore(monkeypatch, _pkce_record("http://proxy-a:4000", seconds_left=-10), {})
        login_calls = _capture_login(
            monkeypatch,
            on_login=lambda: store.log_in(_pkce_record("http://proxy-a:4000", seconds_left=86_400), "sk-pkce-fresh"),
        )

        _ensure_fresh_login(_make_group_ctx("http://proxy-a:4000", api_key=None))

        assert login_calls == [("http://proxy-a:4000", True)]
        assert store.key_requests == ["http://proxy-a:4000"]


class TestUpCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_refuses_double_start_without_touching_settings_file(self, monkeypatch, tmp_path):
        settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        existing_backup = {"existed": False, "content": None}
        backup_path.write_text(json.dumps(existing_backup))

        with (
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh", "base_url": "http://localhost:4000"}),
            patch(f"{UP_MODULE}.get_stored_api_key", return_value="sk-fresh"),
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

        with (
            patch(f"{UP_MODULE}.load_token", return_value=None),
            patch(f"{UP_MODULE}.get_stored_api_key", return_value=None),
        ):
            result = self.runner.invoke(up, obj={"base_url": "http://localhost:4000"})

        assert result.exit_code != 0
        assert "lite login" in result.output

    def test_unreachable_proxy_fails_cleanly(self, monkeypatch, tmp_path):
        _patch_paths(monkeypatch, tmp_path)

        with (
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh", "base_url": "http://localhost:4000"}),
            patch(f"{UP_MODULE}.get_stored_api_key", return_value="sk-fresh"),
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
            patch(f"{UP_MODULE}.load_token", return_value={"key": "sk-fresh", "base_url": "http://localhost:4000"}),
            patch(f"{UP_MODULE}.get_stored_api_key", return_value="sk-fresh"),
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

    def test_surfaces_clean_error_on_a_corrupt_backup_file(self, monkeypatch, tmp_path):
        _settings_path, backup_path = _patch_paths(monkeypatch, tmp_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text("not json at all {{{")

        result = self.runner.invoke(down)

        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "invalid or unexpected JSON" in result.output


class TestUpCanInvokeTheRealLoginCommand:
    """`lite up` calls ctx.invoke(login) on the real command object.

    Every other test in this file monkeypatches `up_module.login` with a fake, so
    none of them would notice a login parameter that ctx.invoke cannot supply.
    """

    def test_ctx_invoke_supplies_every_login_parameter(self):
        from litellm.proxy.client.cli.commands.auth import login as real_login

        reached = []

        @click.command()
        @click.pass_context
        def driver(ctx):
            ctx.obj = {"base_url": "http://127.0.0.1:9"}
            ctx.invoke(real_login, pkce=False)

        with patch(
            f"{AUTH_MODULE}._start_cli_sso_flow",
            side_effect=lambda base_url: reached.append(base_url) or RuntimeError("stop"),
        ):
            result = CliRunner().invoke(driver, [], standalone_mode=False)

        assert not isinstance(result.exception, TypeError), result.exception
        assert reached == ["http://127.0.0.1:9"]

    def test_ctx_invoke_leaves_claude_settings_alone(self, tmp_path):
        from litellm.proxy.client.cli.commands.auth import login as real_login

        settings_path = tmp_path / "settings.json"

        @click.command()
        @click.pass_context
        def driver(ctx):
            ctx.obj = {"base_url": "http://127.0.0.1:9"}
            ctx.invoke(real_login, pkce=False)

        with (
            patch(f"{AUTH_MODULE}.CLAUDE_SETTINGS_PATH", settings_path),
            patch(f"{AUTH_MODULE}._start_cli_sso_flow", side_effect=RuntimeError("stop")),
        ):
            CliRunner().invoke(driver, [], standalone_mode=False)

        assert not settings_path.exists()
