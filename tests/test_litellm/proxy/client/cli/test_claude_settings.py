import json
import shlex
import stat
import time
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from litellm.litellm_core_utils.cli_token_utils import CliTokenRecord
from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.claude_settings import (
    AUTOROUTE_BACKUP_PATH,
    BACKUP_PATH,
    SETTINGS_FILE_OWNERS,
    ClaudeSettingsError,
    SettingsFileOwner,
    resolve_api_key_helper,
    write_claude_settings,
)


def _owners(*backup_paths):
    """Stand-in owners for the real `lite up` / `lite autoroute up` registry."""
    return tuple(SettingsFileOwner(path, "lite up", "lite down") for path in backup_paths)

CLAUDE_SETTINGS_MODULE = "litellm.proxy.client.cli.commands.claude_settings"
AUTH_MODULE = "litellm.proxy.client.cli.commands.auth"


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "claude" / "settings.json", tmp_path / "backup.json"


@pytest.fixture
def lite_on_path():
    with patch(f"{CLAUDE_SETTINGS_MODULE}.shutil.which", return_value="/usr/local/bin/lite"):
        yield


class TestWriteClaudeSettings:
    def test_creates_the_file_and_its_parent_when_missing(self, paths, lite_on_path):
        settings_path, backup_path = paths
        assert not settings_path.parent.exists()

        write_claude_settings("https://proxy.example.com/", settings_path, _owners(backup_path))

        written = json.loads(settings_path.read_text())
        assert written["env"]["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
        assert written["apiKeyHelper"] == "/usr/local/bin/lite --base-url https://proxy.example.com auth print-token"

    def test_updates_an_existing_file_preserving_unrelated_settings(self, paths, lite_on_path):
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "permissions": {"allow": ["Bash"]},
                    "env": {"SOME_OTHER_VAR": "keep-me", "ANTHROPIC_BASE_URL": "https://old.example.com"},
                    "apiKeyHelper": "old-helper",
                }
            )
        )

        write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

        written = json.loads(settings_path.read_text())
        assert written["theme"] == "dark"
        assert written["permissions"] == {"allow": ["Bash"]}
        assert written["env"]["SOME_OTHER_VAR"] == "keep-me"
        assert written["env"]["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
        assert written["apiKeyHelper"] != "old-helper"

    def test_rerunning_against_a_new_proxy_refreshes_both_base_url_and_helper(self, paths, lite_on_path):
        settings_path, backup_path = paths

        write_claude_settings("https://first.example.com", settings_path, _owners(backup_path))
        write_claude_settings("https://second.example.com", settings_path, _owners(backup_path))

        written = json.loads(settings_path.read_text())
        assert written["env"]["ANTHROPIC_BASE_URL"] == "https://second.example.com"
        assert "second.example.com" in written["apiKeyHelper"]
        assert "first.example.com" not in written["apiKeyHelper"]

    def test_drops_a_stray_static_api_key_so_the_helper_token_wins(self, paths, lite_on_path):
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": "sk-leaked"}}))

        write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

        assert "ANTHROPIC_API_KEY" not in json.loads(settings_path.read_text())["env"]

    def test_written_file_is_owner_only(self, paths, lite_on_path):
        settings_path, backup_path = paths
        write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))
        assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600

    def test_refuses_while_lite_up_holds_a_backup(self, paths, lite_on_path):
        settings_path, backup_path = paths
        backup_path.write_text("{}")

        with pytest.raises(ClaudeSettingsError, match="lite down"):
            write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

        assert not settings_path.exists()

    def test_refuses_on_corrupt_existing_settings_without_touching_the_file(self, paths, lite_on_path):
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("not json at all {{{")

        with pytest.raises(ClaudeSettingsError, match="invalid JSON"):
            write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

        assert settings_path.read_text() == "not json at all {{{"

    def test_reports_an_actionable_error_when_lite_is_not_on_path(self, paths):
        settings_path, backup_path = paths
        with patch(f"{CLAUDE_SETTINGS_MODULE}.shutil.which", return_value=None):
            with pytest.raises(ClaudeSettingsError, match="Could not find `lite`"):
                write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

        assert not settings_path.exists()

    def test_reports_an_actionable_error_on_a_non_utf8_file(self, paths, lite_on_path):
        """Bytes that are not valid UTF-8 must not escape as UnicodeDecodeError.

        UnicodeDecodeError is a ValueError, not an OSError, so a decode-side catch
        is easy to miss; login's broad `except Exception` would then relabel it as
        an authentication failure and exit 0.
        """
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_bytes(b'{"theme": "\xff\xfe"}')

        with pytest.raises(ClaudeSettingsError, match="invalid JSON"):
            write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

    def test_reports_an_actionable_error_when_the_file_cannot_be_read(self, paths, lite_on_path):
        """An unreadable settings file must not surface as "Authentication failed".

        login wraps the whole flow in a broad `except Exception`, so any OSError
        escaping this function gets relabelled as an auth failure and sends the
        user looking at their SSO config instead of at file permissions.
        """
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.mkdir()

        with pytest.raises(ClaudeSettingsError, match="Could not read"):
            write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

    def test_reports_an_actionable_error_when_the_file_cannot_be_written(self, paths, lite_on_path):
        settings_path, backup_path = paths
        with patch(
            f"{CLAUDE_SETTINGS_MODULE}.write_private_json",
            side_effect=OSError("Read-only file system"),
        ):
            with pytest.raises(ClaudeSettingsError, match="Read-only file system"):
                write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))


class TestApiKeyHelperIsActuallyInvocable:
    """The helper string is executed verbatim by Claude Code, so it has to parse.

    Asserting only on its text is what let a malformed command (`--base-url`, a
    top-level group option, placed after the `print-token` subcommand) ship: click
    rejects it with "No such option" and every Claude Code request loses its token.
    """

    def _helper_args(self, base_url):
        with patch(f"{CLAUDE_SETTINGS_MODULE}.shutil.which", return_value="/usr/local/bin/lite"):
            return shlex.split(resolve_api_key_helper(base_url))[1:]

    def test_the_generated_command_parses(self):
        result = CliRunner().invoke(cli, self._helper_args("http://localhost:4000"))

        assert "No such option" not in result.output
        assert result.exit_code != 2

    def test_the_generated_command_reaches_print_token(self):
        with patch(f"{AUTH_MODULE}.load_cli_token", return_value=None):
            result = CliRunner().invoke(cli, self._helper_args("http://localhost:4000"))

        assert "Not authenticated" in result.output

    def test_the_generated_command_carries_the_base_url_through(self):
        stale = CliTokenRecord(
            base_url="http://other-proxy.example.com",
            key="sk-stale",
            timestamp=time.time(),
        )
        with patch(f"{AUTH_MODULE}.load_cli_token", return_value=stale):
            result = CliRunner().invoke(cli, self._helper_args("http://localhost:4000"))

        assert "Not authenticated for this server" in result.output


class TestConflictingOwnersOfTheSettingsFile:
    """Both `lite up` and `lite autoroute up` restore a backup when they stop.

    Guarding only one of them leaves the other free to silently revert this
    write, which is the exact hazard the guard exists to prevent.
    """

    def test_any_owner_holding_a_backup_blocks_the_write(self, tmp_path, lite_on_path):
        settings_path = tmp_path / "claude" / "settings.json"

        for index, owner in enumerate(SETTINGS_FILE_OWNERS):
            backup = tmp_path / f"backup-{index}.json"
            backup.write_text("{}")
            stand_in = SettingsFileOwner(backup, owner.start_command, owner.stop_command)
            with pytest.raises(ClaudeSettingsError, match="currently managing"):
                write_claude_settings("https://proxy.example.com", settings_path, (stand_in,))
            backup.unlink()
            assert not settings_path.exists()

    def test_the_error_names_the_owner_that_actually_holds_the_file(self, tmp_path, lite_on_path):
        settings_path = tmp_path / "claude" / "settings.json"
        backup = tmp_path / "auto.json"
        backup.write_text("{}")
        autoroute = SettingsFileOwner(backup, "lite autoroute up", "lite autoroute down")

        with pytest.raises(ClaudeSettingsError, match="`lite autoroute up` is currently managing"):
            write_claude_settings("https://proxy.example.com", settings_path, (autoroute,))
        with pytest.raises(ClaudeSettingsError, match="Run `lite autoroute down` first"):
            write_claude_settings("https://proxy.example.com", settings_path, (autoroute,))

    def test_the_registry_matches_the_paths_the_commands_actually_use(self):
        """A second definition of the autoroute dir must not drift from this one."""
        from litellm.proxy.client.cli.commands.autoroute.process import AUTOROUTE_DIR

        assert AUTOROUTE_BACKUP_PATH == AUTOROUTE_DIR / "claude_settings_backup.json"
        assert {o.backup_path for o in SETTINGS_FILE_OWNERS} == {BACKUP_PATH, AUTOROUTE_BACKUP_PATH}
        assert {o.stop_command for o in SETTINGS_FILE_OWNERS} == {"lite down", "lite autoroute down"}


class TestDoesNotDestroyUserOwnedStructure:
    def test_writes_through_a_symlinked_settings_file(self, tmp_path, lite_on_path):
        """os.replace() swaps the symlink for a regular file, detaching a dotfiles repo.

        There is no backup here to undo that, so the link must survive and its
        target must be the thing that gets updated.
        """
        real = tmp_path / "dotfiles" / "settings.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"theme": "dark"}))
        link = tmp_path / "claude" / "settings.json"
        link.parent.mkdir()
        link.symlink_to(real)

        write_claude_settings("https://proxy.example.com", link, ())

        assert link.is_symlink()
        assert json.loads(real.read_text())["env"]["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
        assert json.loads(real.read_text())["theme"] == "dark"

    def test_refuses_rather_than_discarding_a_non_object_env(self, paths, lite_on_path):
        """merge coerces a non-dict env to {}; that is silent data loss on a persistent write."""
        settings_path, backup_path = paths
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"theme": "dark", "env": "not-an-object"}))

        with pytest.raises(ClaudeSettingsError, match="non-object"):
            write_claude_settings("https://proxy.example.com", settings_path, _owners(backup_path))

        assert json.loads(settings_path.read_text())["env"] == "not-an-object"
