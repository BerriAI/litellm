import json
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner



from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.config import (
    get_config_file_path,
    get_config_value,
    load_config,
    save_config,
)
from litellm.litellm_core_utils.private_json import write_private_json
from litellm.proxy.client.cli.interface import show_commands


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """Point HOME at tmp_path so tests never touch the developer's real ~/.litellm."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    return tmp_path


def _config_path(home: Path) -> Path:
    return home / ".litellm" / "config.json"


def _raise_home_unresolvable() -> str:
    raise RuntimeError("Could not determine home directory.")


class TestConfigSet:
    @pytest.mark.parametrize(
        "value",
        ["https://your-proxy.example.com", "http://your-proxy.example.com:8080"],
    )
    def test_set_stores_value_with_owner_only_permissions(self, cli_runner, isolated_home, value):
        result = cli_runner.invoke(cli, ["config", "set", "base_url", value])

        assert result.exit_code == 0
        config_file = _config_path(isolated_home)
        assert json.loads(config_file.read_text()) == {"base_url": value}
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
        assert str(config_file) in result.output

    def test_set_strips_trailing_slash(self, cli_runner, isolated_home):
        """Downstream commands join paths onto base_url; a stored trailing
        slash would produce double slashes in every request URL."""
        result = cli_runner.invoke(cli, ["config", "set", "base_url", "https://your-proxy.example.com/"])

        assert result.exit_code == 0
        assert json.loads(_config_path(isolated_home).read_text()) == {"base_url": "https://your-proxy.example.com"}

    def test_set_unknown_key_rejected_and_names_allowed_keys(self, cli_runner, isolated_home):
        result = cli_runner.invoke(cli, ["config", "set", "api_key", "sk-secret"])

        assert result.exit_code != 0
        assert "base_url" in result.output
        assert not _config_path(isolated_home).exists()

    @pytest.mark.parametrize("value", ["your-proxy.example.com", "ftp://your-proxy.example.com"])
    def test_set_base_url_without_http_scheme_rejected(self, cli_runner, isolated_home, value):
        result = cli_runner.invoke(cli, ["config", "set", "base_url", value])

        assert result.exit_code != 0
        assert "http" in result.output
        assert not _config_path(isolated_home).exists()

    @pytest.mark.parametrize("value", ["https://", "http://", "https:///some-path"])
    def test_set_base_url_without_host_rejected(self, cli_runner, isolated_home, value):
        """rstrip("/") would otherwise persist a bare "https:" that breaks every later request."""
        result = cli_runner.invoke(cli, ["config", "set", "base_url", value])

        assert result.exit_code != 0
        assert not _config_path(isolated_home).exists()

    @pytest.mark.parametrize(
        "value",
        [
            "https://proxy.example.com?env=prod",
            "https://proxy.example.com#prod",
            "https://proxy.example.com/?",
            "https://proxy.example.com/#",
        ],
    )
    def test_set_base_url_with_query_or_fragment_rejected(self, cli_runner, isolated_home, value):
        """Downstream commands join paths onto base_url; a stored query string or
        fragment would silently corrupt every request URL built from it. Bare
        trailing '?' / '#' parse as EMPTY query/fragment yet still break every
        joined path, so rejection must key off the raw characters."""
        result = cli_runner.invoke(cli, ["config", "set", "base_url", value])

        assert result.exit_code != 0
        assert "query" in result.output or "fragment" in result.output
        assert not _config_path(isolated_home).exists()

    def test_set_base_url_with_path_prefix_accepted(self, cli_runner, isolated_home):
        """Proxies are commonly served under a path prefix; the query/fragment
        rejection must not over-reach into legitimate paths."""
        result = cli_runner.invoke(cli, ["config", "set", "base_url", "https://proxy.example.com/litellm"])

        assert result.exit_code == 0
        assert json.loads(_config_path(isolated_home).read_text()) == {"base_url": "https://proxy.example.com/litellm"}

    def test_set_leaves_no_temp_files_behind(self, cli_runner, isolated_home):
        """The atomic write goes through a .tmp-* sibling; it must be renamed away,
        never abandoned next to the config."""
        result = cli_runner.invoke(cli, ["config", "set", "base_url", "https://your-proxy.example.com"])

        assert result.exit_code == 0
        config_file = _config_path(isolated_home)
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
        assert list(config_file.parent.glob(".tmp-*")) == []


class TestConfigGet:
    def test_get_prints_only_the_value(self, cli_runner, isolated_home):
        """stdout must be exactly the value so scripts can do URL=$(lite config get base_url)."""
        set_result = cli_runner.invoke(cli, ["config", "set", "base_url", "https://your-proxy.example.com"])
        assert set_result.exit_code == 0

        result = cli_runner.invoke(cli, ["config", "get", "base_url"])

        assert result.exit_code == 0
        assert result.stdout.strip() == "https://your-proxy.example.com"

    def test_get_unset_key_exits_one_with_stderr_message(self, cli_runner, isolated_home):
        result = cli_runner.invoke(cli, ["config", "get", "base_url"])

        assert result.exit_code == 1
        assert result.stdout.strip() == ""
        assert result.stderr != ""

    def test_get_without_key_lists_entries(self, cli_runner, isolated_home):
        set_result = cli_runner.invoke(cli, ["config", "set", "base_url", "https://your-proxy.example.com"])
        assert set_result.exit_code == 0

        result = cli_runner.invoke(cli, ["config", "get"])

        assert result.exit_code == 0
        assert "base_url = https://your-proxy.example.com" in result.output

    def test_get_without_key_when_nothing_set(self, cli_runner, isolated_home):
        result = cli_runner.invoke(cli, ["config", "get"])

        assert result.exit_code == 0
        assert "no config" in result.output.lower()


class TestConfigUnset:
    def test_unset_removes_key_from_file(self, cli_runner, isolated_home):
        set_result = cli_runner.invoke(cli, ["config", "set", "base_url", "https://your-proxy.example.com"])
        assert set_result.exit_code == 0

        result = cli_runner.invoke(cli, ["config", "unset", "base_url"])

        assert result.exit_code == 0
        assert "base_url" not in load_config()
        assert cli_runner.invoke(cli, ["config", "get", "base_url"]).exit_code == 1

    def test_unset_missing_key_is_idempotent(self, cli_runner, isolated_home):
        result = cli_runner.invoke(cli, ["config", "unset", "base_url"])

        assert result.exit_code == 0
        assert "not set" in result.output.lower()


class TestHiddenCommands:
    """`hidden_commands` lets a deployment curate what `lite` advertises.

    Two listings exist and both must honor it: click's own `--help` table and the
    hand-rolled block the interactive shell prints.
    """

    def test_nothing_is_hidden_by_default(self, cli_runner, isolated_home):
        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0, result.output
        assert "codex" in result.output
        assert "opencode" in result.output

    def test_configured_commands_drop_out_of_help(self, cli_runner, isolated_home):
        assert cli_runner.invoke(cli, ["config", "set", "hidden_commands", "codex,opencode"]).exit_code == 0

        result = cli_runner.invoke(cli, ["--help"])

        assert result.exit_code == 0, result.output
        assert "claude" in result.output
        assert "codex" not in result.output
        assert "opencode" not in result.output

    def test_configured_commands_drop_out_of_interactive_listing(self, capsys, isolated_home):
        save_config({"hidden_commands": "codex,keys"})

        show_commands()
        listing = capsys.readouterr().out

        assert "claude" in listing
        assert "codex" not in listing
        assert "keys" not in listing
        assert "teams" in listing

    def test_hidden_commands_are_still_invokable(self, cli_runner, isolated_home):
        """Hiding is about the listing only; anyone already scripting the command keeps working."""
        save_config({"hidden_commands": "codex"})

        with patch("litellm.proxy.client.cli.commands.agents.run_agent") as run_agent_mock:
            result = cli_runner.invoke(
                cli,
                ["--base-url", "http://localhost:4000", "--api-key", "sk-key", "codex", "exec", "do a thing"],
            )

        assert result.exit_code == 0, result.output
        _base_url, _api_key, command = run_agent_mock.call_args.args
        assert list(command) == ["codex", "exec", "do a thing"]

    def test_unset_brings_the_commands_back(self, cli_runner, isolated_home):
        assert cli_runner.invoke(cli, ["config", "set", "hidden_commands", "codex"]).exit_code == 0
        assert cli_runner.invoke(cli, ["config", "unset", "hidden_commands"]).exit_code == 0

        assert "codex" in cli_runner.invoke(cli, ["--help"]).output

    def test_set_normalizes_whitespace_and_ordering(self, cli_runner, isolated_home):
        result = cli_runner.invoke(cli, ["config", "set", "hidden_commands", " opencode , codex ,"])

        assert result.exit_code == 0, result.output
        assert json.loads(_config_path(isolated_home).read_text()) == {"hidden_commands": "codex,opencode"}

    @pytest.mark.parametrize("value", ["", " ", ",", " , "])
    def test_set_empty_list_rejected(self, cli_runner, isolated_home, value):
        """An empty value would silently hide nothing; point users at `config unset` instead."""
        result = cli_runner.invoke(cli, ["config", "set", "hidden_commands", value])

        assert result.exit_code != 0
        assert "unset" in result.output
        assert not _config_path(isolated_home).exists()

    def test_set_space_separated_list_rejected(self, cli_runner, isolated_home):
        """`lite config set hidden_commands "codex opencode"` would hide neither."""
        result = cli_runner.invoke(cli, ["config", "set", "hidden_commands", "codex opencode"])

        assert result.exit_code != 0
        assert "without spaces" in result.output
        assert not _config_path(isolated_home).exists()


class TestConfigHelpers:
    def test_get_config_file_path_under_home(self, isolated_home):
        assert get_config_file_path() == str(isolated_home / ".litellm" / "config.json")

    def test_load_config_missing_file_returns_empty(self, isolated_home):
        assert load_config() == {}

    def test_home_unresolvable_does_not_crash_cli(self, cli_runner, isolated_home, monkeypatch):
        """Path.home() raises RuntimeError in HOME-less containers; invocations that
        never needed the home dir (--api-key supplied) must keep working."""
        monkeypatch.setattr(
            "litellm.proxy.client.cli.commands.config.get_config_file_path",
            _raise_home_unresolvable,
        )

        assert load_config() == {}

        result = cli_runner.invoke(cli, ["--api-key", "sk-test", "config", "get"])
        assert result.exit_code == 0
        assert "(no config set)" in result.output

    @pytest.mark.parametrize(
        "content",
        [
            "{not json",
            '{"base_url": 123}',
            '["https://your-proxy.example.com"]',
            '"https://your-proxy.example.com"',
        ],
    )
    def test_load_config_invalid_content_returns_empty(self, isolated_home, content):
        """A corrupt or wrongly-shaped config file must degrade to defaults, never crash the CLI."""
        config_file = _config_path(isolated_home)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(content)

        assert load_config() == {}

    def test_load_config_invalid_utf8_returns_empty(self, isolated_home):
        """json.load raises UnicodeDecodeError (a ValueError but not a JSONDecodeError)
        on undecodable bytes; before catching ValueError this crashed every CLI
        invocation, including the `config set` needed to repair the file."""
        config_file = _config_path(isolated_home)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_bytes(b"\xff\xfe{}")

        assert load_config() == {}

    def test_save_config_round_trip_creates_dir_and_restricts_permissions(self, isolated_home):
        save_config({"base_url": "https://your-proxy.example.com"})

        assert load_config() == {"base_url": "https://your-proxy.example.com"}
        assert stat.S_IMODE(_config_path(isolated_home).stat().st_mode) == 0o600

    def test_get_config_value_unset_then_set(self, isolated_home):
        assert get_config_value("base_url") is None

        save_config({"base_url": "https://your-proxy.example.com"})

        assert get_config_value("base_url") == "https://your-proxy.example.com"

    def test_corrupt_config_file_warns_on_stderr_but_command_succeeds(self, cli_runner, isolated_home):
        """Silently ignoring a broken config file leaves users debugging why their
        stored base_url stopped applying; the CLI must keep working but say why."""
        config_file = _config_path(isolated_home)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("{not json")

        result = cli_runner.invoke(cli, ["config", "get"])

        assert result.exit_code == 0
        assert "Warning: ignoring invalid config file" in result.stderr


class TestWritePrivateJson:
    def test_failed_write_preserves_previous_file_and_removes_temp(self, tmp_path):
        """json.dump can fail partway through serializing; writing to a temp file
        and renaming keeps the previous file intact through a crash mid-write."""
        target = tmp_path / "config.json"
        original = '{"base_url": "https://original.example.com"}'
        target.write_text(original)

        with pytest.raises(TypeError):
            write_private_json(str(target), {"bad": object()})

        assert target.read_text() == original
        assert list(tmp_path.glob(".tmp-*")) == []

    def test_interrupted_write_removes_temp_file(self, tmp_path, monkeypatch):
        """Ctrl-C is BaseException, which `except Exception` misses; an interrupt
        mid-write must not abandon a .tmp-* file next to the config forever."""

        def _interrupt(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt()

        monkeypatch.setattr("litellm.litellm_core_utils.private_json.json.dump", _interrupt)
        target = tmp_path / "config.json"

        with pytest.raises(KeyboardInterrupt):
            write_private_json(str(target), {"base_url": "https://your-proxy.example.com"})

        assert not target.exists()
        assert list(tmp_path.glob(".tmp-*")) == []
