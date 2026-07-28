import json
import os
import stat
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.abspath("../../.."))


from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.config import (
    get_config_file_path,
    get_config_value,
    load_config,
    save_config,
)


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

    def test_save_config_round_trip_creates_dir_and_restricts_permissions(self, isolated_home):
        save_config({"base_url": "https://your-proxy.example.com"})

        assert load_config() == {"base_url": "https://your-proxy.example.com"}
        assert stat.S_IMODE(_config_path(isolated_home).stat().st_mode) == 0o600

    def test_get_config_value_unset_then_set(self, isolated_home):
        assert get_config_value("base_url") is None

        save_config({"base_url": "https://your-proxy.example.com"})

        assert get_config_value("base_url") == "https://your-proxy.example.com"
