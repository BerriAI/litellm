# stdlib imports
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path


import litellm.proxy.client.cli
from litellm._version import version as litellm_version
from litellm.proxy.client.cli import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_cli_version_flag(cli_runner):
    """Test that --version prints the correct version, server URL, and server version, and exits successfully"""
    with (
        patch(
            "litellm.proxy.client.health.HealthManagementClient.get_server_version",
            return_value="1.2.3",
        ),
        patch.dict(os.environ, {"LITELLM_PROXY_URL": "http://localhost:4000"}),
    ):
        result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"LiteLLM Proxy CLI Version: {litellm_version}" in result.output
    assert "LiteLLM Proxy Server URL: http://localhost:4000" in result.output
    assert "LiteLLM Proxy Server Version: 1.2.3" in result.output


def test_cli_source_is_ascii_only():
    """Non-ASCII output (emoji, box-drawing chars) raises UnicodeEncodeError on legacy Windows
    consoles (cp1252), so the whole CLI package must stay ASCII-only."""
    cli_root = Path(litellm.proxy.client.cli.__file__).parent
    offenders = [
        f"{path.relative_to(cli_root)}:{line_number}: {line.strip()}"
        for path in sorted(cli_root.rglob("*.py"))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if not line.isascii()
    ]
    assert offenders == []


def test_base_url_trailing_slash_normalized(cli_runner):
    """A trailing slash on --base-url must not produce a double slash (e.g. '//sso/cli/start')."""
    with (
        patch("webbrowser.open"),
        patch(
            "requests.post",
            return_value=Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "login_id": "cli-test-uuid",
                        "poll_secret": "poll-secret",
                        "user_code": "ABCD-EFGH",
                    }
                ),
                raise_for_status=Mock(),
            ),
        ) as mock_post,
        patch("requests.get", side_effect=ValueError("stop after start request")),
    ):
        cli_runner.invoke(cli, ["--base-url", "https://gateway.litellm-sandbox.ai/", "login"])

    mock_post.assert_called_once_with("https://gateway.litellm-sandbox.ai/sso/cli/start", timeout=10)


def test_cli_version_command(cli_runner):
    """Test that 'version' command prints the correct version, server URL, and server version, and exits successfully"""
    with (
        patch(
            "litellm.proxy.client.health.HealthManagementClient.get_server_version",
            return_value="1.2.3",
        ),
        patch.dict(os.environ, {"LITELLM_PROXY_URL": "http://localhost:4000"}),
    ):
        result = cli_runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert f"LiteLLM Proxy CLI Version: {litellm_version}" in result.output
    assert "LiteLLM Proxy Server URL: http://localhost:4000" in result.output
    assert "LiteLLM Proxy Server Version: 1.2.3" in result.output


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """Point HOME at tmp_path so tests never touch the developer's real ~/.litellm."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    return tmp_path


def _write_config_file(home: Path, config: dict[str, str]) -> None:
    config_dir = home / ".litellm"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(config))


def _invoke_version(cli_runner: CliRunner, *args: str):
    with patch(
        "litellm.proxy.client.health.HealthManagementClient.get_server_version",
        return_value="1.2.3",
    ):
        return cli_runner.invoke(cli, [*args, "version"])


def test_base_url_read_from_config_file(cli_runner, isolated_home):
    """base_url precedence: flag > env > config file > default."""
    _write_config_file(isolated_home, {"base_url": "https://config-proxy.example.com"})

    result = _invoke_version(cli_runner)

    assert result.exit_code == 0
    assert "LiteLLM Proxy Server URL: https://config-proxy.example.com" in result.output


def test_env_var_beats_config_file_base_url(cli_runner, isolated_home, monkeypatch):
    _write_config_file(isolated_home, {"base_url": "https://config-proxy.example.com"})
    monkeypatch.setenv("LITELLM_PROXY_URL", "http://env-proxy.example.com:5000")

    result = _invoke_version(cli_runner)

    assert result.exit_code == 0
    assert "LiteLLM Proxy Server URL: http://env-proxy.example.com:5000" in result.output


def test_base_url_flag_beats_env_var_and_config_file(cli_runner, isolated_home, monkeypatch):
    _write_config_file(isolated_home, {"base_url": "https://config-proxy.example.com"})
    monkeypatch.setenv("LITELLM_PROXY_URL", "http://env-proxy.example.com:5000")

    result = _invoke_version(cli_runner, "--base-url", "http://flag-proxy.example.com:9000")

    assert result.exit_code == 0
    assert "LiteLLM Proxy Server URL: http://flag-proxy.example.com:9000" in result.output


def test_default_base_url_unchanged_without_config_file(cli_runner, isolated_home):
    result = _invoke_version(cli_runner)

    assert result.exit_code == 0
    assert "LiteLLM Proxy Server URL: http://localhost:4000" in result.output


def test_corrupt_config_file_falls_back_to_default(cli_runner, isolated_home):
    """A corrupt config file must never crash the CLI."""
    config_dir = isolated_home / ".litellm"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text("{not json")

    result = _invoke_version(cli_runner)

    assert result.exit_code == 0
    assert "LiteLLM Proxy Server URL: http://localhost:4000" in result.output
