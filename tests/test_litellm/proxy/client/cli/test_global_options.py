# stdlib imports
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


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
        cli_runner.invoke(
            cli, ["--base-url", "https://gateway.litellm-sandbox.ai/", "login"]
        )

    mock_post.assert_called_once_with(
        "https://gateway.litellm-sandbox.ai/sso/cli/start", timeout=10
    )


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
