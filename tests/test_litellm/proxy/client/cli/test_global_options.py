# stdlib imports
import io
import os
import sys
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from litellm._version import version as litellm_version
from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.main import _ensure_utf8_output


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


def test_ensure_utf8_output_reconfigures_legacy_encoding():
    """A cp1252 console must be switched to UTF-8 so emoji output can't raise UnicodeEncodeError."""
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

    with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
        with pytest.raises(UnicodeEncodeError):
            sys.stdout.write("\u274c")

        _ensure_utf8_output()

        assert sys.stdout.encoding == "utf-8"
        assert sys.stderr.encoding == "utf-8"
        sys.stdout.write("\u274c \u2705")


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
