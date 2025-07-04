# stdlib imports
import os
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from litellm._version import version as litellm_version
from litellm.proxy.client.cli import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_cli_version_flag(cli_runner):
    """Test that --version prints the correct version, server URL, and server version, and exits successfully"""
    with patch(
        "litellm.proxy.client.health.HealthManagementClient.get_server_version",
        return_value="1.2.3",
    ), patch.dict(os.environ, {"LITELLM_PROXY_URL": "http://localhost:4000"}):
        result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"LiteLLM Proxy CLI Version: {litellm_version}" in result.output
    assert "LiteLLM Proxy Server URL: http://localhost:4000" in result.output
    assert "LiteLLM Proxy Server Version: 1.2.3" in result.output


def test_cli_version_command(cli_runner):
    """Test that 'version' command prints the correct version, server URL, and server version, and exits successfully"""
    with patch(
        "litellm.proxy.client.health.HealthManagementClient.get_server_version",
        return_value="1.2.3",
    ), patch.dict(os.environ, {"LITELLM_PROXY_URL": "http://localhost:4000"}):
        result = cli_runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert f"LiteLLM Proxy CLI Version: {litellm_version}" in result.output
    assert "LiteLLM Proxy Server URL: http://localhost:4000" in result.output
    assert "LiteLLM Proxy Server Version: 1.2.3" in result.output
