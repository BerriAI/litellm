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


class TestBaseUrlResolution:
    """The base URL stored by `lite login` becomes the default for later runs"""

    def _server_url_line(self, cli_runner, args, stored_base_url, monkeypatch):
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
        with (
            patch(
                "litellm.proxy.client.health.HealthManagementClient.get_server_version",
                return_value="1.2.3",
            ),
            patch(
                "litellm.proxy.client.cli.main.get_stored_base_url",
                return_value=stored_base_url,
            ),
            patch(
                "litellm.proxy.client.cli.main.get_stored_api_key",
                return_value=None,
            ),
        ):
            result = cli_runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        return result.output

    def test_stored_base_url_used_when_nothing_specified(self, cli_runner, monkeypatch):
        output = self._server_url_line(
            cli_runner, ["version"], "https://llm.acme.com", monkeypatch
        )
        assert "LiteLLM Proxy Server URL: https://llm.acme.com" in output

    def test_flag_wins_over_stored_base_url(self, cli_runner, monkeypatch):
        output = self._server_url_line(
            cli_runner,
            ["--base-url", "http://flag:1234", "version"],
            "https://llm.acme.com",
            monkeypatch,
        )
        assert "LiteLLM Proxy Server URL: http://flag:1234" in output

    def test_env_var_wins_over_stored_base_url(self, cli_runner, monkeypatch):
        monkeypatch.setenv("LITELLM_PROXY_URL", "http://env:1234")
        with (
            patch(
                "litellm.proxy.client.health.HealthManagementClient.get_server_version",
                return_value="1.2.3",
            ),
            patch(
                "litellm.proxy.client.cli.main.get_stored_base_url",
                return_value="https://llm.acme.com",
            ),
            patch(
                "litellm.proxy.client.cli.main.get_stored_api_key",
                return_value=None,
            ),
        ):
            result = cli_runner.invoke(cli, ["version"])
        assert result.exit_code == 0, result.output
        assert "LiteLLM Proxy Server URL: http://env:1234" in result.output

    def test_falls_back_to_localhost_without_stored_url(self, cli_runner, monkeypatch):
        output = self._server_url_line(cli_runner, ["version"], None, monkeypatch)
        assert "LiteLLM Proxy Server URL: http://localhost:4000" in output

    def test_version_flag_uses_stored_base_url(self, cli_runner, monkeypatch):
        output = self._server_url_line(
            cli_runner, ["--version"], "https://llm.acme.com", monkeypatch
        )
        assert "LiteLLM Proxy Server URL: https://llm.acme.com" in output

    def test_version_flag_env_var_wins_over_stored(self, cli_runner, monkeypatch):
        monkeypatch.setenv("LITELLM_PROXY_URL", "http://env:1234")
        with (
            patch(
                "litellm.proxy.client.health.HealthManagementClient.get_server_version",
                return_value="1.2.3",
            ),
            patch(
                "litellm.proxy.client.cli.main.get_stored_base_url",
                return_value="https://llm.acme.com",
            ),
        ):
            result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0, result.output
        assert "LiteLLM Proxy Server URL: http://env:1234" in result.output

    def test_version_flag_falls_back_to_localhost(self, cli_runner, monkeypatch):
        output = self._server_url_line(cli_runner, ["--version"], None, monkeypatch)
        assert "LiteLLM Proxy Server URL: http://localhost:4000" in output
