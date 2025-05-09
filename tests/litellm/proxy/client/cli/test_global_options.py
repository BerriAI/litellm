# stdlib imports
from litellm.proxy.client.cli import cli
from litellm._version import version as litellm_version
from click.testing import CliRunner
import pytest


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_cli_version_flag(cli_runner):
    """Test that --version prints the correct version and exits successfully"""
    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"litellm-proxy version: {litellm_version}" in result.output
