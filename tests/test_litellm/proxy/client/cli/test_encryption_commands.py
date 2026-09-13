"""CLI tests for the ``litellm-proxy encryption migrate`` command.

The HTTP client is mocked, so these assert the command's request routing (GET
check vs POST migrate, dry-run param) and its residual-state messaging without a
live proxy.
"""

import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli import main as cli_main
from litellm.proxy.client.cli.commands import encryption as enc_cli


class _FakeHTTPClient:
    """Stand-in for HTTPClient: records the last request and returns a canned body."""

    last = None
    response = {"status": "success", "report": {"residual_legacy": 0, "locations": {}}}

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def request(self, method, path, **kwargs):
        _FakeHTTPClient.last = {"method": method, "path": path, **kwargs}
        return _FakeHTTPClient.response


@pytest.fixture
def runner(monkeypatch):
    _FakeHTTPClient.last = None
    _FakeHTTPClient.response = {
        "status": "success",
        "report": {"residual_legacy": 0, "locations": {}},
    }
    monkeypatch.setattr(enc_cli, "HTTPClient", _FakeHTTPClient)
    return CliRunner()


def test_migrate_check_hits_check_route(runner):
    result = runner.invoke(cli_main.cli, ["encryption", "migrate", "--check"])
    assert result.exit_code == 0, result.output
    assert _FakeHTTPClient.last["method"] == "GET"
    assert _FakeHTTPClient.last["path"] == "/credentials/migrate-encryption/check"
    assert "No legacy values remaining" in result.output


def test_migrate_default_posts_without_dry_run(runner):
    result = runner.invoke(cli_main.cli, ["encryption", "migrate"])
    assert result.exit_code == 0, result.output
    assert _FakeHTTPClient.last["method"] == "POST"
    assert _FakeHTTPClient.last["path"] == "/credentials/migrate-encryption"
    assert _FakeHTTPClient.last["params"] is None


def test_migrate_dry_run_sets_param(runner):
    result = runner.invoke(cli_main.cli, ["encryption", "migrate", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert _FakeHTTPClient.last["method"] == "POST"
    assert _FakeHTTPClient.last["params"] == {"dry_run": "true"}


def test_migrate_reports_residual_legacy(runner):
    _FakeHTTPClient.response = {
        "status": "success",
        "report": {"residual_legacy": 3, "locations": {}},
    }
    result = runner.invoke(cli_main.cli, ["encryption", "migrate", "--check"])
    assert result.exit_code == 0, result.output
    assert "Residual legacy values remaining: 3" in result.output
