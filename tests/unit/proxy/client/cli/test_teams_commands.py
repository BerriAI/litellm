from typing import Final

from click.testing import CliRunner
from requests_mock import Mocker

from litellm.proxy.client.cli import cli


def test_teams_available_exits_nonzero_when_the_proxy_rejects_the_request(requests_mock: Mocker) -> None:
    requests_mock.get(
        "http://localhost:4000/team/available",
        status_code=400,
        json={"detail": {"error": "Database not connected"}},
    )

    result: Final = CliRunner().invoke(
        cli,
        ("--base-url", "http://localhost:4000", "--api-key", "sk-test", "teams", "available"),
    )

    assert result.exit_code == 1, result.output
    assert "HTTP 400" in result.stderr
    assert "Database not connected" in result.stderr
