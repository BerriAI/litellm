import json
from unittest.mock import MagicMock

import pytest
import requests
import responses
from click.testing import CliRunner

from litellm.proxy.client.cli.commands.credentials import PollClock, connect_user_credential
from litellm.proxy.client.cli.main import cli
from litellm.proxy.client.credentials import CredentialsManagementClient


@pytest.fixture
def mock_credentials_client(monkeypatch):
    """Patch the CredentialsManagementClient used by the CLI commands."""
    mock_client = MagicMock()
    monkeypatch.setattr(
        "litellm.proxy.client.cli.commands.credentials.CredentialsManagementClient",
        mock_client,
    )
    return mock_client


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_alist_credentials_table_format(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {
        "credentials": [
            {
                "credential_name": "test-cred-1",
                "credential_info": {"custom_llm_provider": "azure"},
            },
            {
                "credential_name": "test-cred-2",
                "credential_info": {"custom_llm_provider": "anthropic"},
            },
        ]
    }
    mock_instance = mock_credentials_client.return_value
    mock_instance.list.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "list"])

    # Verify
    assert result.exit_code == 0
    assert "test-cred-1" in result.output
    assert "azure" in result.output
    assert "test-cred-2" in result.output
    assert "anthropic" in result.output


def test_alist_credentials_json_format(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {
        "credentials": [
            {
                "credential_name": "test-cred",
                "credential_info": {"custom_llm_provider": "azure"},
            }
        ]
    }
    mock_instance = mock_credentials_client.return_value
    mock_instance.list.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "list", "--format", "json"])

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response


def test_acreate_credential_success(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {"status": "success", "credential_name": "test-cred"}
    mock_instance = mock_credentials_client.return_value
    mock_instance.create.return_value = mock_response

    # Run command
    result = cli_runner.invoke(
        cli,
        [
            "credentials",
            "create",
            "test-cred",
            "--info",
            '{"custom_llm_provider": "azure"}',
            "--values",
            '{"api_key": "test-key"}',
        ],
    )

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.create.assert_called_once_with(
        "test-cred",
        {"custom_llm_provider": "azure"},
        {"api_key": "test-key"},
        display_name=None,
    )


def test_acreate_credential_invalid_json(cli_runner, mock_credentials_client):
    # Run command with invalid JSON
    result = cli_runner.invoke(
        cli,
        [
            "credentials",
            "create",
            "test-cred",
            "--info",
            "invalid-json",
            "--values",
            '{"api_key": "test-key"}',
        ],
    )

    # Verify
    assert result.exit_code == 2
    assert "Invalid JSON" in result.output
    mock_instance = mock_credentials_client.return_value
    mock_instance.create.assert_not_called()


def test_acreate_credential_http_error(cli_runner, mock_credentials_client):
    # Mock HTTP error
    mock_instance = mock_credentials_client.return_value
    mock_error_response = MagicMock()
    mock_error_response.status_code = 400
    mock_error_response.json.return_value = {"error": "Invalid request"}
    mock_instance.create.side_effect = requests.exceptions.HTTPError(response=mock_error_response)

    # Run command
    result = cli_runner.invoke(
        cli,
        [
            "credentials",
            "create",
            "test-cred",
            "--info",
            '{"custom_llm_provider": "azure"}',
            "--values",
            '{"api_key": "test-key"}',
        ],
    )

    # Verify
    assert result.exit_code == 1
    assert "Error: HTTP 400" in result.output
    assert "Invalid request" in result.output


def test_adelete_credential_success(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {"status": "success", "message": "Credential deleted"}
    mock_instance = mock_credentials_client.return_value
    mock_instance.delete.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "delete", "test-cred"])

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.delete.assert_called_once_with("test-cred")


def test_adelete_credential_http_error(cli_runner, mock_credentials_client):
    # Mock HTTP error
    mock_instance = mock_credentials_client.return_value
    mock_error_response = MagicMock()
    mock_error_response.status_code = 404
    mock_error_response.json.return_value = {"error": "Credential not found"}
    mock_instance.delete.side_effect = requests.exceptions.HTTPError(response=mock_error_response)

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "delete", "test-cred"])

    # Verify
    assert result.exit_code == 1
    assert "Error: HTTP 404" in result.output
    assert "Credential not found" in result.output


def test_aget_credential_success(cli_runner, mock_credentials_client):
    # Mock response data
    mock_response = {
        "credential_name": "test-cred",
        "credential_info": {"custom_llm_provider": "azure"},
    }
    mock_instance = mock_credentials_client.return_value
    mock_instance.get.return_value = mock_response

    # Run command
    result = cli_runner.invoke(cli, ["credentials", "get", "test-cred"])

    # Verify
    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data == mock_response
    mock_instance.get.assert_called_once_with("test-cred")


def test_list_credentials_table_shows_display_name_and_source(cli_runner, mock_credentials_client):
    mock_credentials_client.return_value.list.return_value = {
        "credentials": [
            {"credential_name": "openai-prod", "display_name": "Prod OpenAI", "source": "db", "credential_info": {}},
            {"credential_name": "from-yaml", "display_name": None, "source": "config", "credential_info": {}},
        ]
    }

    result = cli_runner.invoke(cli, ["credentials", "list"])

    assert result.exit_code == 0
    assert "Prod OpenAI" in result.output
    assert "config" in result.output
    assert "None" not in result.output


def test_create_credential_forwards_the_display_name(cli_runner, mock_credentials_client):
    mock_instance = mock_credentials_client.return_value
    mock_instance.create.return_value = {"success": True}

    result = cli_runner.invoke(
        cli,
        ["credentials", "create", "openai-prod", "--info", "{}", "--values", "{}", "--display-name", "Prod OpenAI"],
    )

    assert result.exit_code == 0
    mock_instance.create.assert_called_once_with("openai-prod", {}, {}, display_name="Prod OpenAI")


@pytest.mark.parametrize(
    ("flags", "expected"),
    [(["--display-name", "Prod OpenAI"], "Prod OpenAI"), (["--clear-display-name"], None)],
)
def test_update_credential_sets_or_clears_the_display_name(cli_runner, mock_credentials_client, flags, expected):
    mock_instance = mock_credentials_client.return_value
    mock_instance.update_display_name.return_value = {"success": True}

    result = cli_runner.invoke(cli, ["credentials", "update", "openai-prod", *flags])

    assert result.exit_code == 0, result.output
    mock_instance.update_display_name.assert_called_once_with("openai-prod", expected)


@pytest.mark.parametrize("flags", [[], ["--display-name", "Prod", "--clear-display-name"]])
def test_update_credential_requires_exactly_one_display_name_flag(cli_runner, mock_credentials_client, flags):
    result = cli_runner.invoke(cli, ["credentials", "update", "openai-prod", *flags])

    assert result.exit_code != 0
    assert "exactly one" in result.output
    mock_credentials_client.return_value.update_display_name.assert_not_called()


PROXY = "http://proxy.test"
CONNECT_URL = f"{PROXY}/credentials/team%2Fcopilot/user_connection"


def _start_body(interval: int = 0, expires_in: int = 900) -> dict[str, object]:
    return {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "expires_in": expires_in,
        "interval": interval,
        "flow_handle": "sealed-handle",
    }


class _FakeClock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.waits: tuple[float, ...] = ()

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds
        self.waits = (*self.waits, seconds)

    def as_poll_clock(self) -> PollClock:
        return PollClock(now=lambda: self.elapsed, sleep=self.sleep)


def _run_connect(cli_runner):
    return cli_runner.invoke(
        cli, ["--base-url", PROXY, "--api-key", "sk-user", "credentials", "connect", "team/copilot"]
    )


@responses.activate
def test_connect_shows_the_github_code_and_reports_the_connected_login(cli_runner):
    responses.add(responses.POST, f"{CONNECT_URL}/start", json=_start_body())
    responses.add(responses.POST, f"{CONNECT_URL}/poll", json={"status": "pending"})
    responses.add(responses.POST, f"{CONNECT_URL}/poll", json={"status": "connected", "github_login": "octocat"})

    result = _run_connect(cli_runner)

    assert result.exit_code == 0, result.output
    assert "https://github.com/login/device" in result.output
    assert "ABCD-1234" in result.output
    assert "Connected as octocat" in result.output
    polls = [call for call in responses.calls if call.request.url.endswith("/poll")]
    assert len(polls) == 2
    assert all(json.loads(call.request.body) == {"flow_handle": "sealed-handle"} for call in polls)
    assert all(call.request.headers["Authorization"] == "Bearer sk-user" for call in responses.calls)


@pytest.mark.parametrize(
    ("status", "message"),
    [("expired", "expired"), ("denied", "denied"), ("no_copilot_seat", "no GitHub Copilot seat")],
)
@responses.activate
def test_connect_exits_nonzero_when_the_login_does_not_complete(cli_runner, status, message):
    responses.add(responses.POST, f"{CONNECT_URL}/start", json=_start_body())
    responses.add(responses.POST, f"{CONNECT_URL}/poll", json={"status": status})

    result = _run_connect(cli_runner)

    assert result.exit_code == 1
    assert message in result.output
    assert "Connected as" not in result.output


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [(401, "An authenticated user is required"), (404, "Credential not found")],
)
@responses.activate
def test_connect_surfaces_the_proxy_error_and_never_polls(cli_runner, status_code, detail):
    responses.add(responses.POST, f"{CONNECT_URL}/start", status=status_code, json={"detail": detail})

    result = _run_connect(cli_runner)

    assert result.exit_code != 0
    assert f"HTTP {status_code}" in result.output
    assert detail in result.output
    assert not any(call.request.url.endswith("/poll") for call in responses.calls)


@responses.activate
def test_connect_waits_the_interval_github_asks_for_and_backs_off_on_slow_down():
    responses.add(responses.POST, f"{CONNECT_URL}/start", json=_start_body(interval=5))
    for poll in (
        {"status": "slow_down"},
        {"status": "pending"},
        {"status": "slow_down", "interval": 20},
        {"status": "connected", "github_login": "octocat"},
    ):
        responses.add(responses.POST, f"{CONNECT_URL}/poll", json=poll)
    clock = _FakeClock()

    result = connect_user_credential(
        CredentialsManagementClient(PROXY, "sk-user"), "team/copilot", clock.as_poll_clock()
    )

    assert result.status == "connected"
    assert clock.waits == (5, 10, 10, 20)


@responses.activate
def test_connect_stops_polling_once_the_next_poll_would_land_after_the_code_expires():
    responses.add(responses.POST, f"{CONNECT_URL}/start", json=_start_body(interval=5, expires_in=19))
    for _ in range(3):
        responses.add(responses.POST, f"{CONNECT_URL}/poll", json={"status": "pending"})
    responses.add(responses.POST, f"{CONNECT_URL}/poll", status=400, json={"detail": "invalid or expired flow_handle"})
    clock = _FakeClock()

    result = connect_user_credential(
        CredentialsManagementClient(PROXY, "sk-user"), "team/copilot", clock.as_poll_clock()
    )

    assert result.status == "expired"
    assert clock.waits == (5, 5, 5)
    assert sum(call.request.url.endswith("/poll") for call in responses.calls) == 3


@responses.activate
def test_connect_still_collects_an_approval_on_the_last_poll_before_the_code_expires():
    responses.add(responses.POST, f"{CONNECT_URL}/start", json=_start_body(interval=5, expires_in=19))
    responses.add(responses.POST, f"{CONNECT_URL}/poll", json={"status": "pending"})
    responses.add(responses.POST, f"{CONNECT_URL}/poll", json={"status": "pending"})
    responses.add(responses.POST, f"{CONNECT_URL}/poll", json={"status": "connected", "github_login": "octocat"})
    clock = _FakeClock()

    result = connect_user_credential(
        CredentialsManagementClient(PROXY, "sk-user"), "team/copilot", clock.as_poll_clock()
    )

    assert result.status == "connected"
    assert clock.elapsed == 15


@responses.activate
def test_connect_reports_a_lost_connection_while_waiting_and_asks_to_rerun(cli_runner):
    responses.add(responses.POST, f"{CONNECT_URL}/start", json=_start_body())
    responses.add(responses.POST, f"{CONNECT_URL}/poll", body=requests.exceptions.ConnectionError("connection reset"))

    result = _run_connect(cli_runner)

    assert result.exit_code == 1
    assert "ABCD-1234" in result.output
    assert f"could not reach the proxy at {PROXY} (ConnectionError). Run the command again" in result.output
