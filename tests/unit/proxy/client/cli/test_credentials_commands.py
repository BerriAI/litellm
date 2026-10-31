import json
from collections.abc import Iterator
from typing import Final
from unittest.mock import MagicMock

import pytest
import requests
import responses
from click.testing import CliRunner

from litellm.proxy.client.cli.commands.credentials import (
    UserConnectionConnected,
    UserConnectionFailed,
    run_user_connection,
)
from litellm.proxy.client.cli.main import cli
from litellm.proxy.client.credentials import UserConnectionPoll, UserConnectionStart


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
    mock_instance.create.side_effect = requests.exceptions.HTTPError(
        response=mock_error_response
    )

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
    mock_instance.delete.side_effect = requests.exceptions.HTTPError(
        response=mock_error_response
    )

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


class _SequenceUserConnectionClient:
    def __init__(
        self,
        start: UserConnectionStart,
        poll_responses: tuple[UserConnectionPoll, ...],
    ) -> None:
        self._start = start
        self._poll_responses: Iterator[UserConnectionPoll] = iter(poll_responses)

    def start_user_connection(self, credential_name: str) -> UserConnectionStart:
        return self._start

    def poll_user_connection(self, credential_name: str, flow_handle: str) -> UserConnectionPoll:
        return next(self._poll_responses)


class _StringRecorder:
    def __init__(self) -> None:
        self.values: tuple[str, ...] = ()

    def __call__(self, value: str) -> None:
        self.values = (*self.values, value)


class _SleepRecorder:
    def __init__(self) -> None:
        self.values: tuple[float, ...] = ()

    def __call__(self, value: float) -> None:
        self.values = (*self.values, value)


class _AdvancingClock:
    def __init__(self) -> None:
        self.current: float = 0
        self.sleeps: tuple[float, ...] = ()

    def __call__(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds
        self.sleeps = (*self.sleeps, seconds)

    def advance(self, seconds: float) -> None:
        self.current += seconds


class _LateConnectedUserConnectionClient:
    def __init__(self, clock: _AdvancingClock) -> None:
        self._clock = clock

    def start_user_connection(self, _credential_name: str) -> UserConnectionStart:
        return _start_response(interval=0, expires_in=3)

    def poll_user_connection(self, _credential_name: str, _flow_handle: str) -> UserConnectionPoll:
        self._clock.advance(4)
        return UserConnectionPoll(status="connected", github_login="octocat")


def _start_response(
    verification_uri: str = "https://github.com/login/device",
    interval: int = 5,
    expires_in: int = 900,
) -> UserConnectionStart:
    return UserConnectionStart(
        user_code="ABCD-1234",
        verification_uri=verification_uri,
        expires_in=expires_in,
        interval=interval,
        flow_handle="flow-handle",
    )


def _cli_args(*args: str) -> tuple[str, ...]:
    return ("--base-url", "http://fake-proxy", "--api-key", "sk-user", *args)


def _register_start_response(credential_name: str = "copilot", expires_in: int = 900) -> None:
    responses.add(
        responses.POST,
        f"http://fake-proxy/credentials/{credential_name}/user_connection/start",
        json={
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "expires_in": expires_in,
            "interval": 0,
            "flow_handle": "flow-handle",
        },
        status=200,
    )


@pytest.mark.parametrize(
    ("expires_in", "expected_expiry"),
    [(899, "15 minutes"), (60, "1 minute"), (20, "1 minute")],
)
@responses.activate
def test_connect_command_polls_until_connected(
    cli_runner: CliRunner,
    expires_in: int,
    expected_expiry: str,
) -> None:
    _register_start_response(expires_in=expires_in)
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/poll",
        json={"status": "pending"},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/poll",
        json={"status": "slow_down", "interval": 0},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/poll",
        json={"status": "connected", "github_login": "octocat"},
        status=200,
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 0, result.output
    assert "ABCD-1234" in result.output
    assert "https://github.com/login/device" in result.output
    assert f"The code expires in {expected_expiry}" in result.output
    assert "Connected GitHub Copilot for credential 'copilot' as octocat" in result.output


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (
            "expired",
            "The device code expired before it was approved. "
            "Run `lite credentials connect copilot` again to get a new code",
        ),
        (
            "denied",
            "The request was denied on GitHub. Run `lite credentials connect copilot` again if this was a mistake",
        ),
        (
            "no_copilot_seat",
            "This GitHub account does not have GitHub Copilot access, so nothing was saved",
        ),
    ],
)
@responses.activate
def test_connect_command_reports_terminal_poll_statuses(
    cli_runner: CliRunner,
    status: str,
    message: str,
) -> None:
    _register_start_response()
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/poll",
        json={"status": status},
        status=200,
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 1
    assert result.output.splitlines()[-1] == f"Error: {message}"


@responses.activate
def test_connect_command_reports_proxy_connection_error(cli_runner: CliRunner) -> None:
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/start",
        body=requests.exceptions.ConnectionError("connection refused"),
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 1
    assert "Could not connect to the LiteLLM proxy" in result.output


@responses.activate
def test_connect_command_prints_unauthorized_detail(cli_runner: CliRunner) -> None:
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/start",
        json={"detail": "An authenticated user is required"},
        status=401,
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 1
    assert "Error: HTTP 401: An authenticated user is required" in result.output


@responses.activate
def test_connect_command_prints_proxy_404_error_message(cli_runner: CliRunner) -> None:
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/start",
        json={
            "error": {
                "message": "Credential not found",
                "type": "internal_server_error",
                "param": None,
                "code": "404",
            }
        },
        status=404,
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 1
    assert result.output.splitlines()[-1] == "Error: HTTP 404: Credential not found"


@responses.activate
def test_connect_command_prints_proxy_401_error_message(cli_runner: CliRunner) -> None:
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/start",
        json={
            "error": {
                "message": "Authentication Error, Invalid API key provided.",
                "type": "token_not_found_in_db",
                "param": "key",
                "code": "401",
            }
        },
        status=401,
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 1
    assert result.output.splitlines()[-1] == "Error: HTTP 401: Authentication Error, Invalid API key provided."


@responses.activate
def test_connect_command_prints_nested_fastapi_error_detail(cli_runner: CliRunner) -> None:
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/start",
        json={"detail": {"error": "Invalid flow details"}},
        status=400,
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 1
    assert result.output.splitlines()[-1] == "Error: HTTP 400: Invalid flow details"


@responses.activate
def test_connect_command_reports_invalid_flow_handle(cli_runner: CliRunner) -> None:
    _register_start_response()
    responses.add(
        responses.POST,
        "http://fake-proxy/credentials/copilot/user_connection/poll",
        json={"detail": "Invalid flow handle"},
        status=400,
    )

    result: Final = cli_runner.invoke(cli, _cli_args("credentials", "connect", "copilot", "--no-browser"))

    assert result.exit_code == 1
    assert "Error: HTTP 400: Invalid flow handle" in result.output


@pytest.mark.parametrize("output_format", ["table", "json"])
@responses.activate
def test_connections_command_lists_per_user_connections(cli_runner: CliRunner, output_format: str) -> None:
    response: Final = {
        "connections": [
            {
                "credential_name": "copilot-prod",
                "provider": "github_copilot",
                "connected": True,
                "github_login": "octocat",
                "connected_at": "2025-01-19T12:00:00+00:00",
            },
            {
                "credential_name": "copilot-dev",
                "provider": "github_copilot",
                "connected": False,
                "github_login": None,
                "connected_at": None,
            },
        ]
    }
    responses.add(
        responses.GET,
        "http://fake-proxy/credentials/user_connections",
        json=response,
        status=200,
    )

    result: Final = cli_runner.invoke(
        cli,
        _cli_args("credentials", "connections", "--format", output_format),
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    if output_format == "json":
        assert json.loads(result.output) == response
    else:
        assert "GitHub Copilot connections" in result.output
        assert "Credential Name" in result.output
        assert "Connected" in result.output
        assert "GitHub Login" in result.output
        assert "Connected At" in result.output
        prod_row: Final = next((line for line in result.output.splitlines() if "copilot-prod" in line), "")
        dev_row: Final = next((line for line in result.output.splitlines() if "copilot-dev" in line), "")
        assert "yes" in prod_row
        assert "octocat" in prod_row
        assert "2025-01-19T12:00:00+00:00" in prod_row
        assert "no" in dev_row


@responses.activate
def test_disconnect_command_encodes_credential_name(cli_runner: CliRunner) -> None:
    credential_name: Final = "my/ credential"
    responses.add(
        responses.DELETE,
        "http://fake-proxy/credentials/my%2F%20credential/user_connection",
        json={"status": "disconnected"},
        status=200,
    )

    result: Final = cli_runner.invoke(
        cli,
        _cli_args("credentials", "disconnect", credential_name),
    )

    assert result.exit_code == 0, result.output
    assert "Disconnected GitHub Copilot for credential 'my/ credential'" in result.output
    assert responses.calls[0].request.url == "http://fake-proxy/credentials/my%2F%20credential/user_connection"


def test_run_user_connection_uses_response_intervals_and_slow_down_fallback() -> None:
    client: Final = _SequenceUserConnectionClient(
        _start_response(interval=5),
        (
            UserConnectionPoll(status="pending"),
            UserConnectionPoll(status="slow_down", interval=10),
            UserConnectionPoll(status="slow_down"),
            UserConnectionPoll(status="connected", github_login="octocat"),
        ),
    )
    sleep_calls: Final = _SleepRecorder()
    output: Final = _StringRecorder()

    result: Final = run_user_connection(
        client,
        "copilot",
        open_browser=False,
        launch_browser=lambda uri: None,
        sleep=sleep_calls,
        clock=lambda: 0,
        echo=output,
    )

    assert result == UserConnectionConnected(status="connected", github_login="octocat")
    assert sleep_calls.values == (5, 5, 10, 15)
    assert output.values[-1] == "Connected GitHub Copilot for credential 'copilot' as octocat"


def test_run_user_connection_honors_server_poll_interval() -> None:
    sleep_calls: Final = _SleepRecorder()
    client: Final = _SequenceUserConnectionClient(
        _start_response(),
        (
            UserConnectionPoll(status="slow_down", interval=9),
            UserConnectionPoll(status="connected", github_login="octocat"),
        ),
    )

    result: Final = run_user_connection(
        client,
        "copilot",
        open_browser=False,
        launch_browser=lambda uri: None,
        sleep=sleep_calls,
        clock=lambda: 0,
        echo=_StringRecorder(),
    )

    assert result == UserConnectionConnected(status="connected", github_login="octocat")
    assert sleep_calls.values == (5, 9)


def test_run_user_connection_stops_at_local_expiry_deadline() -> None:
    clock: Final = _AdvancingClock()
    client: Final = _SequenceUserConnectionClient(
        _start_response(interval=2, expires_in=3),
        (UserConnectionPoll(status="pending"), UserConnectionPoll(status="pending")),
    )

    result: Final = run_user_connection(
        client,
        "copilot",
        open_browser=False,
        launch_browser=lambda uri: None,
        sleep=clock.sleep,
        clock=clock,
        echo=lambda message: None,
    )

    assert result == UserConnectionFailed(
        status="failed",
        reason="expired",
        message=(
            "The device code expired before it was approved. "
            "Run `lite credentials connect copilot` again to get a new code"
        ),
    )
    assert clock.sleeps == (2, 2)


def test_run_user_connection_accepts_connected_poll_received_after_deadline() -> None:
    clock: Final = _AdvancingClock()
    client: Final = _LateConnectedUserConnectionClient(clock)

    result: Final = run_user_connection(
        client,
        "copilot",
        open_browser=False,
        launch_browser=lambda uri: None,
        sleep=clock.sleep,
        clock=clock,
        echo=lambda message: None,
    )

    assert result == UserConnectionConnected(status="connected", github_login="octocat")


def test_run_user_connection_opens_safe_url_only_when_requested() -> None:
    launches: Final = _StringRecorder()
    client: Final = _SequenceUserConnectionClient(
        _start_response(),
        (UserConnectionPoll(status="connected", github_login="octocat"),),
    )

    result: Final = run_user_connection(
        client,
        "copilot",
        open_browser=True,
        launch_browser=launches,
        sleep=lambda seconds: None,
        clock=lambda: 0,
        echo=lambda message: None,
    )

    assert result == UserConnectionConnected(status="connected", github_login="octocat")
    assert launches.values == ("https://github.com/login/device",)


def test_run_user_connection_does_not_launch_browser_when_disabled() -> None:
    launches: Final = _StringRecorder()
    client: Final = _SequenceUserConnectionClient(
        _start_response(),
        (UserConnectionPoll(status="connected", github_login="octocat"),),
    )

    run_user_connection(
        client,
        "copilot",
        open_browser=False,
        launch_browser=launches,
        sleep=lambda seconds: None,
        clock=lambda: 0,
        echo=lambda message: None,
    )

    assert launches.values == ()


@pytest.mark.parametrize(
    "verification_uri",
    [
        "http://github.com/login/device",
        "https://github.com:443/login/device",
        "https://user@github.com/login/device",
        "https://github.com.evil.com/login/device",
    ],
)
def test_run_user_connection_replaces_unsafe_verification_uri(verification_uri: str) -> None:
    launches: Final = _StringRecorder()
    output: Final = _StringRecorder()
    client: Final = _SequenceUserConnectionClient(
        _start_response(verification_uri=verification_uri, interval=0),
        (UserConnectionPoll(status="connected", github_login="octocat"),),
    )

    run_user_connection(
        client,
        "copilot",
        open_browser=True,
        launch_browser=launches,
        sleep=lambda seconds: None,
        clock=lambda: 0,
        echo=output,
    )

    assert launches.values == ("https://github.com/login/device",)
    assert "https://github.com/login/device" in output.values[0]
    assert verification_uri not in output.values[0]
