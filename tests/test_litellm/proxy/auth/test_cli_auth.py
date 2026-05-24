"""
Tests for litellm/proxy/client/cli/commands/auth.py

This module tests the auth commands and their associated functionality.
"""

import pytest
import requests
from unittest.mock import patch, Mock, call
from litellm.proxy.client.cli.commands.auth import (
    _poll_for_ready_data,
    _poll_for_authentication,
    _start_cli_sso_flow,
)


@patch("litellm.proxy.client.cli.commands.auth.requests.post")
def test_start_cli_sso_flow_rejects_invalid_response(request_mock):
    """Test CLI SSO start rejects malformed server responses"""
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {"login_id": "cli-session", "user_code": "ABCD-EFGH"}
    request_mock.return_value = response

    with pytest.raises(ValueError, match="Invalid CLI SSO start response"):
        _start_cli_sso_flow("https://litellm.com")


@pytest.mark.asyncio
@patch(
    "litellm.proxy.client.cli.commands.auth.requests.get",
    side_effect=[Mock(status_code=404)],
)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_404(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data(
        "https://litellm.com", poll_interval=1, total_timeout=1, request_timeout=42
    )
    assert actual is None
    click_mock.assert_called_once_with("Polling error: HTTP 404")
    request_mock.assert_called_once_with("https://litellm.com", timeout=42)


@pytest.mark.asyncio
@patch(
    "litellm.proxy.client.cli.commands.auth.requests.get",
    side_effect=[
        Mock(
            status_code=200, json=Mock(return_value={"status": "ready", "json": "data"})
        )
    ],
)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_200_ready(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data(
        "https://litellm.com", poll_interval=1, total_timeout=1, request_timeout=42
    )
    assert actual == {"status": "ready", "json": "data"}
    click_mock.assert_not_called()
    request_mock.assert_called_once_with("https://litellm.com", timeout=42)
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
@patch(
    "litellm.proxy.client.cli.commands.auth.requests.get",
    side_effect=[
        Mock(
            status_code=200,
            json=Mock(return_value={"status": "pending", "json": "data"}),
        ),
        Mock(
            status_code=200, json=Mock(return_value={"status": "ready", "json": "data"})
        ),
    ],
)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_single_pending(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data(
        "https://litellm.com", poll_interval=1, total_timeout=2, request_timeout=42
    )
    assert actual == {"status": "ready", "json": "data"}
    click_mock.assert_not_called()
    request_mock.assert_has_calls(
        [
            call("https://litellm.com", timeout=42),
            call("https://litellm.com", timeout=42),
        ]
    )
    sleep_mock.assert_called_once_with(1)


@pytest.mark.asyncio
@patch(
    "litellm.proxy.client.cli.commands.auth.requests.get",
    side_effect=[
        Mock(
            status_code=200,
            json=Mock(return_value={"status": "pending", "json": "data"}),
        ),
        Mock(
            status_code=200,
            json=Mock(return_value={"status": "pending", "json": "data"}),
        ),
    ],
)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_pending(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data(
        "https://litellm.com",
        poll_interval=1,
        total_timeout=2,
        request_timeout=42,
        pending_message="Pending message",
        pending_log_every=1,
    )
    assert actual is None
    click_mock.assert_has_calls([call("Pending message"), call("Pending message")])
    request_mock.assert_has_calls(
        [
            call("https://litellm.com", timeout=42),
            call("https://litellm.com", timeout=42),
        ]
    )
    sleep_mock.assert_has_calls([call(1), call(1)])


@pytest.mark.asyncio
@patch(
    "litellm.proxy.client.cli.commands.auth.requests.get",
    side_effect=[
        requests.RequestException("ERROR"),
        requests.RequestException("ERROR"),
    ],
)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_connection_failure(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data(
        "https://litellm.com", poll_interval=1, total_timeout=2, request_timeout=42
    )
    assert actual is None
    click_mock.assert_called_once_with("Connection error (will retry): ERROR")
    request_mock.assert_has_calls(
        [
            call("https://litellm.com", timeout=42),
        ]
    )
    sleep_mock.assert_has_calls([call(1), call(1)])


@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth._poll_for_ready_data", return_value=None)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
async def test_poll_for_authentication_no_data(click_mock, poll_mock):
    """Test poll_for_authentication function"""
    actual = _poll_for_authentication("https://litellm.com", "key-123", "poll-secret")
    assert actual is None
    poll_mock.assert_called_once_with(
        "https://litellm.com/sso/cli/poll/key-123",
        headers={"x-litellm-cli-poll-secret": "poll-secret"},
        poll_interval=1,
        request_timeout=5,
        pending_message="Still waiting for authentication...",
        pending_log_every=30,
    )
    click_mock.assert_not_called()


@pytest.mark.asyncio
@patch(
    "litellm.proxy.client.cli.commands.auth._poll_for_ready_data",
    return_value={
        "key": "jwt-456",
        "user_id": "user-456",
        "teams": ["team-a", "team-b"],
        "team_id": "team-a",
    },
)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
async def test_poll_for_authentication_returns_jwt(click_mock, poll_mock):
    """Test poll_for_authentication returns JWT from a single poll"""
    actual = _poll_for_authentication("https://litellm.com", "key-123", "poll-secret")
    assert actual == {
        "api_key": "jwt-456",
        "user_id": "user-456",
        "teams": ["team-a", "team-b"],
        "team_id": "team-a",
    }
    poll_mock.assert_called_once()
    click_mock.assert_not_called()
