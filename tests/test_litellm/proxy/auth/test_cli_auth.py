"""
Tests for litellm/proxy/client/cli/commands/auth.py

This module tests the auth commands and their associated functionality.
"""

import pytest
import requests
from unittest.mock import AsyncMock, patch, Mock, call
from litellm.proxy.client.cli.commands.auth import _normalize_teams, _poll_for_ready_data, _poll_for_authentication

@pytest.mark.asyncio
async def test_normalize_teams_teams_only():
    """Test normalize teams helper function"""
    teams = ["1", "2", "3"]
    team_details = []
    result = _normalize_teams(teams, team_details)
    assert result == [{"team_id": "1", "team_alias": None}, {"team_id": "2", "team_alias": None}, {"team_id": "3", "team_alias": None}]

@pytest.mark.asyncio
async def test_normalize_teams_with_details_no_aliases():
    """Test normalize teams helper function"""
    teams = ["4", "5", "6"]
    team_details = [{"team_id": "1"}, {"team_id": "2"}, {"team_id": "3"}]
    result = _normalize_teams(teams, team_details)
    assert result == [{"team_id": "1", "team_alias": None}, {"team_id": "2", "team_alias": None}, {"team_id": "3", "team_alias": None}]

@pytest.mark.asyncio
async def test_normalize_teams_with_details_with_aliases():
    """Test normalize teams helper function"""
    teams = ["4", "5", "6"]
    team_details = [{"team_id": "1", "team_alias": "A"}, {"team_id": "2", "team_alias": "B"}, {"team_id": "3", "team_alias": "C"}]
    result = _normalize_teams(teams, team_details)
    assert result == [{"team_id": "1", "team_alias": "A"}, {"team_id": "2", "team_alias": "B"}, {"team_id": "3", "team_alias": "C"}]

@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth.requests.get", side_effect=[Mock(status_code=404)])
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_404(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data("https://litellm.com", poll_interval=1, total_timeout=1, request_timeout=42)
    assert actual is None
    click_mock.assert_called_once_with("Polling error: HTTP 404")
    request_mock.assert_called_once_with("https://litellm.com", timeout=42)

@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth.requests.get", side_effect=[Mock(status_code=200, json=Mock(return_value={"status": "ready","json": "data"}))])
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_200_ready(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data("https://litellm.com", poll_interval=1, total_timeout=1, request_timeout=42)
    assert actual == {"status": "ready", "json": "data"}
    click_mock.assert_not_called()
    request_mock.assert_called_once_with("https://litellm.com", timeout=42)
    sleep_mock.assert_not_called()

@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth.requests.get", side_effect=[Mock(status_code=200, json=Mock(return_value={"status": "pending","json": "data"})), Mock(status_code=200, json=Mock(return_value={"status": "ready","json": "data"}))])
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_single_pending(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data("https://litellm.com", poll_interval=1, total_timeout=2, request_timeout=42)
    assert actual == {"status": "ready", "json": "data"}
    click_mock.assert_not_called()
    request_mock.assert_has_calls([
        call("https://litellm.com", timeout=42),
        call("https://litellm.com", timeout=42)
    ])
    sleep_mock.assert_called_once_with(1)

@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth.requests.get", side_effect=[Mock(status_code=200, json=Mock(return_value={"status": "pending","json": "data"})), Mock(status_code=200, json=Mock(return_value={"status": "pending","json": "data"}))])
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_pending(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data("https://litellm.com", poll_interval=1, total_timeout=2, request_timeout=42, pending_message="Pending message", pending_log_every=1)
    assert actual is None
    click_mock.assert_has_calls([
        call("Pending message"),
        call("Pending message")
    ])
    request_mock.assert_has_calls([
        call("https://litellm.com", timeout=42),
        call("https://litellm.com", timeout=42)
    ])
    sleep_mock.assert_has_calls([
        call(1),
        call(1)
    ])


@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth.requests.get", side_effect=[requests.RequestException("ERROR"),
                                                                           requests.RequestException("ERROR")])
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
@patch("litellm.proxy.client.cli.commands.auth.time.sleep")
async def test_poll_for_ready_connection_failure(sleep_mock, click_mock, request_mock):
    """Test poll_for_ready function"""
    actual = _poll_for_ready_data("https://litellm.com", poll_interval=1, total_timeout=2, request_timeout=42)
    assert actual is None
    click_mock.assert_called_once_with("Connection error (will retry): ERROR")
    request_mock.assert_has_calls([
        call("https://litellm.com", timeout=42),
    ])
    sleep_mock.assert_has_calls([
        call(1),
        call(1)
    ])


@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth._handle_team_selection_during_polling")
@patch("litellm.proxy.client.cli.commands.auth._poll_for_ready_data", return_value=None)
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
async def test_poll_for_authentication_no_data(click_mock, poll_mock, handle_mock):
    """Test poll_for_authentication function"""
    actual = _poll_for_authentication("https://litellm.com", "key-123")
    assert actual is None
    poll_mock.assert_called_once_with(
        "https://litellm.com/sso/cli/poll/key-123",
        pending_message="Still waiting for authentication...",
    )
    handle_mock.assert_not_called()
    click_mock.assert_not_called()


@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth._handle_team_selection_during_polling")
@patch("litellm.proxy.client.cli.commands.auth._poll_for_ready_data", return_value={"requires_team_selection": True, "teams": [], "team_details": []})
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
async def test_poll_for_authentication_no_teams(click_mock, poll_mock, handle_mock):
    """Test poll_for_authentication function"""
    actual = _poll_for_authentication("https://litellm.com", "key-123")
    assert actual is None
    poll_mock.assert_called_once_with(
        "https://litellm.com/sso/cli/poll/key-123",
        pending_message="Still waiting for authentication...",
    )
    handle_mock.assert_not_called()
    click_mock.assert_called_once()
    assert "No teams available for selection." in click_mock.call_args[0][0]


@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth._handle_team_selection_during_polling", return_value="jwt-123")
@patch("litellm.proxy.client.cli.commands.auth._poll_for_ready_data", return_value={"requires_team_selection": True, "teams": [1, 2], "user_id": "user-123"})
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
async def test_poll_for_authentication_team_selection_success(click_mock, poll_mock, handle_mock):
    """Test poll_for_authentication function"""
    actual = _poll_for_authentication("https://litellm.com", "key-123")
    assert actual == {"api_key": "jwt-123", "user_id": "user-123", "teams": [1, 2], "team_id": None}
    poll_mock.assert_called_once_with(
        "https://litellm.com/sso/cli/poll/key-123",
        pending_message="Still waiting for authentication...",
    )
    handle_mock.assert_called_once_with(
        base_url="https://litellm.com",
        key_id="key-123",
        teams=[{"team_id": "1", "team_alias": None}, {"team_id": "2", "team_alias": None}],
    )
    click_mock.assert_not_called()


@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth._handle_team_selection_during_polling", return_value=None)
@patch("litellm.proxy.client.cli.commands.auth._poll_for_ready_data", return_value={"requires_team_selection": True, "teams": ["team-1"], "user_id": "user-123"})
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
async def test_poll_for_authentication_team_selection_cancelled(click_mock, poll_mock, handle_mock):
    """Test poll_for_authentication function"""
    actual = _poll_for_authentication("https://litellm.com", "key-123")
    assert actual is None
    poll_mock.assert_called_once_with(
        "https://litellm.com/sso/cli/poll/key-123",
        pending_message="Still waiting for authentication...",
    )
    handle_mock.assert_called_once_with(
        base_url="https://litellm.com",
        key_id="key-123",
        teams=[{"team_id": "team-1", "team_alias": None}],
    )
    click_mock.assert_called_once()
    assert "Team selection cancelled" in click_mock.call_args[0][0]


@pytest.mark.asyncio
@patch("litellm.proxy.client.cli.commands.auth._handle_team_selection_during_polling")
@patch("litellm.proxy.client.cli.commands.auth._poll_for_ready_data", return_value={"key": "jwt-456", "user_id": "user-456", "teams": ["team-1"], "team_id": "team-1"})
@patch("litellm.proxy.client.cli.commands.auth.click.echo")
async def test_poll_for_authentication_auto_assigned_team(click_mock, poll_mock, handle_mock):
    """Test poll_for_authentication function"""
    actual = _poll_for_authentication("https://litellm.com", "key-123")
    assert actual == {"api_key": "jwt-456", "user_id": "user-456", "teams": ["team-1"], "team_id": "team-1"}
    poll_mock.assert_called_once_with(
        "https://litellm.com/sso/cli/poll/key-123",
        pending_message="Still waiting for authentication...",
    )
    handle_mock.assert_not_called()
    click_mock.assert_called_once()
    assert "Automatically assigned to team: team-1" in click_mock.call_args[0][0]
