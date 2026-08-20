import json
import os
import stat
import sys
import time
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path


import pytest
from click.testing import CliRunner

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.auth import (
    clear_token,
    get_stored_api_key,
    get_token_file_path,
    load_token,
    login,
    logout,
    print_token,
    save_token,
    whoami,
)
from litellm.proxy.client.cli.commands.claude_settings import SettingsFileOwner


def _mock_cli_sso_start_response(
    login_id: str = "cli-session-uuid-456",
    poll_secret: str = "poll-secret",
    user_code: str = "ABCD-EFGH",
) -> Mock:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "login_id": login_id,
        "poll_secret": poll_secret,
        "user_code": user_code,
    }
    mock_response.raise_for_status = Mock()
    return mock_response


class TestPollingErrorSurfacing:
    def test_client_error_raises_with_server_detail_and_stops_polling(self):
        from litellm.proxy.client.cli.commands.auth import _poll_for_ready_data

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "detail": "Your litellm CLI is out of date and uses a login flow this proxy no longer supports."
        }

        with patch("requests.get", return_value=mock_response) as mock_get, patch("time.sleep"):
            with pytest.raises(ValueError) as exc_info:
                _poll_for_ready_data("http://test/sso/cli/poll/sk-legacy")

        assert mock_get.call_count == 1
        assert (
            "The proxy rejected the login session with HTTP 400: Your litellm CLI is out of date "
            "and uses a login flow this proxy no longer supports." in str(exc_info.value)
        )

    def test_login_command_shows_server_rejection_to_user(self):
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        mock_poll_response = Mock()
        mock_poll_response.status_code = 400
        mock_poll_response.json.return_value = {"detail": "CLI login session not found or expired."}

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", return_value=mock_poll_response),
            patch("time.sleep"),
        ):
            result = CliRunner().invoke(login, obj=mock_context.obj)

        assert result.exit_code == 0
        assert "Authentication failed:" in result.output
        assert "CLI login session not found or expired." in result.output
        assert "Authentication timed out" not in result.output

    def test_server_error_without_json_body_retries_until_timeout(self, capsys):
        from litellm.proxy.client.cli.commands.auth import _poll_for_ready_data

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("no json")

        with patch("requests.get", return_value=mock_response) as mock_get, patch("time.sleep"):
            result = _poll_for_ready_data("http://test/sso/cli/poll/cli-abc", total_timeout=6, poll_interval=2)

        assert result is None
        assert mock_get.call_count == 3
        assert "Polling error: HTTP 500" in capsys.readouterr().out

    def test_rate_limit_is_retried_not_aborted(self, capsys):
        from litellm.proxy.client.cli.commands.auth import _poll_for_ready_data

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"detail": "Too many CLI login attempts. Try again later."}

        with patch("requests.get", return_value=mock_response) as mock_get, patch("time.sleep"):
            result = _poll_for_ready_data("http://test/sso/cli/poll/cli-abc", total_timeout=4, poll_interval=2)

        assert result is None
        assert mock_get.call_count == 2
        assert "Polling error: HTTP 429: Too many CLI login attempts. Try again later." in capsys.readouterr().out


class TestStartCliSsoFlowErrors:
    def test_endpoint_not_found_explains_version_or_base_url(self):
        from litellm.proxy.client.cli.commands.auth import _start_cli_sso_flow

        mock_response = Mock()
        mock_response.status_code = 404

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                _start_cli_sso_flow("https://old-proxy.example.com")

        message = str(exc_info.value)
        assert "HTTP 404" in message
        assert "--base-url" in message
        assert "older than this CLI" in message

    def test_http_error_includes_server_detail(self):
        from litellm.proxy.client.cli.commands.auth import _start_cli_sso_flow

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"detail": "Too many CLI login attempts. Try again later."}

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                _start_cli_sso_flow("https://test.example.com")

        assert "HTTP 429" in str(exc_info.value)
        assert "Too many CLI login attempts. Try again later." in str(exc_info.value)

    def test_non_json_response_names_interception(self):
        from litellm.proxy.client.cli.commands.auth import _start_cli_sso_flow

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("no json")
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>Sign in to corporate VPN</html>"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(ValueError) as exc_info:
                _start_cli_sso_flow("https://test.example.com")

        message = str(exc_info.value)
        assert "non-JSON response" in message
        assert "text/html" in message
        assert "Sign in to corporate VPN" in message

    def test_connection_error_points_at_base_url(self):
        import requests

        from litellm.proxy.client.cli.commands.auth import _start_cli_sso_flow

        with patch("requests.post", side_effect=requests.ConnectionError("Connection refused")):
            with pytest.raises(ValueError) as exc_info:
                _start_cli_sso_flow("https://unreachable.example.com")

        message = str(exc_info.value)
        assert "Could not reach the proxy" in message
        assert "https://unreachable.example.com/sso/cli/start" in message


class TestTokenUtilities:
    """Test token file utility functions"""

    def test_get_token_file_path(self):
        """Test getting token file path"""
        with (
            patch("pathlib.Path.home") as mock_home,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            mock_home.return_value = Path("/home/user")

            result = get_token_file_path()

            assert result == "/home/user/.litellm/token.json"
            mock_mkdir.assert_not_called()

    def test_reading_the_token_never_creates_the_config_directory(self, tmp_path):
        """Every `lite` invocation reads the token; only saving one may touch ~/.litellm"""
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert load_token() is None
            assert not (tmp_path / ".litellm").exists()
            save_token({"key": "sk-test"})
            assert load_token() == {"key": "sk-test"}

    def test_save_token(self, tmp_path):
        """Test saving token data to file"""
        token_data = {
            "key": "test-key",
            "user_id": "test-user",
            "timestamp": 1234567890,
        }
        token_file = tmp_path / "token.json"

        with patch("litellm.proxy.client.cli.commands.auth.get_token_file_path") as mock_path:
            mock_path.return_value = str(token_file)

            save_token(token_data)

        assert json.loads(token_file.read_text()) == token_data
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600

    def test_load_token_success(self):
        """Test loading token data from file successfully"""
        token_data = {
            "key": "test-key",
            "user_id": "test-user",
            "timestamp": 1234567890,
        }

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(token_data))),
            patch("litellm.proxy.client.cli.commands.auth.get_token_file_path") as mock_path,
            patch("os.path.exists", return_value=True),
        ):
            mock_path.return_value = "/test/path/token.json"

            result = load_token()

            assert result == token_data

    def test_load_token_file_not_exists(self):
        """Test loading token when file doesn't exist"""
        with (
            patch("litellm.proxy.client.cli.commands.auth.get_token_file_path") as mock_path,
            patch("os.path.exists", return_value=False),
        ):
            mock_path.return_value = "/test/path/token.json"

            result = load_token()

            assert result is None

    def test_load_token_json_decode_error(self):
        """Test loading token with invalid JSON"""
        with (
            patch("builtins.open", mock_open(read_data="invalid json")),
            patch("litellm.proxy.client.cli.commands.auth.get_token_file_path") as mock_path,
            patch("os.path.exists", return_value=True),
        ):
            mock_path.return_value = "/test/path/token.json"

            result = load_token()

            assert result is None

    def test_load_token_io_error(self):
        """Test loading token with IO error"""
        with (
            patch("builtins.open", side_effect=OSError("Permission denied")),
            patch("litellm.proxy.client.cli.commands.auth.get_token_file_path") as mock_path,
            patch("os.path.exists", return_value=True),
        ):
            mock_path.return_value = "/test/path/token.json"

            result = load_token()

            assert result is None

    def test_clear_token_file_exists(self):
        """Test clearing token when file exists"""
        with (
            patch("litellm.proxy.client.cli.commands.auth.get_token_file_path") as mock_path,
            patch("os.path.exists", return_value=True),
            patch("os.remove") as mock_remove,
        ):
            mock_path.return_value = "/test/path/token.json"

            clear_token()

            mock_remove.assert_called_once_with("/test/path/token.json")

    def test_clear_token_file_not_exists(self):
        """Test clearing token when file doesn't exist"""
        with (
            patch("litellm.proxy.client.cli.commands.auth.get_token_file_path") as mock_path,
            patch("os.path.exists", return_value=False),
            patch("os.remove") as mock_remove,
        ):
            mock_path.return_value = "/test/path/token.json"

            clear_token()

            mock_remove.assert_not_called()

    def test_get_stored_api_key_success(self):
        """Test getting stored API key successfully"""
        token_data = {"key": "test-api-key-123", "user_id": "test-user"}

        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value=token_data,
        ):
            result = get_stored_api_key()
            assert result == "test-api-key-123"

    def test_get_stored_api_key_no_token(self):
        """Test getting stored API key when no token exists"""
        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value=None,
        ):
            result = get_stored_api_key()
            assert result is None

    def test_get_stored_api_key_no_key_field(self):
        """Test getting stored API key when token has no key field"""
        token_data = {"user_id": "test-user"}

        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value=token_data,
        ):
            result = get_stored_api_key()
            assert result is None

    def test_get_stored_api_key_base_url_match(self):
        """Stored key is returned when expected_base_url matches stored origin"""
        token_data = {"key": "sk-prod", "base_url": "https://real-proxy.com"}
        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value=token_data,
        ):
            assert get_stored_api_key(expected_base_url="https://real-proxy.com") == "sk-prod"

    def test_get_stored_api_key_base_url_match_trailing_slash(self):
        """Trailing slash on expected_base_url is normalised before comparison"""
        token_data = {"key": "sk-prod", "base_url": "https://real-proxy.com"}
        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value=token_data,
        ):
            assert get_stored_api_key(expected_base_url="https://real-proxy.com/") == "sk-prod"

    def test_get_stored_api_key_base_url_mismatch(self):
        """Stored key is NOT returned when expected_base_url differs from stored origin"""
        token_data = {"key": "sk-prod", "base_url": "https://real-proxy.com"}
        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value=token_data,
        ):
            assert get_stored_api_key(expected_base_url="https://evil.com") is None

    def test_get_stored_api_key_old_token_no_base_url(self):
        """Old tokens without a base_url field are rejected when origin check is requested"""
        token_data = {"key": "sk-old-token"}
        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value=token_data,
        ):
            assert get_stored_api_key(expected_base_url="https://real-proxy.com") is None


class TestLoginCommand:
    """Test login CLI command"""

    @pytest.fixture(autouse=True)
    def isolated_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        return tmp_path

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_login_replaces_a_pkce_record_and_revokes_its_refresh_token(self):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt",
            "user_id": "test-user-123",
            "team_id": "team-1",
            "teams": ["team-1"],
        }
        _FakeSession.instances.clear()

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", return_value=mock_response),
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FakeSession),
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as mock_save,
            patch("litellm.proxy.client.cli.interface.show_commands"),
        ):
            result = self.runner.invoke(login, obj={"base_url": "https://test.example.com"})

        assert result.exit_code == 0, result.output
        assert "Login successful!" in result.output
        assert "Could not revoke" not in result.output
        assert mock_save.call_args.args[0]["key"] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt"
        assert _FakeSession.instances[0].posts == [
            (
                f"{PKCE_BASE_URL}/revoke",
                {"token": "llm_srefresh_old", "token_type_hint": "refresh_token", "client_id": "llm_dcrc_abc"},
            )
        ]

    def test_login_success(self):
        """Test successful login flow with single team (JWT generated immediately)"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        # Mock the requests for successful authentication with single team
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt",
            "user_id": "test-user-123",
            "team_id": "team-1",
            "teams": ["team-1"],
        }

        with (
            patch("webbrowser.open") as mock_browser,
            patch(
                "requests.post",
                return_value=_mock_cli_sso_start_response(login_id="cli-test-uuid-123"),
            ) as mock_post,
            patch("requests.get", return_value=mock_response) as mock_get,
            patch("litellm.proxy.client.cli.commands.auth.save_token") as mock_save,
            patch("litellm.proxy.client.cli.interface.show_commands") as mock_show_commands,
        ):
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Login successful!" in result.output
            assert "Automatically assigned to team: team-1" in result.output

            # Verify browser was opened with correct URL
            mock_browser.assert_called_once()
            call_args = mock_browser.call_args[0][0]
            assert "https://test.example.com/sso/key/generate" in call_args
            assert "cli-test-uuid-123" in call_args
            assert "Verification code: ABCD-EFGH" in result.output
            mock_post.assert_called_once()
            mock_get.assert_called()
            assert mock_get.call_args.kwargs["headers"] == {"x-litellm-cli-poll-secret": "poll-secret"}

            # Verify JWT was saved
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert saved_data["key"] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt"
            assert saved_data["user_id"] == "test-user-123"

            # Verify commands were shown
            mock_show_commands.assert_called_once()

    def test_login_timeout(self):
        """Test login timeout scenario"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        # Mock response that never returns ready status
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "pending"}

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", return_value=mock_response),
            patch("time.sleep"),
        ):
            # Mock time.sleep to avoid actual delays in tests
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Authentication timed out" in result.output

    def test_login_http_error(self):
        """Test login with HTTP error"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        # Mock response with HTTP error
        mock_response = Mock()
        mock_response.status_code = 500

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", return_value=mock_response),
            patch("time.sleep"),
        ):
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Authentication timed out" in result.output

    def test_login_request_exception(self):
        """Test login with request exception"""
        import requests

        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch(
                "requests.get",
                side_effect=requests.RequestException("Connection failed"),
            ),
            patch("time.sleep"),
        ):
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Authentication timed out" in result.output

    def test_login_keyboard_interrupt(self):
        """Test login cancelled by user"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", side_effect=KeyboardInterrupt),
        ):
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Authentication cancelled by user" in result.output

    def test_login_no_api_key_in_response(self):
        """Test login when response doesn't contain API key"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        # Mock response without API key
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ready"
            # Missing 'key' field
        }

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", return_value=mock_response),
            patch("time.sleep"),
        ):
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Authentication timed out" in result.output

    def test_login_general_exception(self):
        """Test login with general exception (not requests exception)"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", side_effect=ValueError("Invalid value")),
        ):
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Authentication failed: Invalid value" in result.output


class TestLogoutCommand:
    """Test logout CLI command"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_logout_success(self):
        """Test successful logout"""
        with patch("litellm.proxy.client.cli.commands.auth.clear_token") as mock_clear:
            result = self.runner.invoke(logout)

            assert result.exit_code == 0
            assert "Logged out successfully" in result.output
            mock_clear.assert_called_once()


class TestWhoamiCommand:
    """Test whoami CLI command"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_whoami_authenticated(self):
        """Test whoami when user is authenticated"""
        token_data = {
            "user_email": "test@example.com",
            "user_id": "test-user-123",
            "user_role": "admin",
            "timestamp": time.time() - 3600,  # 1 hour ago
        }

        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=token_data):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Authenticated" in result.output
            assert "test@example.com" in result.output
            assert "test-user-123" in result.output
            assert "admin" in result.output
            assert "Token age: 1.0 hours" in result.output

    def test_whoami_not_authenticated(self):
        """Test whoami when user is not authenticated"""
        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=None):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Not authenticated" in result.output
            assert "Run 'lite login'" in result.output

    def test_whoami_old_token(self):
        """Test whoami with old token showing warning"""
        token_data = {
            "user_email": "test@example.com",
            "user_id": "test-user-123",
            "user_role": "admin",
            "timestamp": time.time() - (25 * 3600),  # 25 hours ago
        }

        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=token_data):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Authenticated" in result.output
            assert "Warning: Token is more than 24 hours old" in result.output

    def test_whoami_missing_fields(self):
        """Test whoami with token missing some fields"""
        token_data = {
            "timestamp": time.time() - 3600
            # Missing user_email, user_id, user_role
        }

        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=token_data):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Authenticated" in result.output
            assert "Unknown" in result.output  # Should show "Unknown" for missing fields

    def test_whoami_pkce_record_shows_the_team_and_when_the_key_renews(self):
        token_data = {
            "user_email": "unknown",
            "user_id": "user-1",
            "user_role": "cli",
            "team_id": "team-alpha",
            "timestamp": time.time() - 25 * 3600,
            "expires_at": time.time() + 2 * 3600,
            "refresh_token": "llm_srefresh_abc",
        }

        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=token_data):
            result = self.runner.invoke(whoami)

        assert result.exit_code == 0
        assert "Team ID: team-alpha" in result.output
        assert "Key expires in: 2.0 hours, renewed on next use" in result.output
        assert "Warning" not in result.output

    def test_whoami_expired_key_without_a_refresh_token_asks_for_a_new_login(self):
        token_data = {
            "user_id": "user-1",
            "timestamp": time.time() - 3600,
            "expires_at": time.time() - 60,
        }

        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=token_data):
            result = self.runner.invoke(whoami)

        assert result.exit_code == 0
        assert "Team ID" not in result.output
        assert "Key expired. Run 'lite login' again" in result.output

    def test_whoami_expired_pkce_record_that_could_not_be_renewed_asks_for_a_new_pkce_login(self):
        token_data = {
            "user_id": "user-1",
            "team_id": "team-alpha",
            "timestamp": time.time() - 3600,
            "expires_at": time.time() - 60,
            "refresh_token": "llm_srefresh_spent",
        }

        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=token_data):
            result = self.runner.invoke(whoami)

        assert result.exit_code == 0
        assert "Key expired. Run 'lite login --pkce' again" in result.output
        assert "renewed on next use" not in result.output

    def test_whoami_no_timestamp(self):
        """Test whoami with token missing timestamp"""
        token_data = {
            "user_email": "test@example.com",
            "user_id": "test-user-123",
            "user_role": "admin",
            # Missing timestamp
        }

        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value=token_data,
            ),
            patch("time.time", return_value=1000),
        ):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Authenticated" in result.output
            # Should calculate age based on timestamp=0
            assert "Token age:" in result.output


class TestCLIKeyRegenerationFlow:
    """Test the end-to-end CLI key regeneration flow from CLI perspective"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_login_with_team_selection_flow(self):
        """Test complete login flow when user has multiple teams - should prompt for selection"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        # Mock first response - requires team selection
        mock_first_response = Mock()
        mock_first_response.status_code = 200
        mock_first_response.json.return_value = {
            "status": "ready",
            "requires_team_selection": True,
            "user_id": "test-user-456",
            "teams": ["team-alpha", "team-beta", "team-gamma"],
            # New richer response with team details including aliases
            "team_details": [
                {"team_id": "team-alpha", "team_alias": "Alpha Team"},
                {"team_id": "team-beta", "team_alias": "Beta Team"},
                {"team_id": "team-gamma", "team_alias": "Gamma Team"},
            ],
        }

        # Mock second response after team selection - JWT with selected team
        mock_second_response = Mock()
        mock_second_response.status_code = 200
        mock_second_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.team-beta.jwt",
            "user_id": "test-user-456",
            "team_id": "team-beta",
            "teams": ["team-alpha", "team-beta", "team-gamma"],
        }

        # Simulate user selecting team #2 (team-beta)
        with (
            patch("webbrowser.open") as mock_browser,
            patch(
                "requests.post",
                return_value=_mock_cli_sso_start_response(login_id="cli-session-uuid-456"),
            ),
            patch("requests.get", side_effect=[mock_first_response, mock_second_response]) as mock_get,
            patch("litellm.proxy.client.cli.commands.auth.save_token") as mock_save,
            patch("litellm.proxy.client.cli.interface.show_commands") as mock_show_commands,
            patch("click.prompt", return_value="2"),
        ):  # User selects index 2
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Login successful!" in result.output
            assert "team-beta" in result.output
            # Ensure we surface the human-readable team alias to the user
            assert "Beta Team" in result.output

            # Verify browser was opened
            mock_browser.assert_called_once()
            call_args = mock_browser.call_args[0][0]
            assert "https://test.example.com/sso/key/generate" in call_args

            # Verify two polling requests were made
            assert mock_get.call_count == 2

            # First poll should be without team_id
            first_poll_url = mock_get.call_args_list[0][0][0]
            assert "cli-session-uuid-456" in first_poll_url
            assert "team_id=" not in first_poll_url
            assert mock_get.call_args_list[0].kwargs["headers"] == {"x-litellm-cli-poll-secret": "poll-secret"}

            # Second poll should include team_id=team-beta
            second_poll_url = mock_get.call_args_list[1][0][0]
            assert "team_id=team-beta" in second_poll_url

            # Verify JWT was saved
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert saved_data["key"] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.team-beta.jwt"
            assert saved_data["user_id"] == "test-user-456"

            mock_show_commands.assert_called_once()

    def test_login_without_teams_flow(self):
        """Test complete login flow when user has no teams - JWT generated without team"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}

        # Mock response with no teams
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.no-team.jwt",
            "user_id": "test-user-solo",
            "team_id": None,
            "teams": [],
        }

        with (
            patch("webbrowser.open") as mock_browser,
            patch(
                "requests.post",
                return_value=_mock_cli_sso_start_response(login_id="cli-session-uuid-solo"),
            ),
            patch("requests.get", return_value=mock_response),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as mock_save,
            patch("litellm.proxy.client.cli.interface.show_commands"),
        ):
            result = self.runner.invoke(login, obj=mock_context.obj)

            assert result.exit_code == 0
            assert "Login successful!" in result.output

            # Verify browser was opened
            mock_browser.assert_called_once()
            call_args = mock_browser.call_args[0][0]
            assert "https://test.example.com/sso/key/generate" in call_args
            assert "source=litellm-cli" in call_args
            assert "key=cli-session-uuid-solo" in call_args

            # Verify JWT was saved
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert saved_data["key"] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.no-team.jwt"
            assert saved_data["user_id"] == "test-user-solo"


class TestPrintTokenCommand:
    """Test `lite auth print-token`, used as Claude Code's apiKeyHelper.

    stdout must contain *only* the token -- Claude Code treats stdout
    verbatim as the bearer token, so any diagnostic text on stdout would
    corrupt authentication.

    `lite up` now writes `apiKeyHelper` with an explicit `--base-url` bound
    to whatever proxy it was pointed at (resolve_api_key_helper), so
    print-token enforces that the cached token was actually issued for that
    server -- a token minted for a different, previously-logged-into proxy
    must never be handed to whichever server the helper is invoked for.
    Settings patched by an older `lite up`, or a manually-configured
    apiKeyHelper, can still invoke this bare (no --base-url at all); that
    case falls back to trusting whatever `lite login` stored in token.json,
    since there is no explicit target to check it against. `--base-url`/
    `LITELLM_PROXY_URL` only enforces the match when a caller explicitly
    passes it (tracked via ctx.obj["base_url_explicit"], set by the `cli`
    group from click's ParameterSource); a base_url saved via
    `lite config set` counts as explicit too.
    """

    def setup_method(self):
        self.runner = CliRunner()

    def test_no_stored_token_fails_cleanly(self):
        with patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=None):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code != 0
        assert "Not authenticated" in result.output

    def test_bare_invocation_resolves_server_from_stored_token(self):
        """The legacy/manual invocation shape: no --base-url given at all
        (e.g. settings patched before resolve_api_key_helper started binding
        one). Must use token.json's own base_url, not a hardcoded default."""
        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value={
                    "base_url": "https://litellm-proxy.corp.com",
                    "key": "sk-prod-fresh",
                    "timestamp": time.time(),
                },
            ),
            patch("requests.post") as mock_post,
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code == 0
        assert result.output.strip() == "sk-prod-fresh"
        mock_post.assert_not_called()

    def test_explicit_base_url_mismatch_fails_cleanly(self):
        """When the caller *does* explicitly pass --base-url, a token issued
        for a different server must never be printed. This is the exact
        scenario `lite up`'s own bound --base-url now guards against: a
        token minted for proxy A must not reach a helper invocation aimed
        at proxy B, even though the token itself is otherwise fresh."""
        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value={
                "base_url": "https://other-server.com",
                "key": "sk-should-not-print",
                "timestamp": time.time(),
            },
        ):
            result = self.runner.invoke(
                print_token,
                obj={"base_url": "http://localhost:4000", "base_url_explicit": True},
            )

        assert result.exit_code != 0
        assert "sk-should-not-print" not in result.output

    def test_explicit_base_url_match_prints_token(self):
        """`lite up`'s own bound invocation shape: --base-url matching the token's origin
        must succeed exactly like the bare/legacy invocation does."""
        with patch(
            "litellm.proxy.client.cli.commands.auth.load_token",
            return_value={
                "base_url": "http://localhost:4000",
                "key": "sk-matches",
                "timestamp": time.time(),
            },
        ):
            result = self.runner.invoke(
                print_token,
                obj={"base_url": "http://localhost:4000", "base_url_explicit": True},
            )

        assert result.exit_code == 0
        assert result.output.strip() == "sk-matches"

    def test_fresh_cached_key_printed_without_network_call(self):
        """A recently-issued key should be printed straight from cache -- no
        refresh call on every single invocation (apiKeyHelper gets called
        frequently)."""
        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value={
                    "base_url": "http://localhost:4000",
                    "key": "sk-cached-fresh",
                    "timestamp": time.time(),
                },
            ),
            patch("requests.post") as mock_post,
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code == 0
        assert result.output.strip() == "sk-cached-fresh"
        mock_post.assert_not_called()

    def test_stale_key_fails_fast_without_network_call(self):
        """There is no silent refresh: an expired cached key must fail
        loudly (stderr, nonzero exit) telling the user to `lite login`
        again, rather than making a network call or printing a dead key
        that will just 401 Claude Code."""
        old_timestamp = time.time() - (CLI_JWT_EXPIRATION_HOURS + 1) * 3600

        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value={
                    "base_url": "http://localhost:4000",
                    "key": "sk-stale-key",
                    "timestamp": old_timestamp,
                },
            ),
            patch("requests.post") as mock_post,
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code != 0
        assert "sk-stale-key" not in result.output
        assert "lite login" in result.output
        mock_post.assert_not_called()


def _write_home_json(home: Path, filename: str, payload: dict[str, object]) -> None:
    litellm_dir = home / ".litellm"
    litellm_dir.mkdir(exist_ok=True)
    (litellm_dir / filename).write_text(json.dumps(payload))


class TestPrintTokenWithConfigFile:
    """A config-file base_url is a drop-in replacement for exporting
    LITELLM_PROXY_URL, so print-token must treat it as an explicit server
    choice: a token minted for a different proxy is never handed out."""

    @pytest.fixture
    def isolated_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
        monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
        return tmp_path

    def test_config_base_url_mismatch_fails_closed(self, isolated_home):
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "https://server-a.example.com", "key": "sk-issued-for-a", "timestamp": time.time()},
        )
        _write_home_json(isolated_home, "config.json", {"base_url": "https://server-b.example.com"})

        result = CliRunner().invoke(cli, ["auth", "print-token"])

        assert result.exit_code == 1
        assert "sk-issued-for-a" not in result.output
        assert "Not authenticated for this server" in result.output

    def test_config_base_url_match_prints_token(self, isolated_home):
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "https://server-a.example.com", "key": "sk-issued-for-a", "timestamp": time.time()},
        )
        _write_home_json(isolated_home, "config.json", {"base_url": "https://server-a.example.com"})

        result = CliRunner().invoke(cli, ["auth", "print-token"])

        assert result.exit_code == 0
        assert result.stdout.strip() == "sk-issued-for-a"

    def test_empty_config_base_url_treated_as_unset(self, isolated_home):
        """A hand-edited config.json with base_url "" must behave like no config at all:
        base_url falls back to the default AND explicitness stays False."""
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "https://server-a.example.com", "key": "sk-issued-for-a", "timestamp": time.time()},
        )
        _write_home_json(isolated_home, "config.json", {"base_url": ""})

        result = CliRunner().invoke(cli, ["auth", "print-token"])

        assert result.exit_code == 0
        assert result.stdout.strip() == "sk-issued-for-a"

    def test_bare_invocation_without_config_file_unchanged(self, isolated_home):
        """No config file means base_url_explicit stays False, so the stored
        token's own server is trusted (pre-config behavior must not regress)."""
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "https://server-a.example.com", "key": "sk-issued-for-a", "timestamp": time.time()},
        )

        result = CliRunner().invoke(cli, ["auth", "print-token"])

        assert result.exit_code == 0
        assert result.stdout.strip() == "sk-issued-for-a"


class TestSaveTokenPrivateWrite:
    """token.json holds the real API key: it must never be world-readable at any
    instant, and a failed write must not destroy the previously stored token."""

    @pytest.fixture
    def isolated_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
        monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
        return tmp_path

    def test_save_token_owner_only_permissions_and_no_temp_leftovers(self, isolated_home):
        save_token({"key": "sk-secret", "user_id": "u-1", "timestamp": 1234567890})

        token_file = isolated_home / ".litellm" / "token.json"
        assert json.loads(token_file.read_text()) == {"key": "sk-secret", "user_id": "u-1", "timestamp": 1234567890}
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert list(token_file.parent.glob(".tmp-*")) == []

    def test_save_token_failure_mid_write_preserves_existing_token(self, isolated_home):
        _write_home_json(isolated_home, "token.json", {"key": "sk-original", "timestamp": 1234567890})
        token_file = isolated_home / ".litellm" / "token.json"

        with pytest.raises(TypeError):
            save_token({"key": object()})

        assert json.loads(token_file.read_text()) == {"key": "sk-original", "timestamp": 1234567890}
        assert list(token_file.parent.glob(".tmp-*")) == []


class TestLoginConfigClaude:
    """`lite login --config-claude` wiring into ~/.claude/settings.json"""

    def setup_method(self):
        self.runner = CliRunner()

    def _run_login(self, tmp_path, args, base_url="https://test.example.com"):
        settings_path = tmp_path / "claude" / "settings.json"
        backup_path = tmp_path / "claude_settings_backup.json"
        poll_response = Mock()
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt",
            "user_id": "test-user-123",
            "team_id": "team-1",
            "teams": ["team-1"],
        }
        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", return_value=poll_response),
            patch("litellm.proxy.client.cli.commands.auth.save_token"),
            patch("litellm.proxy.client.cli.interface.show_commands"),
            patch("litellm.proxy.client.cli.commands.auth.CLAUDE_SETTINGS_PATH", settings_path),
            patch(
                "litellm.proxy.client.cli.commands.auth.SETTINGS_FILE_OWNERS",
                (SettingsFileOwner(backup_path, "lite up", "lite down"),),
            ),
            patch(
                "litellm.proxy.client.cli.commands.claude_settings.shutil.which",
                return_value="/usr/local/bin/lite",
            ),
        ):
            result = self.runner.invoke(login, args, obj={"base_url": base_url})
        return result, settings_path, backup_path

    def test_default_login_does_not_touch_claude_settings(self, tmp_path):
        result, settings_path, _backup_path = self._run_login(tmp_path, [])

        assert result.exit_code == 0
        assert "Login successful!" in result.output
        assert not settings_path.exists()
        assert "Configured Claude Code" not in result.output

    def test_flag_writes_the_settings_file_and_reports_success(self, tmp_path):
        result, settings_path, _backup_path = self._run_login(tmp_path, ["--config-claude"])

        assert result.exit_code == 0
        written = json.loads(settings_path.read_text())
        assert written["env"]["ANTHROPIC_BASE_URL"] == "https://test.example.com"
        assert written["apiKeyHelper"] == "/usr/local/bin/lite --base-url https://test.example.com auth print-token"
        assert "Configured Claude Code" in result.output

    def test_flag_preserves_unrelated_settings_on_an_existing_file(self, tmp_path):
        settings_path = tmp_path / "claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"theme": "dark", "env": {"KEEP": "me"}}))

        result, _settings_path, _backup_path = self._run_login(tmp_path, ["--config-claude"])

        assert result.exit_code == 0
        written = json.loads(settings_path.read_text())
        assert written["theme"] == "dark"
        assert written["env"]["KEEP"] == "me"

    def test_settings_failure_is_reported_without_claiming_login_failed(self, tmp_path):
        settings_path = tmp_path / "claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("not json at all {{{")

        result, _settings_path, _backup_path = self._run_login(tmp_path, ["--config-claude"])

        assert result.exit_code != 0
        assert "Login successful!" in result.output
        assert "could not configure Claude Code" in result.output
        assert "invalid JSON" in result.output
        assert "Authentication failed" not in result.output


class _FakeHttpResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for ``requests.Session`` so the CLI's refresh and revoke calls can be observed."""

    instances = []

    def __init__(self):
        self.posts = []
        self.response = _FakeHttpResponse(200, {})
        _FakeSession.instances.append(self)

    def post(self, url, *, data=None, json=None, timeout, allow_redirects):
        self.posts.append((url, data))
        return self.response

    def get(self, url, *, timeout):
        raise AssertionError(f"unexpected GET {url}")


PKCE_BASE_URL = "https://llm.example.com"
PKCE_TOKEN_RESPONSE = {
    "access_token": "sk-cli-rotated",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "llm_srefresh_rotated",
    "user_id": "u1",
    "team_id": "team-b",
}


def _pkce_record(**overrides):
    return {
        "base_url": PKCE_BASE_URL,
        "key": "sk-cli-old",
        "user_id": "u1",
        "user_email": "unknown",
        "user_role": "cli",
        "auth_header_name": "Authorization",
        "jwt_token": "",
        "timestamp": time.time(),
        "expires_at": time.time() + 30,
        "refresh_token": "llm_srefresh_old",
        "client_id": "llm_dcrc_abc",
        "token_endpoint": f"{PKCE_BASE_URL}/token",
        "revocation_endpoint": f"{PKCE_BASE_URL}/revoke",
        "resource": PKCE_BASE_URL,
        "team_id": "team-b",
        **overrides,
    }


def _pkce_credential():
    from litellm.proxy.client.cli.commands.pkce_login import PkceCredential

    return PkceCredential(
        access_token="sk-cli-fresh",
        refresh_token="llm_srefresh_fresh",
        expires_at=time.time() + 3600,
        client_id="llm_dcrc_abc",
        token_endpoint=f"{PKCE_BASE_URL}/token",
        revocation_endpoint=f"{PKCE_BASE_URL}/revoke",
        resource=PKCE_BASE_URL,
        user_id="u1",
        team_id="team-b",
    )


class TestPkceLoginCommand:
    """``lite login --pkce`` swaps the proxy-mediated SSO poll for the browser PKCE flow."""

    @pytest.fixture(autouse=True)
    def isolated_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        return tmp_path

    def setup_method(self):
        self.runner = CliRunner()
        _FakeSession.instances.clear()

    def test_pkce_login_saves_the_new_record_then_revokes_the_refresh_token_it_replaced(self):
        posts_when_saved = []

        with (
            patch("litellm.proxy.client.cli.commands.auth.run_pkce_login", return_value=_pkce_credential()),
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record(team_id="team-a")),
            patch(
                "litellm.proxy.client.cli.commands.auth.save_token",
                side_effect=lambda record: posts_when_saved.append(list(_FakeSession.instances[0].posts)),
            ) as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FakeSession),
            patch("litellm.proxy.client.cli.interface.show_commands"),
        ):
            result = self.runner.invoke(login, ["--pkce"], obj={"base_url": PKCE_BASE_URL})

        assert result.exit_code == 0, result.output
        assert "Login successful!" in result.output
        assert "Could not revoke" not in result.output
        assert save.call_args.args[0]["refresh_token"] == "llm_srefresh_fresh"
        assert save.call_args.args[0]["team_id"] == "team-b"
        assert posts_when_saved == [[]]
        assert _FakeSession.instances[0].posts == [
            (
                f"{PKCE_BASE_URL}/revoke",
                {"token": "llm_srefresh_old", "token_type_hint": "refresh_token", "client_id": "llm_dcrc_abc"},
            )
        ]

    def test_pkce_login_keeps_the_new_record_when_the_old_refresh_token_cannot_be_revoked(self):
        class _FailingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(503, {"error": "temporarily_unavailable"})

        with (
            patch("litellm.proxy.client.cli.commands.auth.run_pkce_login", return_value=_pkce_credential()),
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FailingSession),
            patch("litellm.proxy.client.cli.interface.show_commands"),
        ):
            result = self.runner.invoke(login, ["--pkce"], obj={"base_url": PKCE_BASE_URL})

        assert result.exit_code == 0, result.output
        assert (
            "Could not revoke the previous login's refresh token on the proxy (revocation failed with 503"
            in result.output
        )
        assert "Login successful!" in result.output
        assert save.call_args.args[0]["refresh_token"] == "llm_srefresh_fresh"

    @pytest.mark.parametrize("previous", [None, {"key": "sk-classic", "base_url": PKCE_BASE_URL}])
    def test_pkce_login_without_a_previous_refresh_token_makes_no_revocation_request(self, previous):
        with (
            patch("litellm.proxy.client.cli.commands.auth.run_pkce_login", return_value=_pkce_credential()),
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=previous),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FakeSession),
            patch("litellm.proxy.client.cli.interface.show_commands"),
        ):
            result = self.runner.invoke(login, ["--pkce"], obj={"base_url": PKCE_BASE_URL})

        assert result.exit_code == 0, result.output
        assert "Login successful!" in result.output
        save.assert_called_once()
        assert _FakeSession.instances[0].posts == []

    def test_pkce_login_saves_the_refreshable_record_and_skips_the_sso_poll(self):
        with (
            patch("litellm.proxy.client.cli.commands.auth.run_pkce_login", return_value=_pkce_credential()) as run,
            patch("litellm.proxy.client.cli.commands.auth._start_cli_sso_flow") as sso_start,
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.interface.show_commands"),
        ):
            result = self.runner.invoke(login, ["--pkce"], obj={"base_url": f"{PKCE_BASE_URL}/"})

        assert result.exit_code == 0, result.output
        assert "Login successful!" in result.output
        assert "JWT Token: sk-cli-fresh..." in result.output
        sso_start.assert_not_called()
        assert run.call_args.args[0] == f"{PKCE_BASE_URL}/"
        saved = save.call_args.args[0]
        assert saved["base_url"] == PKCE_BASE_URL
        assert saved["key"] == "sk-cli-fresh"
        assert saved["refresh_token"] == "llm_srefresh_fresh"
        assert saved["client_id"] == "llm_dcrc_abc"
        assert saved["token_endpoint"] == f"{PKCE_BASE_URL}/token"
        assert saved["revocation_endpoint"] == f"{PKCE_BASE_URL}/revoke"
        assert saved["resource"] == PKCE_BASE_URL
        assert saved["user_id"] == "u1"
        assert saved["team_id"] == "team-b"

    def test_pkce_login_failure_is_reported_and_nothing_is_saved(self):
        from litellm.proxy.client.cli.commands.pkce_login import PkceFailure

        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.run_pkce_login",
                return_value=PkceFailure("sign-in was not approved (access_denied): no details"),
            ),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
        ):
            result = self.runner.invoke(login, ["--pkce"], obj={"base_url": PKCE_BASE_URL})

        assert result.exit_code == 0
        assert "Authentication failed: sign-in was not approved (access_denied): no details" in result.output
        save.assert_not_called()

    def test_login_without_the_flag_never_touches_the_pkce_flow(self):
        with (
            patch("litellm.proxy.client.cli.commands.auth.run_pkce_login") as run,
            patch("litellm.proxy.client.cli.commands.auth._start_cli_sso_flow", side_effect=KeyboardInterrupt),
        ):
            result = self.runner.invoke(login, obj={"base_url": PKCE_BASE_URL})

        assert "cancelled" in result.output
        run.assert_not_called()


class TestPkceLogoutCommand:
    def setup_method(self):
        self.runner = CliRunner()
        _FakeSession.instances.clear()

    def test_logout_revokes_the_refresh_token_before_clearing(self):
        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.clear_token") as clear,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FakeSession),
        ):
            result = self.runner.invoke(logout)

        assert result.exit_code == 0
        assert result.output == "Logged out successfully. Authentication token cleared.\n"
        clear.assert_called_once()
        assert _FakeSession.instances[0].posts == [
            (
                f"{PKCE_BASE_URL}/revoke",
                {"token": "llm_srefresh_old", "token_type_hint": "refresh_token", "client_id": "llm_dcrc_abc"},
            )
        ]

    def test_logout_still_clears_when_revocation_fails(self):
        class _FailingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(503, {"error": "temporarily_unavailable"})

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.clear_token") as clear,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FailingSession),
        ):
            result = self.runner.invoke(logout)

        assert result.exit_code == 0
        assert "Could not revoke the refresh token on the proxy (revocation failed with 503" in result.output
        assert "Logged out successfully" in result.output
        clear.assert_called_once()

    def test_logout_of_a_classic_token_makes_no_request(self):
        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value={"key": "sk-classic"}),
            patch("litellm.proxy.client.cli.commands.auth.clear_token") as clear,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FakeSession),
        ):
            result = self.runner.invoke(logout)

        assert result.output == "Logged out successfully. Authentication token cleared.\n"
        clear.assert_called_once()
        assert _FakeSession.instances[0].posts == []


class TestPkcePrintToken:
    """``lite print-token`` is Claude Code's apiKeyHelper, so a near-expiry PKCE key must be
    refreshed silently and stdout must carry nothing but the key."""

    def setup_method(self):
        self.runner = CliRunner()
        _FakeSession.instances.clear()

    def test_print_token_refreshes_a_near_expiry_key_and_saves_the_rotation(self):
        class _RefreshingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(200, PKCE_TOKEN_RESPONSE)

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefreshingSession),
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code == 0, result.output
        assert result.stdout == "sk-cli-rotated\n"
        assert _FakeSession.instances[0].posts[0][0] == f"{PKCE_BASE_URL}/token"
        assert _FakeSession.instances[0].posts[0][1]["refresh_token"] == "llm_srefresh_old"
        saved = save.call_args.args[0]
        assert saved["key"] == "sk-cli-rotated"
        assert saved["refresh_token"] == "llm_srefresh_rotated"

    def test_print_token_prints_a_fresh_pkce_key_without_a_request(self):
        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value=_pkce_record(expires_at=time.time() + 3600),
            ),
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FakeSession),
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.stdout == "sk-cli-old\n"
        assert _FakeSession.instances[0].posts == []

    def test_print_token_fails_when_the_key_expired_and_refresh_is_refused(self):
        class _RefusingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(400, {"error": "invalid_grant"})

        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value=_pkce_record(expires_at=time.time() - 1),
            ),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefusingSession),
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Token expired. Run 'lite login' again." in result.output
        save.assert_not_called()

    def test_print_token_for_an_expired_classic_token_makes_no_request(self):
        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value={"key": "sk-classic", "timestamp": time.time() - (CLI_JWT_EXPIRATION_HOURS + 1) * 3600},
            ),
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _FakeSession),
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code == 1
        assert "Token expired" in result.output
        assert _FakeSession.instances == []


class TestGetStoredApiKeyRefresh:
    def test_get_stored_api_key_refreshes_a_near_expiry_pkce_key(self):
        _FakeSession.instances.clear()

        class _RefreshingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(200, PKCE_TOKEN_RESPONSE)

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefreshingSession),
        ):
            assert get_stored_api_key(PKCE_BASE_URL) == "sk-cli-rotated"
            assert get_stored_api_key("https://other.example.com") is None

        assert save.call_count == 1
        assert len(_FakeSession.instances) == 1
