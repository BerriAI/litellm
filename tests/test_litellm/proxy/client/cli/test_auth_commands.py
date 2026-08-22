import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import Mock, patch



import pytest
from click.testing import CliRunner

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.litellm_core_utils.cli_keyring import (
    DISABLE_KEYRING_ENV_VAR,
    KeyringDisabled,
    KeyringNotInstalled,
    SecretErased,
    SecretStored,
)
from litellm.litellm_core_utils.cli_token_utils import CliTokenRecord, save_cli_token
from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.auth import (
    get_stored_api_key,
    login,
    logout,
    print_token,
    whoami,
)
from litellm.proxy.client.cli.commands.claude_settings import SettingsFileOwner


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    return tmp_path


def _write_home_json(home: Path, filename: str, payload: dict[str, object]) -> None:
    litellm_dir = home / ".litellm"
    litellm_dir.mkdir(exist_ok=True)
    (litellm_dir / filename).write_text(json.dumps(payload))


def _write_token_file(home: Path, *, key: str | None) -> None:
    """A stored login: `key=None` is the metadata half of a keychain-backed pair, a key is a file-backed one."""
    payload: dict[str, object] = {"base_url": "https://test.example.com", "user_id": "u-1", "timestamp": time.time()}
    _write_home_json(home, "token.json", payload if key is None else {**payload, "key": key})


def _secret_blob(base_url: str, key: str) -> str:
    return json.dumps({"base_url": base_url, "key": key, "jwt_token": ""})


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
            with pytest.raises(ValueError, match='Your litellm CLI is out of date and uses a login flow') as exc_info:
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
            with pytest.raises(ValueError, match='Either --base-url is wrong, or the proxy is older than') as exc_info:
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
            with pytest.raises(ValueError, match='Too many CLI login attempts\\. Try again later\\.') as exc_info:
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
            with pytest.raises(ValueError, match='A proxy, load balancer, or auth gateway in front of') as exc_info:
                _start_cli_sso_flow("https://test.example.com")

        message = str(exc_info.value)
        assert "non-JSON response" in message
        assert "text/html" in message
        assert "Sign in to corporate VPN" in message

    def test_connection_error_points_at_base_url(self):
        import requests

        from litellm.proxy.client.cli.commands.auth import _start_cli_sso_flow

        with patch("requests.post", side_effect=requests.ConnectionError("Connection refused")):
            with pytest.raises(ValueError, match='Connection refused\\. Check that the proxy is running') as exc_info:
                _start_cli_sso_flow("https://unreachable.example.com")

        message = str(exc_info.value)
        assert "Could not reach the proxy" in message
        assert "https://unreachable.example.com/sso/cli/start" in message


class TestStoredApiKeyLookup:
    """`get_stored_api_key` is what every other `lite` subcommand authenticates with, so the
    keychain split and the origin check both have to be invisible to it."""

    def test_returns_the_secret_the_keychain_holds(self, isolated_home, secret_vault_factory):
        _write_home_json(isolated_home, "token.json", {"base_url": "https://real-proxy.com", "user_id": "u-1"})
        vault = secret_vault_factory(blob=_secret_blob("https://real-proxy.com", "sk-from-keychain"))

        assert get_stored_api_key(vault=vault) == "sk-from-keychain"

    def test_returns_a_legacy_plaintext_key(self, isolated_home, secret_vault_factory):
        _write_home_json(isolated_home, "token.json", {"base_url": "https://real-proxy.com", "key": "sk-legacy"})

        assert get_stored_api_key(vault=secret_vault_factory()) == "sk-legacy"

    def test_no_token_at_all_returns_nothing(self, isolated_home, secret_vault_factory):
        assert get_stored_api_key(vault=secret_vault_factory()) is None

    def test_metadata_without_a_secret_returns_nothing(self, isolated_home, secret_vault_factory):
        _write_home_json(isolated_home, "token.json", {"base_url": "https://real-proxy.com", "user_id": "u-1"})

        assert get_stored_api_key(vault=secret_vault_factory()) is None

    def test_matching_base_url_returns_the_key(self, isolated_home, secret_vault_factory):
        _write_home_json(isolated_home, "token.json", {"base_url": "https://real-proxy.com", "key": "sk-prod"})

        assert get_stored_api_key("https://real-proxy.com", vault=secret_vault_factory()) == "sk-prod"

    def test_trailing_slash_on_the_expected_url_is_normalised(self, isolated_home, secret_vault_factory):
        _write_home_json(isolated_home, "token.json", {"base_url": "https://real-proxy.com", "key": "sk-prod"})

        assert get_stored_api_key("https://real-proxy.com/", vault=secret_vault_factory()) == "sk-prod"

    def test_mismatched_base_url_withholds_the_key(self, isolated_home, secret_vault_factory):
        _write_home_json(isolated_home, "token.json", {"base_url": "https://real-proxy.com", "key": "sk-prod"})

        assert get_stored_api_key("https://evil.com", vault=secret_vault_factory()) is None

    def test_old_tokens_without_a_base_url_are_rejected_when_an_origin_is_expected(
        self, isolated_home, secret_vault_factory
    ):
        _write_home_json(isolated_home, "token.json", {"key": "sk-old-token"})

        assert get_stored_api_key("https://real-proxy.com", vault=secret_vault_factory()) is None


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
            patch("litellm.proxy.client.cli.commands.auth.save_token", return_value=SecretStored()) as mock_save,
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
            patch("litellm.proxy.client.cli.commands.auth.save_cli_token") as mock_save,
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
            assert saved_data.key == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt"
            assert saved_data.user_id == "test-user-123"

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

    def test_logout_success(self, isolated_home, secret_vault_factory):
        """Test successful logout"""
        vault = secret_vault_factory(blob=_secret_blob("https://test.example.com", "sk-stored"))
        _write_token_file(isolated_home, key=None)

        result = self.runner.invoke(logout, obj={"secret_vault": vault})

        assert result.exit_code == 0
        assert "Logged out successfully" in result.output
        assert vault.blob is None
        assert not (isolated_home / ".litellm" / "token.json").exists()

    def test_logout_without_the_keyring_package_does_not_claim_the_keychain_is_clear(
        self, isolated_home, secret_vault_factory
    ):
        """Logging out from an install without the cli extra cannot touch an entry a keychain-backed
        login left behind, so it must point at the package rather than report a clean logout."""
        _write_token_file(isolated_home, key=None)

        result = self.runner.invoke(
            logout, obj={"secret_vault": secret_vault_factory(available=False, failure=KeyringNotInstalled())}
        )

        assert result.exit_code == 0
        assert "Logged out successfully" not in result.output
        assert "could not be checked" in result.output
        assert "pip install 'litellm[cli]'" in result.output

    def test_logout_does_not_call_an_unusable_keychain_clean(self, isolated_home, secret_vault_factory):
        """A keychain-backed login, then a login that fell back to the file because the keychain had
        become unusable, leaves the first entry live. The file's own secret says nothing about it,
        so a clean bill of health here is the one answer that cannot be justified."""
        _write_token_file(isolated_home, key="sk-in-file")
        vault = secret_vault_factory(available=False, failure=KeyringDisabled())

        result = self.runner.invoke(logout, obj={"secret_vault": vault})

        assert result.exit_code == 0
        assert "Logged out successfully" not in result.output
        assert "could not be checked" in result.output
        assert DISABLE_KEYRING_ENV_VAR in result.output

    def test_logout_warns_when_the_keychain_refuses_to_release_the_entry(
        self, isolated_home, secret_vault_factory
    ):
        """A locked keychain leaves a live credential behind that the user believes is gone."""
        vault = secret_vault_factory(
            blob=_secret_blob("https://test.example.com", "sk-stored"), erasable=False
        )
        _write_token_file(isolated_home, key=None)

        result = self.runner.invoke(logout, obj={"secret_vault": vault})

        assert result.exit_code == 0
        assert "Logged out successfully" not in result.output
        assert "still in the OS keychain" in result.output
        assert "Unlock your keychain" in result.output

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_logout_reports_a_token_file_it_cannot_clear(self, isolated_home, secret_vault_factory):
        """`lite logout` on a read-only ~/.litellm holding a read-only token file used to end in a
        PermissionError traceback with the credential still sitting in the file. The user has to be
        told what is left and where."""
        _write_token_file(isolated_home, key="sk-in-file")
        config_dir = isolated_home / ".litellm"
        path = config_dir / "token.json"
        path.chmod(0o400)
        config_dir.chmod(0o500)
        try:
            result = self.runner.invoke(logout, obj={"secret_vault": secret_vault_factory()})
        finally:
            config_dir.chmod(0o700)
            path.chmod(0o600)

        assert result.exit_code == 0
        assert "Logged out successfully" not in result.output
        assert "still in" in result.output
        assert str(config_dir / "token.json") in result.output

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_logout_on_a_read_only_directory_still_takes_the_secret_out_of_the_file(
        self, isolated_home, secret_vault_factory
    ):
        """A ~/.litellm that will accept no replacement file and no removal still lets the file it
        has be shortened, so the logout the user asked for happens rather than being handed back to
        them with instructions."""
        _write_token_file(isolated_home, key="sk-in-file")
        config_dir = isolated_home / ".litellm"
        path = config_dir / "token.json"
        config_dir.chmod(0o500)
        try:
            result = self.runner.invoke(logout, obj={"secret_vault": secret_vault_factory()})
        finally:
            config_dir.chmod(0o700)

        assert result.exit_code == 0
        assert "Logged out successfully" in result.output
        assert "sk-in-file" not in path.read_text()

    def test_logout_without_the_keyring_package_still_warns_about_a_file_held_secret(
        self, isolated_home, secret_vault_factory
    ):
        """A file holding its own secret only says the login that wrote it had no keychain to write
        to. An earlier login on this machine may have had one, and no install without the package
        can look, so the honest answer is that the keychain went unchecked."""
        _write_token_file(isolated_home, key="sk-in-file")

        result = self.runner.invoke(
            logout, obj={"secret_vault": secret_vault_factory(available=False, failure=KeyringNotInstalled())}
        )

        assert result.exit_code == 0
        assert "Logged out successfully" not in result.output
        assert "could not be checked" in result.output
        assert "pip install 'litellm[cli]'" in result.output


class TestWhoamiCommand:
    """Test whoami CLI command"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_whoami_authenticated(self):
        """Test whoami when user is authenticated"""
        token_data = CliTokenRecord(
            user_email="test@example.com",
            user_id="test-user-123",
            user_role="admin",
            key="sk-live",
            timestamp=time.time() - 3600,
        )

        with patch("litellm.proxy.client.cli.commands.auth.load_cli_token", return_value=token_data):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Authenticated" in result.output
            assert "test@example.com" in result.output
            assert "test-user-123" in result.output
            assert "admin" in result.output
            assert "Token age: 1.0 hours" in result.output

    def test_whoami_not_authenticated(self):
        """Test whoami when user is not authenticated"""
        with patch("litellm.proxy.client.cli.commands.auth.load_cli_token", return_value=None):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Not authenticated" in result.output
            assert "Run 'lite login'" in result.output

    def test_whoami_old_token(self):
        """Test whoami with old token showing warning"""
        token_data = CliTokenRecord(
            user_email="test@example.com",
            user_id="test-user-123",
            user_role="admin",
            key="sk-live",
            timestamp=time.time() - (25 * 3600),
        )

        with patch("litellm.proxy.client.cli.commands.auth.load_cli_token", return_value=token_data):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Authenticated" in result.output
            assert "Warning: Token is more than 24 hours old" in result.output

    def test_whoami_missing_fields(self):
        """Test whoami with token missing some fields"""
        token_data = CliTokenRecord(key="sk-live", timestamp=time.time() - 3600)

        with patch("litellm.proxy.client.cli.commands.auth.load_cli_token", return_value=token_data):
            result = self.runner.invoke(whoami)

            assert result.exit_code == 0
            assert "Authenticated" in result.output
            assert "Unknown" in result.output  # Should show "Unknown" for missing fields

    def test_whoami_pkce_record_shows_the_team_and_when_the_key_renews(self):
        token_data = {
            "key": "sk-cli",
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
            "key": "sk-cli",
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
            "key": "sk-cli",
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
        token_data = CliTokenRecord(
            user_email="test@example.com",
            user_id="test-user-123",
            user_role="admin",
            key="sk-live",
        )

        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_cli_token",
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
            patch("litellm.proxy.client.cli.commands.auth.save_cli_token") as mock_save,
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
            assert saved_data.key == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.team-beta.jwt"
            assert saved_data.user_id == "test-user-456"

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
            patch("litellm.proxy.client.cli.commands.auth.save_cli_token") as mock_save,
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
            assert saved_data.key == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.no-team.jwt"
            assert saved_data.user_id == "test-user-solo"


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
        with patch("litellm.proxy.client.cli.commands.auth.load_cli_token", return_value=None):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code != 0
        assert "Not authenticated" in result.output

    def test_bare_invocation_resolves_server_from_stored_token(self):
        """The legacy/manual invocation shape: no --base-url given at all
        (e.g. settings patched before resolve_api_key_helper started binding
        one). Must use token.json's own base_url, not a hardcoded default."""
        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_cli_token",
                return_value=CliTokenRecord(
                    base_url="https://litellm-proxy.corp.com",
                    key="sk-prod-fresh",
                    timestamp=time.time(),
                ),
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
            "litellm.proxy.client.cli.commands.auth.load_cli_token",
            return_value=CliTokenRecord(
                base_url="https://other-server.com",
                key="sk-should-not-print",
                timestamp=time.time(),
            ),
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
            "litellm.proxy.client.cli.commands.auth.load_cli_token",
            return_value=CliTokenRecord(
                base_url="http://localhost:4000",
                key="sk-matches",
                timestamp=time.time(),
            ),
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
                "litellm.proxy.client.cli.commands.auth.load_cli_token",
                return_value=CliTokenRecord(
                    base_url="http://localhost:4000",
                    key="sk-cached-fresh",
                    timestamp=time.time(),
                ),
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
                "litellm.proxy.client.cli.commands.auth.load_cli_token",
                return_value=CliTokenRecord(
                    base_url="http://localhost:4000",
                    key="sk-stale-key",
                    timestamp=old_timestamp,
                ),
            ),
            patch("requests.post") as mock_post,
        ):
            result = self.runner.invoke(print_token, obj={})

        assert result.exit_code != 0
        assert "sk-stale-key" not in result.output
        assert "lite login" in result.output
        mock_post.assert_not_called()


class TestPrintTokenWithConfigFile:
    """A config-file base_url is a drop-in replacement for exporting
    LITELLM_PROXY_URL, so print-token must treat it as an explicit server
    choice: a token minted for a different proxy is never handed out."""

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


class TestFileFallbackStorage:
    """On a headless box with no keychain the token file is still the only store, so it has to
    stay owner-only and survive a failed write."""

    def test_owner_only_file_and_directory_with_no_temp_leftovers(self, isolated_home, secret_vault_factory):
        save_cli_token(
            CliTokenRecord(base_url="https://proxy.example.com", key="sk-secret", user_id="u-1", timestamp=1234567890),
            vault=secret_vault_factory(available=False),
        )

        token_file = isolated_home / ".litellm" / "token.json"
        assert json.loads(token_file.read_text())["key"] == "sk-secret"
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(token_file.parent.stat().st_mode) == 0o700
        assert list(token_file.parent.glob(".tmp-*")) == []

    def test_a_failed_write_preserves_the_existing_token(self, isolated_home, secret_vault_factory, monkeypatch):
        _write_home_json(isolated_home, "token.json", {"key": "sk-original", "timestamp": 1234567890})
        token_file = isolated_home / ".litellm" / "token.json"

        def _explode(*args, **kwargs):
            raise TypeError("not serialisable")

        monkeypatch.setattr("litellm.litellm_core_utils.private_json.json.dump", _explode)

        with pytest.raises(TypeError):
            save_cli_token(CliTokenRecord(key="sk-new"), vault=secret_vault_factory(available=False))

        assert json.loads(token_file.read_text()) == {"key": "sk-original", "timestamp": 1234567890}
        assert list(token_file.parent.glob(".tmp-*")) == []


class TestKeychainBackedCommands:
    """End-to-end through the `lite` commands: the secret lives in the keychain, the file keeps
    only metadata, and every command still reads and writes through that split."""

    def setup_method(self):
        self.runner = CliRunner()

    def _login(self, vault, base_url="https://test.example.com"):
        poll_response = Mock()
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "status": "ready",
            "key": "sk-minted",
            "user_id": "test-user-123",
            "team_id": "team-1",
            "teams": ["team-1"],
        }
        with (
            patch("webbrowser.open"),
            patch("requests.post", return_value=_mock_cli_sso_start_response()),
            patch("requests.get", return_value=poll_response),
            patch("litellm.proxy.client.cli.interface.show_commands"),
        ):
            return self.runner.invoke(login, obj={"base_url": base_url, "secret_vault": vault})

    def test_login_puts_the_secret_in_the_keychain_and_not_in_the_file(self, isolated_home, secret_vault_factory):
        vault = secret_vault_factory()

        result = self._login(vault)

        token_file = isolated_home / ".litellm" / "token.json"
        assert result.exit_code == 0
        assert "Credential stored in your OS keychain." in result.output
        assert json.loads(vault.blob)["key"] == "sk-minted"
        assert "sk-minted" not in token_file.read_text()
        assert json.loads(token_file.read_text())["user_id"] == "test-user-123"

    def test_login_without_a_keychain_says_where_the_credential_went(self, isolated_home, secret_vault_factory):
        result = self._login(secret_vault_factory(available=False))

        token_file = isolated_home / ".litellm" / "token.json"
        assert result.exit_code == 0
        assert "No OS keychain available" in result.output
        assert str(token_file) in result.output
        assert json.loads(token_file.read_text())["key"] == "sk-minted"

    def test_login_points_a_user_missing_the_keyring_package_at_the_install(
        self, isolated_home, secret_vault_factory
    ):
        """`lite` ships with every install, the keyring package only with the cli extra. Telling
        that user their machine has no keychain sends them looking for a problem they do not have."""
        result = self._login(secret_vault_factory(available=False, failure=KeyringNotInstalled()))

        token_file = isolated_home / ".litellm" / "token.json"
        assert result.exit_code == 0
        assert "pip install 'litellm[cli]'" in result.output
        assert "No OS keychain available" not in result.output
        assert json.loads(token_file.read_text())["key"] == "sk-minted"

    def test_login_keeps_the_credential_when_the_backend_keeps_nothing(
        self, isolated_home, secret_vault_factory
    ):
        """A backend that accepts writes and stores nothing must not be reported as keychain
        storage, because the file is then told to drop the only remaining copy."""
        result = self._login(secret_vault_factory(discards=True))

        token_file = isolated_home / ".litellm" / "token.json"
        assert result.exit_code == 0
        assert "Credential stored in your OS keychain." not in result.output
        assert "keyring --enable" in result.output
        assert json.loads(token_file.read_text())["key"] == "sk-minted"

    def test_login_names_the_kill_switch_instead_of_blaming_the_machine(
        self, isolated_home, secret_vault_factory
    ):
        result = self._login(secret_vault_factory(available=False, failure=KeyringDisabled()))

        assert result.exit_code == 0
        assert DISABLE_KEYRING_ENV_VAR in result.output
        assert "No OS keychain available" not in result.output
        assert json.loads((isolated_home / ".litellm" / "token.json").read_text())["key"] == "sk-minted"

    def test_whoami_and_print_token_read_through_the_keychain(self, isolated_home, secret_vault_factory):
        vault = secret_vault_factory()
        self._login(vault)
        obj = {"base_url": "https://test.example.com", "secret_vault": vault}

        whoami_result = self.runner.invoke(whoami, obj=obj)
        print_result = self.runner.invoke(print_token, obj=obj)

        assert "Authenticated" in whoami_result.output
        assert "test-user-123" in whoami_result.output
        assert print_result.exit_code == 0
        assert print_result.stdout.strip() == "sk-minted"

    def test_logout_clears_the_keychain_as_well_as_the_file(self, isolated_home, secret_vault_factory):
        vault = secret_vault_factory()
        self._login(vault)

        result = self.runner.invoke(logout, obj={"base_url": "https://test.example.com", "secret_vault": vault})

        assert result.exit_code == 0
        assert "Logged out successfully" in result.output
        assert vault.blob is None
        assert not (isolated_home / ".litellm" / "token.json").exists()

    def test_logout_warns_when_the_keychain_will_not_release_the_secret(self, isolated_home, secret_vault_factory):
        """Silently reporting success would leave a live credential in the keychain."""
        vault = secret_vault_factory(erasable=False)
        self._login(vault)

        result = self.runner.invoke(logout, obj={"base_url": "https://test.example.com", "secret_vault": vault})

        assert result.exit_code == 0
        assert "could not be removed" in result.output
        assert not (isolated_home / ".litellm" / "token.json").exists()

    def test_print_token_explains_a_locked_keychain_instead_of_printing_nothing(
        self, isolated_home, secret_vault_factory
    ):
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "https://test.example.com", "user_id": "u-1", "timestamp": time.time()},
        )
        obj = {"base_url": "https://test.example.com", "secret_vault": secret_vault_factory(available=False)}

        result = self.runner.invoke(print_token, obj=obj)

        assert result.exit_code == 1
        assert "could not be read" in result.output
        assert "lite login" in result.output

    def test_whoami_does_not_call_a_credential_it_cannot_read_authenticated(
        self, isolated_home, secret_vault_factory
    ):
        """A login whose secret is stuck in an unreachable keychain authenticates nothing. Leading
        with "Authenticated" and a token age reads as a working session, and sends the user looking
        for the problem somewhere other than the keychain the notice underneath names."""
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "https://test.example.com", "user_id": "u-1", "timestamp": time.time()},
        )
        obj = {"base_url": "https://test.example.com", "secret_vault": secret_vault_factory(available=False)}

        result = self.runner.invoke(whoami, obj=obj)

        assert "Authenticated" not in result.output
        assert "the credential cannot be read" in result.output
        assert "could not be read" in result.output

    def test_whoami_names_the_kill_switch_rather_than_a_missing_package(
        self, isolated_home, secret_vault_factory
    ):
        """Every unreachable keychain used to be described as a locked one needing the keyring
        package installed. Someone who set the kill switch has the package and an unlocked keychain,
        so that advice sends them to fix two things that were never wrong."""
        _write_token_file(isolated_home, key=None)
        vault = secret_vault_factory(available=False, failure=KeyringDisabled())

        result = self.runner.invoke(whoami, obj={"base_url": "https://test.example.com", "secret_vault": vault})

        assert DISABLE_KEYRING_ENV_VAR in result.output
        assert "pip install" not in result.output

    def test_print_token_points_an_install_without_keyring_at_the_package(
        self, isolated_home, secret_vault_factory
    ):
        _write_token_file(isolated_home, key=None)
        vault = secret_vault_factory(available=False, failure=KeyringNotInstalled())
        obj = {"base_url": "https://test.example.com", "secret_vault": vault}

        result = self.runner.invoke(print_token, obj=obj)

        assert result.exit_code == 1
        assert "pip install 'litellm[cli]'" in result.output
        assert DISABLE_KEYRING_ENV_VAR not in result.output


class TestApiKeyPrecedence:
    """`LITELLM_PROXY_API_KEY` and `--api-key` outrank the stored credential; moving the secret
    into the keychain must not disturb that order."""

    def _resolved_key(self, args, obj=None):
        with patch("litellm.proxy.client.cli.main.print_version") as mock_print_version:
            result = CliRunner().invoke(cli, [*args, "version"], obj=obj)
        assert result.exit_code == 0, result.output
        return mock_print_version.call_args[0][1]

    def test_the_stored_credential_is_the_fallback(self, isolated_home):
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "http://localhost:4000", "key": "sk-stored", "timestamp": time.time()},
        )

        assert self._resolved_key([]) == "sk-stored"

    def test_the_stored_credential_is_read_through_the_injected_keychain(self, isolated_home, secret_vault_factory):
        """The vault handed to the CLI through ctx.obj must be the one the group callback reads,
        so a keychain-held secret resolves without ever touching the host OS keychain."""
        _write_home_json(isolated_home, "token.json", {"base_url": "http://localhost:4000", "timestamp": time.time()})
        vault = secret_vault_factory(_secret_blob("http://localhost:4000", "sk-keychain"))

        assert self._resolved_key([], obj={"secret_vault": vault}) == "sk-keychain"

    def test_env_var_beats_the_stored_credential(self, isolated_home, monkeypatch):
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "http://localhost:4000", "key": "sk-stored", "timestamp": time.time()},
        )
        monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-from-env")

        assert self._resolved_key([]) == "sk-from-env"

    def test_explicit_api_key_beats_both(self, isolated_home, monkeypatch):
        _write_home_json(
            isolated_home,
            "token.json",
            {"base_url": "http://localhost:4000", "key": "sk-stored", "timestamp": time.time()},
        )
        monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-from-env")

        assert self._resolved_key(["--api-key", "sk-explicit"]) == "sk-explicit"


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
            patch("litellm.proxy.client.cli.commands.auth.save_cli_token"),
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

        def record_posts(record, **_):
            posts_when_saved.append(list(_FakeSession.instances[0].posts))
            return SecretStored()

        with (
            patch("litellm.proxy.client.cli.commands.auth.run_pkce_login", return_value=_pkce_credential()),
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record(team_id="team-a")),
            patch("litellm.proxy.client.cli.commands.auth.save_token", side_effect=record_posts) as save,
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
            patch("litellm.proxy.client.cli.commands.auth.save_token", return_value=SecretStored()) as save,
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
            patch("litellm.proxy.client.cli.commands.auth.save_token", return_value=SecretStored()) as save,
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
            patch("litellm.proxy.client.cli.commands.auth.save_token", return_value=SecretStored()) as save,
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
            patch("litellm.proxy.client.cli.commands.auth.save_token", return_value=SecretStored()) as save,
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
            patch("litellm.proxy.client.cli.commands.auth.clear_cli_token", return_value=SecretErased()) as clear,
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

    def test_logout_still_clears_when_the_proxy_refuses_the_revocation(self):
        class _RefusingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(401, {"error": "invalid_client"})

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.clear_cli_token", return_value=SecretErased()) as clear,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefusingSession),
        ):
            result = self.runner.invoke(logout)

        assert result.exit_code == 0
        assert (
            "Could not revoke the refresh token on the proxy (revocation failed with 401: invalid_client); "
            "it expires on its own." in result.output
        )
        assert "Logged out successfully" in result.output
        clear.assert_called_once()

    def test_logout_keeps_the_record_when_the_proxy_cannot_record_the_revocation(self):
        class _UnavailableSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(
                    503, {"error": "temporarily_unavailable", "error_description": "the record is unavailable"}
                )

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.clear_cli_token", return_value=SecretErased()) as clear,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _UnavailableSession),
        ):
            result = self.runner.invoke(logout)

        assert result.exit_code == 1
        assert (
            "Error: The proxy could not record the revocation (revocation failed with 503: the record is unavailable). "
            "Nothing was cleared; run `lite logout` again shortly." in result.output
        )
        assert "Logged out successfully" not in result.output
        clear.assert_not_called()

    def test_logout_of_a_classic_token_makes_no_request(self):
        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value={"key": "sk-classic"}),
            patch("litellm.proxy.client.cli.commands.auth.clear_cli_token", return_value=SecretErased()) as clear,
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
        assert "Could not renew the key: token request failed with 400: invalid_grant" in result.output
        assert "Key expired. Run 'lite login --pkce' again." in result.output
        assert "Run 'lite login' again" not in result.output
        save.assert_not_called()

    def test_print_token_through_the_cli_group_renews_once_and_reports_a_refusal_once(self, monkeypatch):
        monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        class _RefusingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(
                    400, {"error": "invalid_grant", "error_description": "the refresh token was already used"}
                )

        with (
            patch(
                "litellm.proxy.client.cli.commands.auth.load_token",
                return_value=_pkce_record(expires_at=time.time() - 1),
            ),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefusingSession),
        ):
            result = self.runner.invoke(cli, ["--base-url", PKCE_BASE_URL, "auth", "print-token"])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert sum(len(session.posts) for session in _FakeSession.instances) == 1
        assert result.output.count("Could not renew the key") == 1
        assert "Could not renew the key: token request failed with 400: the refresh token was already used" in result.output
        assert "Key expired. Run 'lite login --pkce' again." in result.output
        save.assert_not_called()

    def test_print_token_through_the_cli_group_prints_the_key_the_group_renewed(self, monkeypatch):
        monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        class _RefreshingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(200, PKCE_TOKEN_RESPONSE)

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefreshingSession),
        ):
            result = self.runner.invoke(cli, ["--base-url", PKCE_BASE_URL, "auth", "print-token"])

        assert result.exit_code == 0, result.output
        assert result.stdout == "sk-cli-rotated\n"
        assert sum(len(session.posts) for session in _FakeSession.instances) == 1
        assert save.call_count == 1

    def test_print_token_invoked_bare_for_another_server_renews_once(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        class _RefreshingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(200, PKCE_TOKEN_RESPONSE)

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefreshingSession),
        ):
            result = self.runner.invoke(cli, ["auth", "print-token"])

        assert result.exit_code == 0, result.output
        assert result.stdout == "sk-cli-rotated\n"
        assert sum(len(session.posts) for session in _FakeSession.instances) == 1
        assert save.call_count == 1

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

    def test_get_stored_api_key_reports_a_refused_renewal_on_stderr_and_keeps_the_valid_key(self, capsys):
        _FakeSession.instances.clear()

        class _RefusingSession(_FakeSession):
            def __init__(self):
                super().__init__()
                self.response = _FakeHttpResponse(503, {"error": "temporarily_unavailable"})

        with (
            patch("litellm.proxy.client.cli.commands.auth.load_token", return_value=_pkce_record()),
            patch("litellm.proxy.client.cli.commands.auth.save_token") as save,
            patch("litellm.proxy.client.cli.commands.auth.requests.Session", _RefusingSession),
        ):
            assert get_stored_api_key(PKCE_BASE_URL) == "sk-cli-old"

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "Could not renew the key: token request failed with 503: temporarily_unavailable\n"
        save.assert_not_called()
