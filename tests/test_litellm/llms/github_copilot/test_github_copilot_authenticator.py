import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import pytest

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import (
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)


class TestGitHubCopilotAuthenticator:
    @pytest.fixture
    def authenticator(self):
        auth = Authenticator()
        return auth

    @pytest.fixture
    def mock_http_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.return_value = None
        return mock_client, mock_response

    def test_init(self):
        """Test the initialization of the authenticator."""
        with patch("os.makedirs") as mock_makedirs:
            auth = Authenticator()
            assert os.path.basename(auth.token_dir) == "github_copilot"
            assert os.path.basename(auth.access_token_file) == "access-token"
            assert os.path.basename(auth.api_key_file) == "api-key.json"
            mock_makedirs.assert_not_called()

    def test_ensure_token_dir(self):
        """Test that the token directory is created if it doesn't exist."""
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
        ):
            auth = Authenticator()
            auth._ensure_token_dir()
            mock_makedirs.assert_called_once_with(auth.token_dir, mode=0o700, exist_ok=True)

    def test_ensure_token_dir_permission_error_fallback(self):
        """Test that _ensure_token_dir falls back to temp directory on PermissionError."""
        auth = Authenticator()
        original_dir = auth.token_dir
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs", side_effect=[PermissionError("Permission denied"), None]),
        ):
            auth._ensure_token_dir()
            assert auth.token_dir != original_dir
            assert "litellm" in auth.token_dir

    def test_get_api_key_with_explicit_token_isolation(self, authenticator):
        """Test that explicit tokens use isolated in-memory caching and do not read stale disk files."""
        mock_data = {"token": "token-b-session-key", "expires_at": (datetime.now() + timedelta(hours=1)).timestamp()}
        with (
            patch.object(authenticator, "_refresh_api_key", return_value=mock_data) as mock_refresh,
            patch("builtins.open") as mock_file_open,
        ):
            api_key = authenticator.get_api_key(access_token="user-b-custom-token")
            assert api_key == "token-b-session-key"
            mock_refresh.assert_called_once_with("user-b-custom-token")
            mock_file_open.assert_not_called()

            # Second call with the same token should hit in-memory cache without calling _refresh_api_key again
            cached_key = authenticator.get_api_key(access_token="user-b-custom-token")
            assert cached_key == "token-b-session-key"
            assert mock_refresh.call_count == 1

    def test_get_api_key_with_explicit_token_missing_token_in_response(self, authenticator):
        """Test that get_api_key raises GetAPIKeyError when API response lacks token."""
        with patch.object(authenticator, "_refresh_api_key", return_value={}):
            with pytest.raises(GetAPIKeyError):
                authenticator.get_api_key(access_token="token-without-key")

    def test_get_api_key_with_explicit_token_refresh_error(self, authenticator):
        """Test that get_api_key handles RefreshAPIKeyError when refreshing explicit token."""
        with patch.object(
            authenticator, "_refresh_api_key", side_effect=RefreshAPIKeyError(message="Refresh failed", status_code=401)
        ):
            with pytest.raises(GetAPIKeyError):
                authenticator.get_api_key(access_token="failing-token")

    def test_get_github_headers(self, authenticator):
        """Test that GitHub headers are correctly generated."""
        headers = authenticator._get_github_headers()
        assert "accept" in headers
        assert "editor-version" in headers
        assert "user-agent" in headers
        assert "content-type" in headers

        headers_with_token = authenticator._get_github_headers("test-token")
        assert headers_with_token["authorization"] == "token test-token"

    def test_get_access_token_from_file(self, authenticator):
        """Test retrieving an access token from a file."""
        mock_token = "mock-access-token"

        with patch("builtins.open", mock_open(read_data=mock_token)):
            token = authenticator.get_access_token()
            assert token == mock_token

    def test_get_access_token_login(self, authenticator):
        """Test logging in to get an access token."""
        mock_token = "mock-access-token"

        with (
            patch.object(authenticator, "_login", return_value=mock_token),
            patch("builtins.open", side_effect=IOError),
        ):
            token = authenticator.get_access_token()
            assert token == mock_token
            authenticator._login.assert_called_once()

    def test_get_access_token_failure(self, authenticator):
        """Test that an exception is raised after multiple login failures."""
        with (
            patch.object(
                authenticator,
                "_login",
                side_effect=GetDeviceCodeError(message="Test error", status_code=400),
            ),
            patch("builtins.open", side_effect=IOError),
        ):
            with pytest.raises(GetAccessTokenError):
                authenticator.get_access_token()
            assert authenticator._login.call_count == 3

    def test_get_api_key_from_file(self, authenticator):
        """Test retrieving an API key from a file."""
        future_time = (datetime.now() + timedelta(hours=1)).timestamp()
        mock_api_key_data = json.dumps({"token": "mock-api-key", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=mock_api_key_data)):
            api_key = authenticator.get_api_key()
            assert api_key == "mock-api-key"

    def test_get_api_key_expired(self, authenticator):
        """Test refreshing an expired API key."""
        past_time = (datetime.now() - timedelta(hours=1)).timestamp()
        mock_expired_data = json.dumps({"token": "expired-api-key", "expires_at": past_time})
        mock_new_data = {
            "token": "new-api-key",
            "expires_at": (datetime.now() + timedelta(hours=1)).timestamp(),
        }

        with (
            patch("builtins.open", mock_open(read_data=mock_expired_data)),
            patch.object(authenticator, "_refresh_api_key", return_value=mock_new_data),
            patch("json.dump"),
        ):
            api_key = authenticator.get_api_key()
            assert api_key == "new-api-key"
            authenticator._refresh_api_key.assert_called_once()

    def test_refresh_api_key(self, authenticator, mock_http_client):
        """Test refreshing an API key."""
        mock_client, mock_response = mock_http_client
        mock_token = "mock-access-token"
        mock_api_key_data = {"token": "new-api-key", "expires_at": 12345}

        with (
            patch.object(authenticator, "get_access_token", return_value=mock_token),
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
            patch.object(mock_response, "json", return_value=mock_api_key_data),
        ):
            result = authenticator._refresh_api_key()
            assert result == mock_api_key_data
            mock_client.get.assert_called_once()
            authenticator.get_access_token.assert_called_once()

    def test_refresh_api_key_failure(self, authenticator, mock_http_client):
        """Test failure to refresh an API key."""
        mock_client, mock_response = mock_http_client
        mock_token = "mock-access-token"

        with (
            patch.object(authenticator, "get_access_token", return_value=mock_token),
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
            patch.object(mock_response, "json", return_value={}),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()
            assert mock_client.get.call_count == 3

    def test_get_device_code(self, authenticator, mock_http_client):
        """Test getting a device code."""
        mock_client, mock_response = mock_http_client
        mock_device_code_data = {
            "device_code": "mock-device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
        }

        with (
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
            patch.object(mock_response, "json", return_value=mock_device_code_data),
        ):
            result = authenticator._get_device_code()
            assert result == mock_device_code_data
            mock_client.post.assert_called_once()

    def test_poll_for_access_token(self, authenticator, mock_http_client):
        """Test polling for an access token."""
        mock_client, mock_response = mock_http_client
        mock_token_data = {"access_token": "mock-access-token"}

        with (
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
            patch.object(mock_response, "json", return_value=mock_token_data),
            patch("time.sleep"),
        ):
            result = authenticator._poll_for_access_token("mock-device-code")
            assert result == "mock-access-token"
            mock_client.post.assert_called_once()

    def test_login(self, authenticator):
        """Test the login process."""
        mock_device_code_data = {
            "device_code": "mock-device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
        }
        mock_token = "mock-access-token"

        with (
            patch.object(authenticator, "_get_device_code", return_value=mock_device_code_data),
            patch.object(authenticator, "_poll_for_access_token", return_value=mock_token),
            patch("builtins.print") as mock_print,
        ):
            result = authenticator._login()
            assert result == mock_token
            authenticator._get_device_code.assert_called_once()
            authenticator._poll_for_access_token.assert_called_once_with("mock-device-code")
            mock_print.assert_called_once()

    def test_get_api_base_from_file(self, authenticator):
        """Test retrieving the API base endpoint from a file."""
        mock_api_key_data = json.dumps(
            {
                "token": "mock-api-key",
                "expires_at": (datetime.now() + timedelta(hours=1)).timestamp(),
                "endpoints": {"api": "https://api.enterprise.githubcopilot.com"},
            }
        )
        with patch("builtins.open", mock_open(read_data=mock_api_key_data)):
            api_base = authenticator.get_api_base()
            assert api_base == "https://api.enterprise.githubcopilot.com"

    def test_get_device_code_with_custom_url(self, authenticator, mock_http_client):
        """GITHUB_COPILOT_DEVICE_CODE_URL env var must be used by _get_device_code at call time."""
        mock_client, mock_response = mock_http_client
        custom_url = "https://custom.example.com/device"
        mock_response.json.return_value = {
            "device_code": "dc",
            "user_code": "UC",
            "verification_uri": "https://example.com",
        }
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_DEVICE_CODE_URL": custom_url}),
            patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client),
        ):
            authenticator._get_device_code()
            assert mock_client.post.call_args[0][0] == custom_url

    def test_get_device_code_with_custom_client_id(self, authenticator, mock_http_client):
        """GITHUB_COPILOT_CLIENT_ID env var must appear as client_id in the device-code request body."""
        mock_client, mock_response = mock_http_client
        custom_id = "custom_client_id"
        mock_response.json.return_value = {
            "device_code": "dc",
            "user_code": "UC",
            "verification_uri": "https://example.com",
        }
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_CLIENT_ID": custom_id}),
            patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client),
        ):
            authenticator._get_device_code()
            assert mock_client.post.call_args[1]["json"]["client_id"] == custom_id

    def test_poll_for_access_token_with_custom_url(self, authenticator, mock_http_client):
        """GITHUB_COPILOT_ACCESS_TOKEN_URL env var must be used by _poll_for_access_token at call time."""
        mock_client, mock_response = mock_http_client
        custom_url = "https://custom.example.com/token"
        mock_response.json.return_value = {"access_token": "tok"}
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ACCESS_TOKEN_URL": custom_url}),
            patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client),
            patch("time.sleep"),
        ):
            authenticator._poll_for_access_token("dc")
            assert mock_client.post.call_args[0][0] == custom_url

    def test_poll_for_access_token_with_custom_client_id(self, authenticator, mock_http_client):
        """GITHUB_COPILOT_CLIENT_ID env var must appear as client_id in the polling request body."""
        mock_client, mock_response = mock_http_client
        custom_id = "custom_client_id"
        mock_response.json.return_value = {"access_token": "tok"}
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_CLIENT_ID": custom_id}),
            patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client),
            patch("time.sleep"),
        ):
            authenticator._poll_for_access_token("dc")
            assert mock_client.post.call_args[1]["json"]["client_id"] == custom_id

    def test_refresh_api_key_with_custom_url(self, authenticator, mock_http_client):
        """GITHUB_COPILOT_API_KEY_URL env var must be used by _refresh_api_key at call time."""
        mock_client, mock_response = mock_http_client
        custom_url = "https://custom.example.com/api-key"
        mock_response.json.return_value = {"token": "api-tok", "expires_at": 9999999999}
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_API_KEY_URL": custom_url}),
            patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client),
            patch.object(authenticator, "get_access_token", return_value="access-tok"),
        ):
            authenticator._refresh_api_key()
            assert mock_client.get.call_args[0][0] == custom_url
