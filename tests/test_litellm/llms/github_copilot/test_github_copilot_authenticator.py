import json
import os
import stat
from datetime import datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import pytest

from litellm.llms.github_copilot.authenticator import (
    Authenticator,
    get_authenticator_for_litellm_params,
)
from litellm.llms.github_copilot.common_utils import (
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)


class TestGitHubCopilotAuthenticator:
    @pytest.fixture
    def authenticator(self, tmp_path):
        return Authenticator(token_dir=str(tmp_path))

    @pytest.fixture
    def mock_http_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.return_value = None
        return mock_client, mock_response

    def test_init(self, tmp_path, monkeypatch):
        """Test the initialization of the authenticator."""
        env_token_dir = tmp_path / "env-account"
        explicit_token_dir = tmp_path / "explicit-account"
        monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(env_token_dir))

        with patch("os.makedirs") as mock_makedirs:
            auth = Authenticator(token_dir=str(explicit_token_dir))

        assert auth.token_dir == str(explicit_token_dir)
        assert auth.access_token_file.endswith("/access-token")
        assert auth.api_key_file.endswith("/api-key.json")
        mock_makedirs.assert_not_called()

    def test_ensure_token_dir(self, tmp_path):
        """Test that the token directory is created if it doesn't exist."""
        token_dir = tmp_path / "nested" / "account"
        auth = Authenticator(token_dir=str(token_dir))

        auth._ensure_token_dir()

        assert token_dir.is_dir()
        assert stat.S_IMODE(token_dir.stat().st_mode) == 0o700

    def test_private_write_is_atomic_and_owner_only(self, tmp_path):
        token_dir = tmp_path / "account"
        auth = Authenticator(token_dir=str(token_dir))

        auth._write_private_text(auth.access_token_file, "access-token-value")

        assert (token_dir / "access-token").read_text() == "access-token-value"
        assert stat.S_IMODE((token_dir / "access-token").stat().st_mode) == 0o600
        assert list(token_dir.glob(".litellm-*")) == []

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

    def test_get_api_base_without_global_token_file(self, authenticator):
        """Per-deployment auth does not require a global endpoint file."""
        assert authenticator.get_api_base() is None

    def test_get_api_base_with_invalid_json(self, authenticator):
        authenticator._write_private_text(authenticator.api_key_file, "not-json")

        assert authenticator.get_api_base() is None

    def test_authenticator_selection_reuses_default_without_override(self, authenticator):
        assert get_authenticator_for_litellm_params(authenticator, None) is authenticator

    def test_authenticator_selection_reuses_matching_directory(self, authenticator):
        selected = get_authenticator_for_litellm_params(
            authenticator,
            {"github_copilot_token_dir": authenticator.token_dir},
        )

        assert selected is authenticator

    def test_authenticator_selection_ignores_empty_override(self, authenticator):
        selected = get_authenticator_for_litellm_params(
            authenticator,
            {"github_copilot_token_dir": "  "},
        )

        assert selected is authenticator

    def test_authenticator_selection_uses_distinct_directory(self, authenticator, tmp_path):
        token_dir = tmp_path / "second-account"

        selected = get_authenticator_for_litellm_params(
            authenticator,
            {"github_copilot_token_dir": str(token_dir)},
        )

        assert selected is not authenticator
        assert selected.token_dir == str(token_dir)

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
