import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import pytest

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import (
    APIKeyExpiredError,
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)


class TestGitHubCopilotAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=False), patch("os.makedirs") as mock_makedirs:
            auth = Authenticator()
            mock_makedirs.assert_called_once()
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
        with patch("os.path.exists", return_value=False), patch("os.makedirs") as mock_makedirs:
            auth = Authenticator()
            assert auth.token_dir.endswith("/github_copilot")
            assert auth.access_token_file.endswith("/access-token")
            assert auth.api_key_file.endswith("/api-key.json")
            mock_makedirs.assert_called_once()

    def test_ensure_token_dir(self):
        """Test that the token directory is created if it doesn't exist."""
        with patch("os.path.exists", return_value=False), patch("os.makedirs") as mock_makedirs:
            auth = Authenticator()
            mock_makedirs.assert_called_once_with(auth.token_dir, exist_ok=True)

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
        
        with patch.object(authenticator, "_login", return_value=mock_token), \
             patch("builtins.open", mock_open()), \
             patch("builtins.open", side_effect=IOError) as mock_read:
            token = authenticator.get_access_token()
            assert token == mock_token
            authenticator._login.assert_called_once()

    def test_get_access_token_failure(self, authenticator):
        """Test that an exception is raised after multiple login failures."""
        with patch.object(authenticator, "_login", side_effect=GetDeviceCodeError(message="Test error", status_code=400)), \
             patch("builtins.open", side_effect=IOError):
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
        mock_new_data = {"token": "new-api-key", "expires_at": (datetime.now() + timedelta(hours=1)).timestamp()}
        
        with patch("builtins.open", mock_open(read_data=mock_expired_data)), \
             patch.object(authenticator, "_refresh_api_key", return_value=mock_new_data), \
             patch("json.dump") as mock_json_dump:
            api_key = authenticator.get_api_key()
            assert api_key == "new-api-key"
            authenticator._refresh_api_key.assert_called_once()

    def test_refresh_api_key(self, authenticator, mock_http_client):
        """Test refreshing an API key."""
        mock_client, mock_response = mock_http_client
        mock_token = "mock-access-token"
        mock_api_key_data = {"token": "new-api-key", "expires_at": 12345}
        
        with patch.object(authenticator, "get_access_token", return_value=mock_token), \
             patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client), \
             patch.object(mock_response, "json", return_value=mock_api_key_data):
            result = authenticator._refresh_api_key()
            assert result == mock_api_key_data
            mock_client.get.assert_called_once()
            authenticator.get_access_token.assert_called_once()

    def test_refresh_api_key_failure(self, authenticator, mock_http_client):
        """Test failure to refresh an API key."""
        mock_client, mock_response = mock_http_client
        mock_token = "mock-access-token"
        
        with patch.object(authenticator, "get_access_token", return_value=mock_token), \
             patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client), \
             patch.object(mock_response, "json", return_value={}):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()
            assert mock_client.get.call_count == 3

    def test_get_device_code(self, authenticator, mock_http_client):
        """Test getting a device code."""
        mock_client, mock_response = mock_http_client
        mock_device_code_data = {
            "device_code": "mock-device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device"
        }
        
        with patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client), \
             patch.object(mock_response, "json", return_value=mock_device_code_data):
            result = authenticator._get_device_code()
            assert result == mock_device_code_data
            mock_client.post.assert_called_once()

    def test_poll_for_access_token(self, authenticator, mock_http_client):
        """Test polling for an access token."""
        mock_client, mock_response = mock_http_client
        mock_token_data = {"access_token": "mock-access-token"}
        
        with patch("litellm.llms.github_copilot.authenticator._get_httpx_client", return_value=mock_client), \
             patch.object(mock_response, "json", return_value=mock_token_data), \
             patch("time.sleep"):
            result = authenticator._poll_for_access_token("mock-device-code")
            assert result == "mock-access-token"
            mock_client.post.assert_called_once()

    def test_login(self, authenticator):
        """Test the login process."""
        mock_device_code_data = {
            "device_code": "mock-device-code",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device"
        }
        mock_token = "mock-access-token"
        
        with patch.object(authenticator, "_get_device_code", return_value=mock_device_code_data), \
             patch.object(authenticator, "_poll_for_access_token", return_value=mock_token), \
             patch("builtins.print") as mock_print:
            result = authenticator._login()
            assert result == mock_token
            authenticator._get_device_code.assert_called_once()
            authenticator._poll_for_access_token.assert_called_once_with("mock-device-code")
            mock_print.assert_called_once()

    def test_get_api_base_from_file(self, authenticator):
        """Test retrieving the API base endpoint from a file."""
        mock_api_key_data = json.dumps({
            "token": "mock-api-key",
            "expires_at": (datetime.now() + timedelta(hours=1)).timestamp(),
            "endpoints": {"api": "https://api.enterprise.githubcopilot.com"}
        })
        with patch("builtins.open", mock_open(read_data=mock_api_key_data)):
            api_base = authenticator.get_api_base()
            assert api_base == "https://api.enterprise.githubcopilot.com"
