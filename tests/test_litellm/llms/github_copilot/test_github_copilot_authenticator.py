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

    def test_get_access_token_no_file_raises(self, authenticator):
        """Test that GetAccessTokenError is raised when no access-token file exists."""
        with patch("builtins.open", side_effect=IOError):
            with pytest.raises(GetAccessTokenError) as exc_info:
                authenticator.get_access_token()
            assert "https://docs.litellm.ai/docs/providers/github_copilot" in str(exc_info.value)

    def test_get_access_token_empty_file_raises(self, authenticator):
        """Test that GetAccessTokenError is raised when the access-token file is empty."""
        with patch("builtins.open", mock_open(read_data="")):
            with pytest.raises(GetAccessTokenError):
                authenticator.get_access_token()

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

    def test_get_github_headers_static(self):
        """Test that get_github_headers works as a static method."""
        headers = Authenticator.get_github_headers()
        assert "accept" in headers
        assert "content-type" in headers
        assert "authorization" not in headers

        headers_with_token = Authenticator.get_github_headers("my-token")
        assert headers_with_token["authorization"] == "token my-token"

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


class TestAuthenticatorCredentialMode:
    """Tests for credential mode (injected access token)."""

    def test_init_credential_mode_no_file_io(self):
        """Credential mode should not create any directories."""
        auth = Authenticator(access_token="test-token")
        assert auth._injected_access_token == "test-token"
        assert not hasattr(auth, "token_dir")

    def test_get_access_token_returns_injected(self):
        """get_access_token returns the injected token directly."""
        auth = Authenticator(access_token="my-github-token")
        assert auth.get_access_token() == "my-github-token"

    def test_get_api_key_credential_mode(self):
        """get_api_key in credential mode calls _refresh_api_key and caches."""
        import litellm.llms.github_copilot.authenticator as auth_module

        access_token = "my-github-token-caching-test"
        # Clear any stale cache entry before the test
        auth_module._credential_api_key_cache.pop(access_token, None)

        auth = Authenticator(access_token=access_token)
        future_time = (datetime.now() + timedelta(hours=1)).timestamp()
        mock_api_key_info = {"token": "copilot-api-key", "expires_at": future_time}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_api_key_info
        mock_client.get.return_value = mock_response

        with patch(
            "litellm.llms.github_copilot.authenticator._get_httpx_client",
            return_value=mock_client,
        ):
            api_key = auth.get_api_key()
            assert api_key == "copilot-api-key"

            # Second call should use cache (no additional HTTP calls)
            api_key2 = auth.get_api_key()
            assert api_key2 == "copilot-api-key"
            assert mock_client.get.call_count == 1  # Only called once

        # Cleanup
        auth_module._credential_api_key_cache.pop(access_token, None)

    def test_get_api_key_credential_mode_expired_cache(self):
        """get_api_key re-fetches when cached token is expired."""
        import litellm.llms.github_copilot.authenticator as auth_module

        past_time = (datetime.now() - timedelta(hours=1)).timestamp()
        future_time = (datetime.now() + timedelta(hours=1)).timestamp()
        access_token = "my-github-token-expired-test"

        auth = Authenticator(access_token=access_token)
        # Pre-populate module-level cache with an expired entry
        auth_module._credential_api_key_cache[access_token] = {
            "token": "old-key",
            "expires_at": past_time,
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"token": "new-key", "expires_at": future_time}
        mock_client.get.return_value = mock_response

        with patch(
            "litellm.llms.github_copilot.authenticator._get_httpx_client",
            return_value=mock_client,
        ):
            api_key = auth.get_api_key()
            assert api_key == "new-key"
            assert mock_client.get.call_count == 1

        # Cleanup
        auth_module._credential_api_key_cache.pop(access_token, None)

    def test_get_api_base_credential_mode_no_cache(self):
        """get_api_base returns None when no cache is available."""
        import litellm.llms.github_copilot.authenticator as auth_module

        access_token = "my-github-token-no-cache-test"
        auth = Authenticator(access_token=access_token)
        # Ensure no stale cache
        auth_module._credential_api_key_cache.pop(access_token, None)
        assert auth.get_api_base() is None

    def test_get_api_base_credential_mode_with_cache(self):
        """get_api_base returns endpoint from module-level cache."""
        import litellm.llms.github_copilot.authenticator as auth_module

        access_token = "my-github-token-cache-test"
        auth = Authenticator(access_token=access_token)
        auth_module._credential_api_key_cache[access_token] = {
            "token": "test",
            "endpoints": {"api": "https://custom.copilot.api"},
        }
        assert auth.get_api_base() == "https://custom.copilot.api"

        # Cleanup
        auth_module._credential_api_key_cache.pop(access_token, None)
