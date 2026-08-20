import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import (
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    get_copilot_default_headers,
)


class TestGitHubCopilotAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
        ):
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
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
        ):
            auth = Authenticator()
            assert auth.token_dir.endswith("/github_copilot")
            assert auth.access_token_file.endswith("/access-token")
            mock_makedirs.assert_called_once()

    def test_ensure_token_dir(self):
        """Test that the token directory is created if it doesn't exist."""
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
        ):
            auth = Authenticator()
            mock_makedirs.assert_called_once_with(auth.token_dir, exist_ok=True)

    def test_get_api_base_prefers_environment(self, authenticator):
        with patch.dict(
            os.environ,
            {"GITHUB_COPILOT_API_BASE": "https://configured.example.com"},
            clear=True,
        ):
            assert authenticator.get_api_base() == "https://configured.example.com"

    @pytest.mark.parametrize(
        "api_base",
        (
            "http://api.githubcopilot.com",
            "https://user:password@api.githubcopilot.com",
            "https://api.githubcopilot.com?tenant=example",
            "https://api.githubcopilot.com#fragment",
        ),
    )
    def test_get_api_base_rejects_insecure_configuration(self, authenticator, api_base):
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_API_BASE": api_base}, clear=True),
            patch("litellm.llms.github_copilot.authenticator.verbose_logger.warning") as mock_warning,
        ):
            assert authenticator.get_api_base() is None

        mock_warning.assert_called_once_with(
            "Ignoring GITHUB_COPILOT_API_BASE because it must be an HTTPS URL without credentials, query, or fragment"
        )

    def test_get_api_base_uses_default_when_unconfigured(self, authenticator):
        with patch.dict(os.environ, {}, clear=True):
            assert authenticator.get_api_base() is None

    def test_get_api_base_prefers_trusted_deployment_endpoint(self, authenticator):
        with patch.dict(
            os.environ,
            {"GITHUB_COPILOT_API_BASE": "https://configured.example.com"},
            clear=True,
        ):
            assert authenticator.get_api_base("https://deployment.example.com") == "https://deployment.example.com"

    def test_get_api_base_falls_back_from_untrusted_deployment_endpoint(self, authenticator):
        with (
            patch.dict(
                os.environ,
                {"GITHUB_COPILOT_API_BASE": "https://configured.example.com"},
                clear=True,
            ),
            patch("litellm.llms.github_copilot.authenticator.verbose_logger.warning") as mock_warning,
        ):
            assert authenticator.get_api_base("http://attacker.example.com") == "https://configured.example.com"

        mock_warning.assert_called_once_with(
            "Ignoring deployment api_base because it must be an HTTPS URL without credentials, query, or fragment"
        )

    def test_get_github_headers(self, authenticator):
        headers = authenticator._get_github_headers()
        assert headers == {
            "accept": "application/json",
            "content-type": "application/json",
            "copilot-integration-id": "vscode-chat",
            "editor-version": "vscode/1.115.0",
            "editor-plugin-version": "copilot-chat/0.26.7",
            "user-agent": "GitHubCopilotChat/0.26.7",
        }

    def test_auth_requests_support_opencode_identity(self, authenticator, mock_http_client):
        mock_client, mock_response = mock_http_client
        mock_response.json.side_effect = (
            {
                "device_code": "dc",
                "user_code": "UC",
                "verification_uri": "https://github.com/login/device",
            },
            {"access_token": "opencode-oauth-token"},
        )
        environment = {
            "GITHUB_COPILOT_CLIENT_ID": "Ov23li8tweQw6odWQebz",
            "GITHUB_COPILOT_USER_AGENT": "opencode/1.18.7",
            "GITHUB_COPILOT_INTEGRATION_ID": "",
            "GITHUB_COPILOT_EDITOR_VERSION": "",
            "GITHUB_COPILOT_EDITOR_PLUGIN_VERSION": "",
            "GITHUB_COPILOT_API_VERSION": "2026-06-01",
            "GITHUB_COPILOT_OPENAI_INTENT": "conversation-edits",
            "GITHUB_COPILOT_API_BASE": "https://api.githubcopilot.com",
        }

        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
            patch.object(authenticator, "get_access_token", return_value="opencode-oauth-token"),
        ):
            authenticator._get_device_code()
            assert authenticator._poll_for_access_token("dc") == "opencode-oauth-token"
            assert authenticator.get_api_key() == "opencode-oauth-token"
            assert authenticator.get_api_base() == "https://api.githubcopilot.com"
            request_headers = get_copilot_default_headers("opencode-oauth-token")

        expected_auth_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "user-agent": "opencode/1.18.7",
        }
        assert mock_client.post.call_args_list[0].kwargs == {
            "headers": expected_auth_headers,
            "json": {
                "client_id": "Ov23li8tweQw6odWQebz",
                "scope": "read:user",
            },
        }
        assert mock_client.post.call_args_list[1].kwargs == {
            "headers": expected_auth_headers,
            "json": {
                "client_id": "Ov23li8tweQw6odWQebz",
                "device_code": "dc",
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        }
        assert request_headers == {
            **expected_auth_headers,
            "openai-intent": "conversation-edits",
            "x-github-api-version": "2026-06-01",
            "Authorization": "Bearer opencode-oauth-token",
        }
        mock_client.get.assert_not_called()

    def test_get_access_token_from_file(self, authenticator):
        """Test retrieving an access token from a file."""
        mock_token = "mock-access-token"

        with patch("builtins.open", mock_open(read_data=mock_token)):
            token = authenticator.get_access_token()
            assert token == mock_token

    def test_get_access_token_login(self, authenticator):
        mock_token = "mock-access-token"
        write_open = mock_open()

        with (
            patch.object(authenticator, "_login", return_value=mock_token) as mock_login,
            patch("builtins.open", side_effect=(IOError, write_open.return_value)),
        ):
            token = authenticator.get_access_token()

        assert token == mock_token
        mock_login.assert_called_once()
        write_open().write.assert_called_once_with(mock_token)

    def test_get_access_token_survives_persistence_failure(self, authenticator):
        mock_token = "mock-access-token"

        with (
            patch.object(authenticator, "_login", return_value=mock_token) as mock_login,
            patch("builtins.open", side_effect=IOError),
            patch("litellm.llms.github_copilot.authenticator.verbose_logger.error") as mock_error,
        ):
            token = authenticator.get_access_token()

        assert token == mock_token
        mock_login.assert_called_once()
        mock_error.assert_called_once_with("Error saving access token to file")

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

    def test_get_api_key_maps_access_token_failure(self, authenticator):
        with patch.object(
            authenticator,
            "get_access_token",
            side_effect=GetAccessTokenError(message="OAuth failed", status_code=401),
        ):
            with pytest.raises(GetAPIKeyError, match="Failed to get OAuth access token"):
                authenticator.get_api_key()

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
