import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, mock_open, patch

import httpx
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

    @staticmethod
    def _make_http_error(status_code: int, message: str = "") -> httpx.HTTPStatusError:
        """Build an ``httpx.HTTPStatusError`` with the given status code."""
        msg = message or f"{status_code} Error"
        request = MagicMock(spec=httpx.Request)
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        return httpx.HTTPStatusError(msg, request=request, response=response)

    @staticmethod
    def _make_failing_response(error: httpx.HTTPStatusError) -> MagicMock:
        """Return a mock response whose ``raise_for_status`` throws *error*."""
        resp = MagicMock()
        resp.raise_for_status.side_effect = error
        return resp

    @staticmethod
    def _make_success_response(payload: dict) -> MagicMock:
        """Return a mock response that succeeds with *payload*."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = payload
        return resp

    def test_init(self):
        """Test the initialization of the authenticator."""
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
        ):
            auth = Authenticator()
            assert auth.token_dir.endswith("/github_copilot")
            assert auth.access_token_file.endswith("/access-token")
            assert auth.api_key_file.endswith("/api-key.json")
            mock_makedirs.assert_called_once()

    def test_ensure_token_dir(self):
        """Test that the token directory is created if it doesn't exist."""
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs") as mock_makedirs,
        ):
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

        with (
            patch.object(authenticator, "_login", return_value=mock_token),
            patch("builtins.open", mock_open()),
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
        mock_api_key_data = json.dumps(
            {"token": "mock-api-key", "expires_at": future_time}
        )

        with patch("builtins.open", mock_open(read_data=mock_api_key_data)):
            api_key = authenticator.get_api_key()
            assert api_key == "mock-api-key"

    def test_get_api_key_expired(self, authenticator):
        """Test refreshing an expired API key."""
        past_time = (datetime.now() - timedelta(hours=1)).timestamp()
        mock_expired_data = json.dumps(
            {"token": "expired-api-key", "expires_at": past_time}
        )
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
            patch.object(
                authenticator, "_get_device_code", return_value=mock_device_code_data
            ),
            patch.object(
                authenticator, "_poll_for_access_token", return_value=mock_token
            ),
            patch("builtins.print") as mock_print,
        ):
            result = authenticator._login()
            assert result == mock_token
            authenticator._get_device_code.assert_called_once()
            authenticator._poll_for_access_token.assert_called_once_with(
                "mock-device-code"
            )
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
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            authenticator._get_device_code()
            assert mock_client.post.call_args[0][0] == custom_url

    def test_get_device_code_with_custom_client_id(
        self, authenticator, mock_http_client
    ):
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
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            authenticator._get_device_code()
            assert mock_client.post.call_args[1]["json"]["client_id"] == custom_id

    def test_poll_for_access_token_with_custom_url(
        self, authenticator, mock_http_client
    ):
        """GITHUB_COPILOT_ACCESS_TOKEN_URL env var must be used by _poll_for_access_token at call time."""
        mock_client, mock_response = mock_http_client
        custom_url = "https://custom.example.com/token"
        mock_response.json.return_value = {"access_token": "tok"}
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ACCESS_TOKEN_URL": custom_url}),
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
            patch("time.sleep"),
        ):
            authenticator._poll_for_access_token("dc")
            assert mock_client.post.call_args[0][0] == custom_url

    def test_poll_for_access_token_with_custom_client_id(
        self, authenticator, mock_http_client
    ):
        """GITHUB_COPILOT_CLIENT_ID env var must appear as client_id in the polling request body."""
        mock_client, mock_response = mock_http_client
        custom_id = "custom_client_id"
        mock_response.json.return_value = {"access_token": "tok"}
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_CLIENT_ID": custom_id}),
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
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
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
            patch.object(authenticator, "get_access_token", return_value="access-tok"),
        ):
            authenticator._refresh_api_key()
            assert mock_client.get.call_args[0][0] == custom_url

    def test_refresh_api_key_401_invalidates_and_retries(self, authenticator):
        """A 401 deletes the cached token, re-acquires, and retries with the
        fresh token in the Authorization header."""
        http_401 = self._make_http_error(401, "401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            self._make_failing_response(http_401),
            self._make_success_response({"token": "new-api-key", "expires_at": 99999}),
        ]

        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ENABLE_AUTH_RECOVERY": "true"}),
            patch.object(
                authenticator,
                "get_access_token",
                side_effect=["stale-token", "fresh-token"],
            ),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            result = authenticator._refresh_api_key()

            assert result == {"token": "new-api-key", "expires_at": 99999}
            mock_remove.assert_called_once_with(authenticator.access_token_file)

            # Verify stale token on first call, fresh on second
            first_headers = mock_client.get.call_args_list[0][1]["headers"]
            assert first_headers["authorization"] == "token stale-token"
            second_headers = mock_client.get.call_args_list[1][1]["headers"]
            assert second_headers["authorization"] == "token fresh-token"

    def test_refresh_api_key_401_invalidates_only_once(self, authenticator):
        """Persistent 401s delete the token exactly once; remaining retries
        use the fresh token until exhaustion."""
        http_401 = self._make_http_error(401, "401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_failing_response(http_401)

        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ENABLE_AUTH_RECOVERY": "true"}),
            patch.object(
                authenticator,
                "get_access_token",
                side_effect=["stale-token", "fresh-token"],
            ),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            mock_remove.assert_called_once_with(authenticator.access_token_file)
            assert mock_client.get.call_count == 3

            # Attempt 1 used stale token, attempts 2+3 used fresh token
            assert (
                mock_client.get.call_args_list[0][1]["headers"]["authorization"]
                == "token stale-token"
            )
            assert (
                mock_client.get.call_args_list[1][1]["headers"]["authorization"]
                == "token fresh-token"
            )
            assert (
                mock_client.get.call_args_list[2][1]["headers"]["authorization"]
                == "token fresh-token"
            )

    def test_refresh_api_key_401_on_second_attempt(self, authenticator):
        """A 401 on attempt 2 (after a 500 on attempt 1) still deletes the
        token and retries with a fresh one."""
        http_500 = self._make_http_error(500, "500 Internal Server Error")
        http_401 = self._make_http_error(401, "401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            self._make_failing_response(http_500),
            self._make_failing_response(http_401),
            self._make_success_response({"token": "recovered", "expires_at": 99999}),
        ]

        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ENABLE_AUTH_RECOVERY": "true"}),
            patch.object(
                authenticator,
                "get_access_token",
                side_effect=["stale-token", "fresh-token"],
            ),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            result = authenticator._refresh_api_key()

            assert result == {"token": "recovered", "expires_at": 99999}
            mock_remove.assert_called_once()

            # Attempts 1 (500) and 2 (401) used stale token
            assert (
                mock_client.get.call_args_list[0][1]["headers"]["authorization"]
                == "token stale-token"
            )
            assert (
                mock_client.get.call_args_list[1][1]["headers"]["authorization"]
                == "token stale-token"
            )
            # Attempt 3 used fresh token after invalidation
            assert (
                mock_client.get.call_args_list[2][1]["headers"]["authorization"]
                == "token fresh-token"
            )

    def test_refresh_api_key_non_401_does_not_invalidate(self, authenticator):
        """Non-401 HTTP errors do NOT trigger token invalidation."""
        http_500 = self._make_http_error(500, "500 Internal Server Error")
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_failing_response(http_500)

        with (
            patch.object(authenticator, "get_access_token", return_value="mock-token"),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            mock_remove.assert_not_called()
            assert mock_client.get.call_count == 3

    def test_refresh_api_key_401_reacquire_fails_raises_cleanly(self, authenticator):
        """When device flow fails after token deletion, the file is already
        gone so token_invalidated is set True — no further invalidation
        attempts. The loop exhausts and raises RefreshAPIKeyError."""
        http_401 = self._make_http_error(401, "401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_failing_response(http_401)

        reacquire_err = GetAccessTokenError(
            message="device flow failed", status_code=401
        )
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ENABLE_AUTH_RECOVERY": "true"}),
            patch.object(
                authenticator,
                "get_access_token",
                side_effect=["stale-token", reacquire_err],
            ),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            assert mock_client.get.call_count == 3
            # File deleted once on first 401; no re-attempts (file already gone)
            mock_remove.assert_called_once()
            # All 3 attempts used the stale token (re-acquire never succeeded)
            for call in mock_client.get.call_args_list:
                assert call[1]["headers"]["authorization"] == "token stale-token"

    def test_refresh_api_key_401_reacquire_device_code_error(self, authenticator):
        """When get_access_token raises GetDeviceCodeError after token
        deletion, the file is already gone so token_invalidated is set True
        — no further invalidation attempts."""
        http_401 = self._make_http_error(401, "401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_failing_response(http_401)

        reacquire_err = GetDeviceCodeError(message="no device code", status_code=400)
        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ENABLE_AUTH_RECOVERY": "true"}),
            patch.object(
                authenticator,
                "get_access_token",
                side_effect=["stale-token", reacquire_err],
            ),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            mock_remove.assert_called_once()
            assert mock_client.get.call_count == 3
            for call in mock_client.get.call_args_list:
                assert call[1]["headers"]["authorization"] == "token stale-token"

    def test_refresh_api_key_401_os_remove_fails(self, authenticator):
        """When os.remove raises OSError (e.g. permission denied), the code
        sets token_invalidated=True, skips re-acquire (file still has stale
        token), and continues the retry loop without re-entering the 401
        block on subsequent attempts."""
        http_401 = self._make_http_error(401, "401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_failing_response(http_401)

        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ENABLE_AUTH_RECOVERY": "true"}),
            patch.object(
                authenticator, "get_access_token", return_value="stale-token"
            ) as mock_get_token,
            patch(
                "os.remove", side_effect=PermissionError("Permission denied")
            ) as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            assert mock_client.get.call_count == 3
            # os.remove attempted only once — token_invalidated set on failure
            mock_remove.assert_called_once()
            # get_access_token called only once (initial), never for re-acquire
            mock_get_token.assert_called_once()
            # All attempts used stale token since file was never deleted
            for call in mock_client.get.call_args_list:
                assert call[1]["headers"]["authorization"] == "token stale-token"

    def test_refresh_api_key_403_does_not_invalidate(self, authenticator):
        """A 403 (Forbidden) does NOT trigger token invalidation — only 401."""
        http_403 = self._make_http_error(403, "403 Forbidden")
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_failing_response(http_403)

        with (
            patch.object(authenticator, "get_access_token", return_value="mock-token"),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            mock_remove.assert_not_called()
            assert mock_client.get.call_count == 3

    def test_refresh_api_key_401_then_500_uses_fresh_token(self, authenticator):
        """After a 401 triggers invalidation and re-acquire succeeds, a
        subsequent 500 does NOT re-trigger invalidation and retries with
        the fresh token."""
        http_401 = self._make_http_error(401, "401 Unauthorized")
        http_500 = self._make_http_error(500, "500 Internal Server Error")
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            self._make_failing_response(http_401),
            self._make_failing_response(http_500),
            self._make_success_response({"token": "recovered", "expires_at": 99999}),
        ]

        with (
            patch.dict(os.environ, {"GITHUB_COPILOT_ENABLE_AUTH_RECOVERY": "true"}),
            patch.object(
                authenticator,
                "get_access_token",
                side_effect=["stale-token", "fresh-token"],
            ),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            result = authenticator._refresh_api_key()

            assert result == {"token": "recovered", "expires_at": 99999}
            mock_remove.assert_called_once_with(authenticator.access_token_file)
            # Attempt 1 used stale, attempts 2+3 used fresh (even though attempt 2 was a 500)
            assert (
                mock_client.get.call_args_list[0][1]["headers"]["authorization"]
                == "token stale-token"
            )
            assert (
                mock_client.get.call_args_list[1][1]["headers"]["authorization"]
                == "token fresh-token"
            )
            assert (
                mock_client.get.call_args_list[2][1]["headers"]["authorization"]
                == "token fresh-token"
            )

    def test_refresh_api_key_connection_error_does_not_invalidate(self, authenticator):
        """A non-HTTP exception (e.g. ConnectionError) is caught by the
        broad except Exception branch and does not trigger invalidation."""
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("Connection refused")

        with (
            patch.object(authenticator, "get_access_token", return_value="mock-token"),
            patch("os.remove") as mock_remove,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            mock_remove.assert_not_called()
            assert mock_client.get.call_count == 3

    def test_get_api_key_catches_access_token_error(self, authenticator):
        """get_api_key catches GetAccessTokenError from _refresh_api_key
        and wraps it in GetAPIKeyError."""
        with (
            patch("builtins.open", side_effect=IOError),
            patch.object(
                authenticator,
                "_refresh_api_key",
                side_effect=GetAccessTokenError(
                    message="device flow timeout", status_code=401
                ),
            ),
        ):
            with pytest.raises(GetAPIKeyError, match="refresh API key"):
                authenticator.get_api_key()

    def test_refresh_api_key_401_default_no_auth_recovery(self, authenticator):
        """By default, a 401 does NOT delete the
        cached token or run device flow. The retry loop drains with the
        stale token and raises RefreshAPIKeyError, matching pre-fix behavior.
        Recovery is opt-in via GITHUB_COPILOT_ENABLE_AUTH_RECOVERY=true."""
        http_401 = self._make_http_error(401, "401 Unauthorized")
        mock_client = MagicMock()
        mock_client.get.return_value = self._make_failing_response(http_401)

        env_without_var = {
            k: v
            for k, v in os.environ.items()
            if k != "GITHUB_COPILOT_ENABLE_AUTH_RECOVERY"
        }
        with (
            patch.dict(os.environ, env_without_var, clear=True),
            patch.object(authenticator, "get_access_token", return_value="stale-token"),
            patch("os.remove") as mock_remove,
            patch.object(authenticator, "_get_device_code") as mock_device_code,
            patch.object(authenticator, "_poll_for_access_token") as mock_poll,
            patch(
                "litellm.llms.github_copilot.authenticator._get_httpx_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(RefreshAPIKeyError):
                authenticator._refresh_api_key()

            mock_remove.assert_not_called()
            mock_device_code.assert_not_called()
            mock_poll.assert_not_called()
            assert mock_client.get.call_count == 3
