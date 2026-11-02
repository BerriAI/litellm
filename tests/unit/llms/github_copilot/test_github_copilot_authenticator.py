import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock, mock_open, patch

import pytest
import respx

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import (
    APIKeyExpiredError,
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)

API_KEY_URL: Final = "https://copilot-key.test/copilot_internal/v2/token"
REFRESHED_KEY: Final = MappingProxyType({"token": "new-api-key", "expires_at": 4102444800})
TENANT_API: Final = "https://api.tenant.githubcopilot.com"
REPAIRED_API: Final = "https://api.repaired.githubcopilot.com"
TRUNCATED_UTF_16: Final = b"\xff\xfe\x00\xd8"


@pytest.fixture
def logged_in_token_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_COPILOT_API_KEY_URL", API_KEY_URL)
    (tmp_path / "access-token").write_text("gho_logged_in")
    return tmp_path


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

    def test_get_access_token_from_file(self, logged_in_token_dir: Path) -> None:
        """Test retrieving an access token from a file."""
        assert Authenticator().get_access_token() == "gho_logged_in"

    def test_get_access_token_login(self, authenticator):
        """Test logging in to get an access token."""
        mock_token = "mock-access-token"

        with (
            patch.object(authenticator, "_login", return_value=mock_token),
            patch("builtins.open", mock_open()),
            patch("builtins.open", side_effect=IOError) as mock_read,
        ):
            token = authenticator.get_access_token()
            assert token == mock_token
            authenticator._login.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_access_token_refuses_device_code_login_in_event_loop(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_login") as mock_login,
        ):
            with pytest.raises(GetAccessTokenError) as exc:
                authenticator.get_access_token()

        assert exc.value.status_code == 401
        assert "event loop" in str(exc.value)
        assert authenticator.access_token_file not in str(exc.value)
        mock_login.assert_not_called()

    def test_get_access_token_refuses_device_code_login_in_worker_thread(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_login") as mock_login,
        ):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(GetAccessTokenError) as exc:
                    pool.submit(authenticator.get_access_token).result()

        assert exc.value.status_code == 401
        assert "worker thread" in str(exc.value)
        assert authenticator.access_token_file not in str(exc.value)
        mock_login.assert_not_called()

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-8-sig"])
    def test_saved_access_token_in_another_encoding_is_sent_on_refresh(
        self, logged_in_token_dir: Path, encoding: str
    ) -> None:
        """An access-token file a Windows editor saved as UTF-16 or with a BOM still refreshes the key."""
        (logged_in_token_dir / "access-token").write_bytes("gho_logged_in\n".encode(encoding))

        with respx.mock(assert_all_called=True) as router:
            route: Final = router.get(API_KEY_URL).respond(json=dict(REFRESHED_KEY))
            assert Authenticator().get_api_key() == "new-api-key"

        assert route.calls.last.request.headers["authorization"] == "token gho_logged_in"

    def test_undecodable_access_token_is_refused_like_a_missing_one(self, logged_in_token_dir: Path) -> None:
        """A proxy worker gets the login hint for an access-token file it cannot decode, and the file is kept."""
        access_token_file: Final = logged_in_token_dir / "access-token"
        access_token_file.write_bytes(TRUNCATED_UTF_16)

        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(GetAccessTokenError) as exc:
            pool.submit(Authenticator().get_access_token).result()

        assert "worker thread" in str(exc.value)
        assert access_token_file.read_bytes() == TRUNCATED_UTF_16

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
        ).encode()

        with patch("builtins.open", mock_open(read_data=mock_api_key_data)):
            api_key = authenticator.get_api_key()
            assert api_key == "mock-api-key"

    def test_get_api_key_expired(self, authenticator):
        """Test refreshing an expired API key."""
        past_time = (datetime.now() - timedelta(hours=1)).timestamp()
        mock_expired_data = json.dumps(
            {"token": "expired-api-key", "expires_at": past_time}
        ).encode()
        mock_new_data = {
            "token": "new-api-key",
            "expires_at": (datetime.now() + timedelta(hours=1)).timestamp(),
        }

        with (
            patch("builtins.open", mock_open(read_data=mock_expired_data)),
            patch.object(authenticator, "_refresh_api_key", return_value=mock_new_data),
            patch("json.dump") as mock_json_dump,
        ):
            api_key = authenticator.get_api_key()
            assert api_key == "new-api-key"
            authenticator._refresh_api_key.assert_called_once()

    @pytest.mark.parametrize(
        "malformed",
        [
            b"[]",
            json.dumps({"token": "tid=abc", "expires_at": "soon"}).encode(),
            TRUNCATED_UTF_16,
            json.dumps({"token": "tid=abc", "expires_at": 4102444800, "endpoints": []}).encode(),
        ],
        ids=["json-list", "text-expires-at", "truncated-utf-16", "list-endpoints-beside-live-key"],
    )
    def test_get_api_key_malformed_file_refreshes_like_missing(
        self, logged_in_token_dir: Path, malformed: bytes
    ) -> None:
        """A file that is not a Copilot key object is ignored, refreshed over, never answered with a parse error."""
        api_key_file: Final = logged_in_token_dir / "api-key.json"
        api_key_file.write_bytes(malformed)

        with respx.mock(assert_all_called=True) as router:
            route: Final = router.get(API_KEY_URL).respond(json=dict(REFRESHED_KEY))
            assert Authenticator().get_api_key() == "new-api-key"

        assert route.call_count == 1
        assert route.calls.last.request.headers["authorization"] == "token gho_logged_in"
        assert json.loads(api_key_file.read_text()) == REFRESHED_KEY

    @pytest.mark.parametrize(
        "malformed",
        [
            b"[]",
            json.dumps({"token": "tid=abc", "expires_at": "soon"}).encode(),
            TRUNCATED_UTF_16,
            json.dumps({"token": "tid=abc", "expires_at": 4102444800, "endpoints": []}).encode(),
        ],
        ids=["json-list", "text-expires-at", "truncated-utf-16", "list-endpoints-beside-live-key"],
    )
    def test_get_api_key_malformed_file_is_kept_when_refresh_fails(
        self, logged_in_token_dir: Path, malformed: bytes
    ) -> None:
        """Ignoring the file never deletes it, so a failed refresh leaves the operator's file for inspection."""
        api_key_file: Final = logged_in_token_dir / "api-key.json"
        api_key_file.write_bytes(malformed)

        with respx.mock(assert_all_called=True) as router:
            router.get(API_KEY_URL).respond(401)
            with pytest.raises(GetAPIKeyError):
                Authenticator().get_api_key()

        assert api_key_file.read_bytes() == malformed

    def test_refreshed_key_that_cannot_be_saved_is_reused_with_its_endpoint(self, logged_in_token_dir: Path) -> None:
        unwritable_key_file: Final = logged_in_token_dir / "api-key.json"
        unwritable_key_file.mkdir()
        first: Final = Authenticator()
        second: Final = Authenticator()

        with respx.mock(assert_all_called=True) as router:
            route: Final = router.get(API_KEY_URL).respond(json={**REFRESHED_KEY, "endpoints": {"api": TENANT_API}})
            served: Final = (first.get_api_key(), second.get_api_key(), first.get_api_key())

        assert served == ("new-api-key",) * 3
        assert (first.get_api_base(), second.get_api_base()) == (TENANT_API, TENANT_API)
        assert route.call_count == 1
        assert unwritable_key_file.is_dir()

    def test_refreshed_key_that_cannot_be_saved_and_has_expired_is_refreshed_again(
        self, logged_in_token_dir: Path
    ) -> None:
        (logged_in_token_dir / "api-key.json").mkdir()
        auth: Final = Authenticator()

        with respx.mock(assert_all_called=True) as router:
            route: Final = router.get(API_KEY_URL).respond(json={"token": "expired-api-key", "expires_at": 1})
            served: Final = (auth.get_api_key(), auth.get_api_key())

        assert served == ("expired-api-key",) * 2
        assert route.call_count == 2

    def test_kept_key_and_its_endpoint_stay_paired_after_the_key_file_is_repaired(
        self, logged_in_token_dir: Path
    ) -> None:
        key_file: Final = logged_in_token_dir / "api-key.json"
        key_file.mkdir()
        auth: Final = Authenticator()

        with respx.mock(assert_all_called=True) as router:
            router.get(API_KEY_URL).respond(json={**REFRESHED_KEY, "endpoints": {"api": TENANT_API}})
            auth.get_api_key()
        key_file.rmdir()
        key_file.write_text(
            json.dumps({"token": "repaired-api-key", "expires_at": 9999999999, "endpoints": {"api": REPAIRED_API}})
        )

        assert (auth.get_api_key(), auth.get_api_base()) in (
            ("new-api-key", TENANT_API),
            ("repaired-api-key", REPAIRED_API),
        )

    def test_unsaved_key_is_not_reused_after_the_access_token_changes(self, logged_in_token_dir: Path) -> None:
        (logged_in_token_dir / "api-key.json").mkdir()

        with respx.mock(assert_all_called=True) as router:
            route: Final = router.get(API_KEY_URL).respond(json=dict(REFRESHED_KEY))
            Authenticator().get_api_key()
            (logged_in_token_dir / "access-token").write_text("gho_logged_in_again")
            Authenticator().get_api_key()

        assert [call.request.headers["authorization"] for call in route.calls] == [
            "token gho_logged_in",
            "token gho_logged_in_again",
        ]

    @pytest.mark.parametrize(
        "malformed",
        [
            b"[]",
            json.dumps({"endpoints": {"api": 5}}).encode(),
            json.dumps({"token": "t", "expires_at": 1, "endpoints": []}).encode(),
            TRUNCATED_UTF_16,
        ],
        ids=["json-list", "number-api", "list-endpoints", "truncated-utf-16"],
    )
    def test_get_api_base_malformed_file_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: bytes
    ) -> None:
        """get_api_base runs before get_api_key on every request, so it must survive malformed shapes too."""
        monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
        (tmp_path / "api-key.json").write_bytes(malformed)

        assert Authenticator().get_api_base() is None

    def test_get_api_base_keeps_endpoints_when_the_key_is_malformed(self, logged_in_token_dir: Path) -> None:
        """The endpoint is read on its own, so a bad key never sends the refreshed key to the default host."""
        (logged_in_token_dir / "api-key.json").write_text(
            json.dumps({"token": "tid=abc", "expires_at": "soon", "endpoints": {"api": TENANT_API}})
        )
        auth: Final = Authenticator()

        assert auth.get_api_base() == TENANT_API
        with respx.mock(assert_all_called=True) as router:
            router.get(API_KEY_URL).respond(json=dict(REFRESHED_KEY))
            assert auth.get_api_key() == "new-api-key"

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-8-sig"])
    def test_live_key_file_in_another_json_encoding_is_used_without_a_refresh(
        self, logged_in_token_dir: Path, encoding: str
    ) -> None:
        """A key file a Windows editor saved as UTF-16 or with a BOM still holds a live key and tenant."""
        live_key: Final = json.dumps({"token": "tid=abc", "expires_at": 4102444800, "endpoints": {"api": TENANT_API}})
        api_key_file: Final = logged_in_token_dir / "api-key.json"
        api_key_file.write_bytes(live_key.encode(encoding))
        auth: Final = Authenticator()

        with respx.mock:
            assert auth.get_api_base() == TENANT_API
            assert auth.get_api_key() == "tid=abc"

        assert api_key_file.read_bytes() == live_key.encode(encoding)

    def test_ignored_key_file_is_logged_without_its_contents(
        self, logged_in_token_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning names the file but never echoes a token saved in the wrong shape."""
        (logged_in_token_dir / "api-key.json").write_text(
            json.dumps({"token": ["tid=secret-value"], "expires_at": 4102444800})
        )

        with caplog.at_level(logging.WARNING, logger="LiteLLM"), respx.mock(assert_all_called=True) as router:
            router.get(API_KEY_URL).respond(json=dict(REFRESHED_KEY))
            assert Authenticator().get_api_key() == "new-api-key"

        assert "not a Copilot API key file" in caplog.text
        assert "tid=secret-value" not in caplog.text

    def test_refresh_api_key(self, authenticator, mock_http_client):
        """Test refreshing an API key."""
        mock_client, mock_response = mock_http_client
        mock_token = "mock-access-token"
        mock_api_key_data = {"token": "new-api-key", "expires_at": 12345}

        with (
            patch.object(authenticator, "get_access_token", return_value=mock_token),
            patch(
                "litellm.llms.github_copilot.authenticator.get_httpx_client",
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
                "litellm.llms.github_copilot.authenticator.get_httpx_client",
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
                "litellm.llms.github_copilot.authenticator.get_httpx_client",
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
                "litellm.llms.github_copilot.authenticator.get_httpx_client",
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
        ).encode()
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
            patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=mock_client),
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
            patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=mock_client),
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
            patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=mock_client),
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
            patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=mock_client),
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
            patch("litellm.llms.github_copilot.authenticator.get_httpx_client", return_value=mock_client),
            patch.object(authenticator, "get_access_token", return_value="access-tok"),
        ):
            authenticator._refresh_api_key()
            assert mock_client.get.call_args[0][0] == custom_url
