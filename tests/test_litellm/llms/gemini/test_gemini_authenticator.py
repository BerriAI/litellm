from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.gemini.authenticator import GeminiAuthenticator


def test_get_token_raises_if_login_response_missing_access_token(tmp_path):
    auth = GeminiAuthenticator()
    auth.oauth_creds_file = str(tmp_path / "oauth_creds.json")

    with (
        patch.object(auth, "_login", return_value={"refresh_token": "abc"}),
        patch.object(auth, "_write_oauth_creds"),
    ):
        with pytest.raises(Exception, match="missing access_token"):
            auth.get_token()


def test_refresh_token_raises_if_response_missing_access_token(tmp_path):
    auth = GeminiAuthenticator()
    auth.oauth_creds_file = str(tmp_path / "oauth_creds.json")

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"expires_in": 3600}

    with (
        patch.object(
            GeminiAuthenticator,
            "_get_oauth_client_credentials",
            return_value=("client-id", "client-secret"),
        ),
        patch("litellm.llms.gemini.authenticator.httpx.post", return_value=mock_resp),
        patch.object(auth, "_write_oauth_creds"),
    ):
        with pytest.raises(Exception, match="missing access_token"):
            auth._refresh_token("refresh-token")


def test_login_times_out_without_oauth_callback(monkeypatch):
    auth = GeminiAuthenticator()
    monkeypatch.setenv("GEMINI_OAUTH_LOOPBACK_TIMEOUT_SECONDS", "1")

    class _NoRequestServer:
        def __init__(self, *args, **kwargs):
            self.server_port = 12345
            self.timeout = None

        def handle_request(self):
            return None

        def server_close(self):
            return None

    with (
        patch.object(
            GeminiAuthenticator,
            "_get_oauth_client_credentials",
            return_value=("client-id", "client-secret"),
        ),
        patch(
            "litellm.llms.gemini.authenticator.http.server.HTTPServer", _NoRequestServer
        ),
        patch("litellm.llms.gemini.authenticator.webbrowser.open", return_value=True),
    ):
        with pytest.raises(Exception, match="OAuth login timed out"):
            auth._login()
