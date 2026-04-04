import base64
import json
import os
import time
from unittest.mock import mock_open, patch

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import GetAccessTokenError


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with patch("builtins.open", mock_open(read_data=auth_data)), patch.object(
            authenticator, "_refresh_tokens", return_value=refreshed
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        auth_data = json.dumps({"id_token": id_token})

        with patch("builtins.open", mock_open(read_data=auth_data)), patch.object(
            authenticator, "_write_auth_file"
        ) as mock_write:
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"

    def test_explicit_auth_file_path_overrides_env(self):
        with patch("os.path.exists", return_value=True):
            authenticator = Authenticator(
                auth_file_path="~/custom-chatgpt/auth.json",
                api_base="https://example.chatgpt.local",
            )

        assert authenticator.auth_file == os.path.expanduser(
            "~/custom-chatgpt/auth.json"
        )
        assert authenticator.token_dir == os.path.expanduser("~/custom-chatgpt")
        assert authenticator.get_api_base() == "https://example.chatgpt.local"

    def test_explicit_auth_file_path_missing_fails_fast(self):
        with patch("os.path.exists", return_value=True):
            authenticator = Authenticator(
                auth_file_path="~/custom-chatgpt/auth.json",
            )

        with patch("builtins.open", side_effect=IOError), patch.object(
            authenticator, "_login_device_code"
        ) as mock_login_device_code, pytest.raises(GetAccessTokenError) as exc_info:
            authenticator.get_access_token()

        mock_login_device_code.assert_not_called()
        assert "Interactive ChatGPT device-code login is disabled" in str(
            exc_info.value
        )

    def test_disable_interactive_login_env_missing_auth_fails_fast(self):
        with patch("os.path.exists", return_value=True):
            authenticator = Authenticator()

        with patch.dict(
            os.environ, {"CHATGPT_DISABLE_INTERACTIVE_LOGIN": "true"}
        ), patch("builtins.open", side_effect=IOError), patch.object(
            authenticator, "_login_device_code"
        ) as mock_login_device_code, pytest.raises(GetAccessTokenError) as exc_info:
            authenticator.get_access_token()

        mock_login_device_code.assert_not_called()
        assert "Interactive ChatGPT device-code login is disabled" in str(
            exc_info.value
        )
