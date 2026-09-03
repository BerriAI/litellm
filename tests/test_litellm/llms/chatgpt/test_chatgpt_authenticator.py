import base64
import json
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

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    @pytest.mark.asyncio
    async def test_get_access_token_refuses_device_code_login_in_event_loop(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_login_device_code") as mock_login,
            patch.object(authenticator, "_wait_for_access_token") as mock_wait,
        ):
            with pytest.raises(GetAccessTokenError) as exc:
                authenticator.get_access_token()

        assert exc.value.status_code == 401
        assert "event loop" in str(exc.value)
        mock_login.assert_not_called()
        mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_access_token_refuses_cooldown_wait_in_event_loop(self, authenticator):
        auth_data = json.dumps({"device_code_requested_at": time.time()})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_login_device_code") as mock_login,
            patch.object(authenticator, "_wait_for_access_token") as mock_wait,
        ):
            with pytest.raises(GetAccessTokenError) as exc:
                authenticator.get_access_token()

        assert exc.value.status_code == 401
        assert "event loop" in str(exc.value)
        mock_login.assert_not_called()
        mock_wait.assert_not_called()

    def test_get_access_token_device_code_login_without_event_loop(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_login_device_code", return_value={"access_token": "tok"}),
        ):
            token = authenticator.get_access_token()

        assert token == "tok"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"
