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

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"

    def test_headless_environment_refuses_interactive_device_login(self, authenticator):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(authenticator, "_can_run_interactive_device_login", return_value=False),
            pytest.raises(GetAccessTokenError) as exc_info,
        ):
            authenticator.get_access_token()

        assert exc_info.value.status_code == 401
        assert "cannot run in a non-interactive/headless environment" in str(exc_info.value)

    def test_headless_wait_for_access_token_returns_none_immediately(self, authenticator):
        with patch.object(authenticator, "_can_run_interactive_device_login", return_value=False):
            assert authenticator._wait_for_access_token(timeout_seconds=900) is None

    def test_can_run_interactive_device_login_env_branches(self, authenticator, monkeypatch):
        monkeypatch.setenv("LITELLM_ALLOW_INTERACTIVE_AUTH", "true")
        assert authenticator._can_run_interactive_device_login() is True

        monkeypatch.delenv("LITELLM_ALLOW_INTERACTIVE_AUTH", raising=False)
        monkeypatch.setenv("LITELLM_PROXY", "1")
        assert authenticator._can_run_interactive_device_login() is False

        monkeypatch.delenv("LITELLM_PROXY", raising=False)
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        assert authenticator._can_run_interactive_device_login() is False

        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        monkeypatch.setenv("PORT", "8000")
        assert authenticator._can_run_interactive_device_login() is False

    def test_get_access_token_in_proxy_environment_raises_promptly(self, authenticator, monkeypatch):
        monkeypatch.setenv("LITELLM_PROXY", "1")
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            pytest.raises(GetAccessTokenError) as exc_info,
        ):
            authenticator.get_access_token()

        assert exc_info.value.status_code == 401
        assert "cannot run in a non-interactive/headless environment" in str(exc_info.value)
