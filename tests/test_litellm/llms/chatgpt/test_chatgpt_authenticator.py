import base64
import json
import time
from unittest.mock import mock_open, patch

import pytest

from litellm.llms.chatgpt.authenticator import (
    Authenticator,
    get_cached_authenticator,
    get_chatgpt_auth_file,
)
from litellm.types.router import GenericLiteLLMParams


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


def _write_auth_record(path, token: str, account_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "account_id": account_id,
                "expires_at": time.time() + 3600,
            }
        )
    )


class TestChatGPTMultiAccountAuthenticator:
    def test_explicit_auth_file_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "env-dir"))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "env.json")
        custom_file = tmp_path / "account-a" / "auth.json"

        authenticator = Authenticator(auth_file=str(custom_file))

        assert authenticator.auth_file == str(custom_file)
        assert authenticator.token_dir == str(custom_file.parent)
        assert custom_file.parent.exists()

    def test_default_auth_file_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "env-dir"))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "env.json")

        authenticator = Authenticator()

        assert authenticator.auth_file == str(tmp_path / "env-dir" / "env.json")

    def test_two_auth_files_are_isolated(self, tmp_path):
        file_a = tmp_path / "account-a.json"
        file_b = tmp_path / "account-b.json"
        _write_auth_record(file_a, "token-a", "acct-a")
        _write_auth_record(file_b, "token-b", "acct-b")

        authenticator_a = Authenticator(auth_file=str(file_a))
        authenticator_b = Authenticator(auth_file=str(file_b))

        assert authenticator_a.get_access_token() == "token-a"
        assert authenticator_a.get_account_id() == "acct-a"
        assert authenticator_b.get_access_token() == "token-b"
        assert authenticator_b.get_account_id() == "acct-b"

    def test_get_cached_authenticator_reuses_instance_per_path(self, tmp_path):
        file_a = tmp_path / "account-a.json"
        file_b = tmp_path / "account-b.json"

        authenticator_a = get_cached_authenticator(str(file_a))

        assert get_cached_authenticator(str(file_a)) is authenticator_a
        assert get_cached_authenticator(str(file_b)) is not authenticator_a

    def test_get_chatgpt_auth_file(self):
        assert get_chatgpt_auth_file(None) is None
        assert get_chatgpt_auth_file({}) is None
        assert get_chatgpt_auth_file({"chatgpt_auth_file": ""}) is None
        assert get_chatgpt_auth_file({"chatgpt_auth_file": "/a/auth.json"}) == "/a/auth.json"
        assert get_chatgpt_auth_file(GenericLiteLLMParams()) is None
        assert get_chatgpt_auth_file(GenericLiteLLMParams(chatgpt_auth_file="/b/auth.json")) == "/b/auth.json"
