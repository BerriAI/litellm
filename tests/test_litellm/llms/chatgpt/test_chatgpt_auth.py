import base64
import json
import time
from unittest.mock import patch

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE, GetAccessTokenError
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


def _fake_jwt(payload: dict) -> str:
    encoded_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    encoded_payload = encoded_payload.rstrip("=")
    return f"header.{encoded_payload}.signature"


def test_chatgpt_chat_uses_supplied_api_key_without_device_auth(monkeypatch, tmp_path):
    token_dir = tmp_path / "missing"
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))

    with patch.object(
        Authenticator,
        "get_access_token",
        side_effect=AssertionError("device auth should not be called"),
    ):
        (
            api_base,
            api_key,
            provider,
        ) = ChatGPTConfig()._get_openai_compatible_provider_info(
            model="gpt-5.3-codex",
            api_base=None,
            api_key="client-access-token",
            custom_llm_provider="chatgpt",
        )

    assert api_base == CHATGPT_API_BASE
    assert api_key == "client-access-token"
    assert provider == "chatgpt"
    assert not token_dir.exists()


def test_chatgpt_provider_discovery_does_not_start_device_auth(monkeypatch, tmp_path):
    token_dir = tmp_path / "missing"
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))

    with patch.object(
        Authenticator,
        "_login_device_code",
        side_effect=AssertionError("device auth should not be called"),
    ):
        (
            api_base,
            api_key,
            provider,
        ) = ChatGPTConfig()._get_openai_compatible_provider_info(
            model="gpt-5.3-codex",
            api_base=None,
            api_key=None,
            custom_llm_provider="chatgpt",
        )

    assert api_base == CHATGPT_API_BASE
    assert api_key is None
    assert provider == "chatgpt"
    assert not token_dir.exists()


def test_chatgpt_responses_uses_supplied_api_key_without_device_auth(
    monkeypatch, tmp_path
):
    token_dir = tmp_path / "missing"
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_dir))
    config = ChatGPTResponsesAPIConfig()

    with patch.object(
        config.authenticator,
        "get_access_token",
        side_effect=AssertionError("device auth should not be called"),
    ), patch.object(config.authenticator, "get_account_id", return_value=None):
        headers = config.validate_environment(
            headers={},
            model="gpt-5.3-codex",
            litellm_params=GenericLiteLLMParams(api_key="client-access-token"),
        )

    assert headers["Authorization"] == "Bearer client-access-token"
    assert not token_dir.exists()


def test_chatgpt_authenticator_can_disable_device_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("CHATGPT_DISABLE_DEVICE_AUTH", "true")

    with patch.object(
        Authenticator,
        "_login_device_code",
        side_effect=AssertionError("device auth should not be called"),
    ):
        with pytest.raises(GetAccessTokenError):
            Authenticator().get_access_token()


def test_chatgpt_authenticator_uses_access_token_from_env(monkeypatch, tmp_path):
    access_token = _fake_jwt({"exp": int(time.time()) + 3600})
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("CHATGPT_ACCESS_TOKEN", access_token)

    with patch.object(
        Authenticator,
        "_login_device_code",
        side_effect=AssertionError("device auth should not be called"),
    ):
        assert Authenticator().get_access_token() == access_token
