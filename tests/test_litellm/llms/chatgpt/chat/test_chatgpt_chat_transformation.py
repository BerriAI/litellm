"""
Tests for ChatGPT subscription chat transformation

Source: litellm/llms/chatgpt/chat/transformation.py
"""

import json
import time

import pytest

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.types.router import GenericLiteLLMParams


def _write_auth_record(path, token: str, account_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "account_id": account_id,
                "expires_at": time.time() + 3600,
            }
        )
    )


@pytest.fixture
def auth_files(tmp_path, monkeypatch):
    default_dir = tmp_path / "default"
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(default_dir))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    _write_auth_record(default_dir / "auth.json", "token-default", "acct-default")
    file_a = tmp_path / "account-a" / "auth.json"
    file_b = tmp_path / "account-b" / "auth.json"
    _write_auth_record(file_a, "token-a", "acct-a")
    _write_auth_record(file_b, "token-b", "acct-b")
    return file_a, file_b


class TestChatGPTMultiAccountChat:
    def test_validate_environment_uses_per_model_auth_file(self, auth_files):
        file_a, file_b = auth_files
        config = ChatGPTConfig()

        headers_a = config.validate_environment(
            headers={},
            model="gpt-5.4",
            messages=[],
            optional_params={},
            litellm_params={"chatgpt_auth_file": str(file_a)},
            api_key="token-default",
        )
        headers_b = config.validate_environment(
            headers={},
            model="gpt-5.4",
            messages=[],
            optional_params={},
            litellm_params={"chatgpt_auth_file": str(file_b)},
            api_key="token-default",
        )

        assert headers_a["Authorization"] == "Bearer token-a"
        assert headers_a["ChatGPT-Account-Id"] == "acct-a"
        assert headers_b["Authorization"] == "Bearer token-b"
        assert headers_b["ChatGPT-Account-Id"] == "acct-b"

    def test_validate_environment_defaults_to_global_auth_file(self, auth_files):
        config = ChatGPTConfig()

        headers = config.validate_environment(
            headers={},
            model="gpt-5.4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="token-default",
        )

        assert headers["Authorization"] == "Bearer token-default"
        assert headers["ChatGPT-Account-Id"] == "acct-default"

    def test_provider_info_uses_per_model_auth_file(self, auth_files):
        file_a, _ = auth_files
        config = ChatGPTConfig()

        _, per_model_key, _ = config._get_openai_compatible_provider_info(
            "gpt-5.4", None, None, "chatgpt", {"chatgpt_auth_file": str(file_a)}
        )
        _, default_key, _ = config._get_openai_compatible_provider_info("gpt-5.4", None, None, "chatgpt")

        assert per_model_key == "token-a"
        assert default_key == "token-default"

    def test_get_llm_provider_threads_auth_file(self, auth_files):
        file_a, _ = auth_files

        model, provider, dynamic_api_key, _ = get_llm_provider(
            model="chatgpt/gpt-5.4",
            litellm_params=GenericLiteLLMParams(chatgpt_auth_file=str(file_a)),
        )

        assert model == "gpt-5.4"
        assert provider == "chatgpt"
        assert dynamic_api_key == "token-a"
