from unittest.mock import MagicMock

import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.common_utils import GetAccessTokenError

GATEWAY_BASE = "https://chatgpt.example/backend-api/codex"


def _gateway_authenticator() -> MagicMock:
    authenticator = MagicMock()
    authenticator.get_api_base.return_value = GATEWAY_BASE
    authenticator.get_access_token.return_value = "gateway-access-token"
    return authenticator


def _provider_info(
    config: ChatGPTConfig, api_base: str | None, api_key: str | None
) -> tuple[str | None, str | None, str]:
    return config._get_openai_compatible_provider_info(
        model="gpt-5.4", api_base=api_base, api_key=api_key, custom_llm_provider="chatgpt"
    )


class TestChatGPTChatProviderInfo:
    def test_configured_key_uses_configured_base_without_gateway_login(self):
        authenticator = _gateway_authenticator()
        config = ChatGPTConfig(authenticator=authenticator)

        assert _provider_info(config, "http://127.0.0.1:9999", "configured-key") == (
            "http://127.0.0.1:9999",
            "configured-key",
            "chatgpt",
        )
        authenticator.get_access_token.assert_not_called()

    def test_configured_key_without_base_uses_gateway_base(self):
        config = ChatGPTConfig(authenticator=_gateway_authenticator())

        assert _provider_info(config, None, "configured-key") == (GATEWAY_BASE, "configured-key", "chatgpt")

    def test_gateway_login_is_never_sent_to_a_custom_base(self):
        config = ChatGPTConfig(authenticator=_gateway_authenticator())

        assert _provider_info(config, "http://attacker.example", None) == (
            GATEWAY_BASE,
            "gateway-access-token",
            "chatgpt",
        )

    def test_missing_gateway_login_raises_authentication_error(self):
        authenticator = _gateway_authenticator()
        authenticator.get_access_token.side_effect = GetAccessTokenError(status_code=401, message="not logged in")
        config = ChatGPTConfig(authenticator=authenticator)

        with pytest.raises(AuthenticationError, match="not logged in"):
            _provider_info(config, None, None)
