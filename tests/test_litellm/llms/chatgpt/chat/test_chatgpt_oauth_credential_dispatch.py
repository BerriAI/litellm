"""
Regression test: when the ChatGPT *chat* transformation runs with
``api_key="oauth:<name>"``, it must route through ``DBAuthenticator``.

The filesystem ``Authenticator`` would otherwise trigger the 15-minute
device-code flow on the server thread (no tokens on disk) — manifesting
as a hung "Test model" in the UI.
"""

from unittest.mock import MagicMock, patch

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.db_authenticator import (
    OAUTH_CREDENTIAL_API_KEY_PREFIX,
    DBAuthenticator,
)


class TestChatTransformationDispatch:
    def test_get_openai_compatible_provider_info_uses_db_authenticator(self):
        config = ChatGPTConfig()
        fs_auth = MagicMock(spec=Authenticator)
        fs_auth.get_access_token.side_effect = AssertionError(
            "Filesystem authenticator must not be called for oauth: prefix"
        )
        fs_auth.get_api_base.side_effect = AssertionError(
            "Filesystem authenticator must not be called for oauth: prefix"
        )
        config.authenticator = fs_auth

        with (
            patch.object(
                DBAuthenticator,
                "get_api_base",
                return_value="https://chatgpt.com/backend-api/codex",
            ),
            patch.object(
                DBAuthenticator, "get_access_token", return_value="db-access-token"
            ),
        ):
            base, key, _ = config._get_openai_compatible_provider_info(
                model="chatgpt/gpt-5.3-codex",
                api_base=None,
                api_key=f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds",
                custom_llm_provider="chatgpt",
            )
        assert key == "db-access-token"
        assert base == "https://chatgpt.com/backend-api/codex"

    def test_validate_environment_uses_db_authenticator(self):
        config = ChatGPTConfig()
        fs_auth = MagicMock(spec=Authenticator)
        fs_auth.get_account_id.side_effect = AssertionError(
            "Filesystem authenticator must not be called for oauth: prefix"
        )
        config.authenticator = fs_auth

        with (
            patch.object(DBAuthenticator, "get_account_id", return_value="acct-db"),
            patch(
                "litellm.llms.openai.openai.OpenAIConfig.validate_environment",
                return_value={},
            ),
        ):
            headers = config.validate_environment(
                headers={},
                model="chatgpt/gpt-5.3-codex",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
                api_key=f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds",
                api_base=None,
            )
        assert headers.get("ChatGPT-Account-Id") == "acct-db"
