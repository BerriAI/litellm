from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig
from litellm.llms.github_copilot.db_authenticator import (
    OAUTH_CREDENTIAL_API_KEY_PREFIX,
    DBAuthenticator,
)
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture(autouse=True)
def _reset_cache():
    DBAuthenticator._api_key_cache.clear()
    yield
    DBAuthenticator._api_key_cache.clear()


class TestChatTransformationDispatch:
    def test_oauth_prefix_api_key_uses_db_authenticator(self):
        config = GithubCopilotConfig()
        fs_auth = MagicMock(spec=Authenticator)
        fs_auth.get_api_base.side_effect = AssertionError(
            "Filesystem authenticator must not be used for oauth: prefix"
        )
        fs_auth.get_api_key.side_effect = AssertionError(
            "Filesystem authenticator must not be used for oauth: prefix"
        )
        config.authenticator = fs_auth

        with (
            patch.object(
                DBAuthenticator, "get_api_base", return_value="https://x.example"
            ),
            patch.object(DBAuthenticator, "get_api_key", return_value="cop-key"),
        ):
            base, key, _ = config._get_openai_compatible_provider_info(
                model="gpt-5",
                api_base=None,
                api_key=f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds",
                custom_llm_provider="github_copilot",
            )
        assert key == "cop-key"
        assert base == "https://x.example"

    def test_plain_api_key_uses_filesystem_authenticator(self):
        config = GithubCopilotConfig()
        config.authenticator = MagicMock(spec=Authenticator)
        config.authenticator.get_api_base.return_value = None
        config.authenticator.get_api_key.return_value = "fs-key"

        _, key, _ = config._get_openai_compatible_provider_info(
            model="gpt-5",
            api_base=None,
            api_key="sk-plain",
            custom_llm_provider="github_copilot",
        )
        assert key == "fs-key"


class TestResponsesTransformationDispatch:
    def test_oauth_prefix_routes_to_db_authenticator(self):
        config = GithubCopilotResponsesAPIConfig()
        fs_auth = MagicMock(spec=Authenticator)
        fs_auth.get_api_key.side_effect = AssertionError(
            "Filesystem authenticator must not be used"
        )
        config.authenticator = fs_auth

        with patch.object(DBAuthenticator, "get_api_key", return_value="cop-key"):
            headers = config.validate_environment(
                headers={},
                model="gpt-5.1-codex",
                litellm_params=GenericLiteLLMParams(
                    api_key=f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds"
                ),
            )
        # The Copilot responses config sets Authorization with the api_key.
        assert any("cop-key" in v for v in headers.values())
