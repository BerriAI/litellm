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
    def test_get_openai_compatible_provider_info_does_not_eagerly_fetch_api_key(self):
        """
        Symmetric to the ChatGPT fix: the resolution stage must not call
        ``get_api_key`` on either authenticator. That would otherwise hit
        GitHub's ``/copilot_internal/v2/token`` (or trigger a device-code
        login if no access token is stored) at every
        ``add_deployment`` cycle during proxy startup.
        """
        config = GithubCopilotConfig()
        fs_auth = MagicMock(spec=Authenticator)
        fs_auth.get_api_key.side_effect = AssertionError(
            "get_api_key must not be called at resolution time"
        )
        fs_auth.get_api_base.return_value = None
        config.authenticator = fs_auth

        with patch.object(
            DBAuthenticator,
            "get_api_key",
            side_effect=AssertionError(
                "get_api_key must not be called at resolution time"
            ),
        ):
            base, key, _ = config._get_openai_compatible_provider_info(
                model="gpt-5",
                api_base=None,
                api_key=f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds",
                custom_llm_provider="github_copilot",
            )
        # Passthrough — validate_environment resolves at request time.
        assert key == f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds"
        assert base is not None  # Falls back to GITHUB_COPILOT_API_BASE


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
