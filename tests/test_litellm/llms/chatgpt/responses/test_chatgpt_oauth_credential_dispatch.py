from unittest.mock import MagicMock, patch

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.db_authenticator import (
    OAUTH_CREDENTIAL_API_KEY_PREFIX,
    DBAuthenticator,
    resolve_authenticator,
)
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


class TestResolveAuthenticator:
    def test_plain_api_key_returns_fallback(self):
        fallback = MagicMock(spec=Authenticator)
        resolved = resolve_authenticator(
            GenericLiteLLMParams(api_key="sk-plain"), fallback
        )
        assert resolved is fallback

    def test_none_litellm_params_returns_fallback(self):
        fallback = MagicMock(spec=Authenticator)
        assert resolve_authenticator(None, fallback) is fallback

    def test_oauth_prefix_returns_db_authenticator(self):
        fallback = MagicMock(spec=Authenticator)
        resolved = resolve_authenticator(
            GenericLiteLLMParams(api_key=f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds"),
            fallback,
        )
        assert isinstance(resolved, DBAuthenticator)
        assert resolved.credential_name == "my-creds"

    def test_oauth_prefix_with_empty_suffix(self):
        fallback = MagicMock(spec=Authenticator)
        resolved = resolve_authenticator(
            GenericLiteLLMParams(api_key=OAUTH_CREDENTIAL_API_KEY_PREFIX), fallback
        )
        assert isinstance(resolved, DBAuthenticator)
        assert resolved.credential_name == ""


class TestValidateEnvironmentRoutesThroughDispatch:
    def test_validate_environment_uses_db_authenticator_when_prefixed(self):
        """
        With ``api_key=oauth:<name>`` the config should ask the DB-backed
        authenticator (not ``self.authenticator``) for tokens.
        """
        config = ChatGPTResponsesAPIConfig()
        fs_auth = MagicMock(spec=Authenticator)
        fs_auth.get_access_token.side_effect = AssertionError(
            "Filesystem authenticator must not be called"
        )
        config.authenticator = fs_auth

        with (
            patch.object(DBAuthenticator, "get_access_token", return_value="db-token"),
            patch.object(DBAuthenticator, "get_account_id", return_value="acct-db"),
        ):
            headers = config.validate_environment(
                headers={},
                model="chatgpt/gpt-5.3-codex",
                litellm_params=GenericLiteLLMParams(
                    api_key=f"{OAUTH_CREDENTIAL_API_KEY_PREFIX}my-creds"
                ),
            )
        assert headers["Authorization"] == "Bearer db-token"
        assert headers["ChatGPT-Account-Id"] == "acct-db"
