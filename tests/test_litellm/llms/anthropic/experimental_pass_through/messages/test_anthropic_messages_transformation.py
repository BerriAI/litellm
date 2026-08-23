"""OAuth2 client_credentials bearer injection on the native /v1/messages path.

A deployment with oauth_client_credentials in litellm_params fronts the
Anthropic Messages API with an OAuth2 gateway; the minted bearer must replace
x-api-key auth.
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.openai_like.oauth_authenticator import _token_cache

OAUTH_LITELLM_PARAMS = {
    "oauth_client_credentials": True,
    "oauth_token_url": "https://idp.test/token",
    "oauth_client_id": "cid-messages",
    "oauth_client_secret": "secret-messages",
}


@pytest.fixture(autouse=True)
def _flush_token_cache():
    _token_cache.flush_cache()
    yield
    _token_cache.flush_cache()


def _token_client(access_token="tok-messages", expires_in=3600):
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"access_token": access_token, "expires_in": expires_in}
    client.post.return_value = response
    return client


class TestValidateAnthropicMessagesEnvironmentOAuth:
    def test_flag_swaps_x_api_key_for_minted_bearer(self):
        token_client = _token_client(access_token="tok-messages")
        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=token_client,
        ):
            headers, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
                headers={"x-api-key": "sk-config"},
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params=dict(OAUTH_LITELLM_PARAMS),
                api_key="sk-config",
                api_base="https://gateway.test",
            )

        token_client.post.assert_called_once()
        assert "x-api-key" not in headers
        assert headers["authorization"] == "Bearer tok-messages"

    def test_flag_sets_bearer_when_no_prior_auth_header(self):
        token_client = _token_client(access_token="tok-fresh")
        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=token_client,
        ):
            headers, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
                headers={},
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params=dict(OAUTH_LITELLM_PARAMS),
                api_key="sk-config",
                api_base="https://gateway.test",
            )

        assert headers["authorization"] == "Bearer tok-fresh"
        assert "x-api-key" not in headers

    def test_no_flag_keeps_x_api_key_and_fetches_nothing(self):
        token_client = _token_client(access_token="should-not-be-used")
        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=token_client,
        ):
            headers, _ = AnthropicMessagesConfig().validate_anthropic_messages_environment(
                headers={"x-api-key": "sk-config"},
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={"oauth_token_url": "https://idp.test/token"},
                api_key="sk-config",
                api_base="https://gateway.test",
            )

        token_client.post.assert_not_called()
        assert headers["x-api-key"] == "sk-config"
        assert "authorization" not in headers
