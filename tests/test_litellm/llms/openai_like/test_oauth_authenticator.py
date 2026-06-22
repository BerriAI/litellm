"""
Tests for the custom_oauth provider: OAuth2 client_credentials token fetch +
caching, and bearer injection through the dynamic JSON-provider config.

Regression coverage for https://github.com/BerriAI/litellm/issues/12367
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.openai_like.oauth_authenticator import (
    OAuthClientCredentialsError,
    _token_cache,
    get_client_credentials_token,
)


@pytest.fixture(autouse=True)
def _flush_token_cache():
    _token_cache.flush_cache()
    yield
    _token_cache.flush_cache()


def _token_client(access_token="tok-abc", expires_in=3600):
    """A fake HTTPHandler whose .post returns a token response."""
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "access_token": access_token,
        "expires_in": expires_in,
    }
    client.post.return_value = response
    return client


class TestClientCredentialsTokenFetch:
    def test_posts_client_credentials_grant(self):
        client = _token_client(access_token="tok-1")
        token = get_client_credentials_token(
            token_url="https://idp.test/oauth/token",
            client_id="cid-1",
            client_secret="secret-1",
            scope="scope-1",
            http_client=client,
        )

        assert token == "tok-1"
        client.post.assert_called_once()
        args, kwargs = client.post.call_args
        assert args[0] == "https://idp.test/oauth/token"
        data = kwargs["data"]
        assert data["grant_type"] == "client_credentials"
        assert data["client_id"] == "cid-1"
        assert data["client_secret"] == "secret-1"
        assert data["scope"] == "scope-1"

    def test_scope_omitted_when_not_provided(self):
        client = _token_client()
        get_client_credentials_token(
            token_url="https://idp.test/oauth/token",
            client_id="cid-2",
            client_secret="secret-2",
            http_client=client,
        )
        assert "scope" not in client.post.call_args.kwargs["data"]

    def test_token_is_cached_within_ttl(self):
        client = _token_client(access_token="tok-cached", expires_in=3600)
        kwargs = dict(
            token_url="https://idp.test/cache",
            client_id="cid-cache",
            client_secret="s",
            http_client=client,
        )
        first = get_client_credentials_token(**kwargs)
        second = get_client_credentials_token(**kwargs)

        assert first == second == "tok-cached"
        client.post.assert_called_once()

    def test_not_cached_when_expiry_within_buffer(self):
        # expires_in (30s) is inside TOKEN_EXPIRY_BUFFER_SECONDS (60s) -> never cached,
        # so every call re-fetches. Guards the ttl = expires_in - buffer arithmetic.
        client = _token_client(access_token="tok-short", expires_in=30)
        kwargs = dict(
            token_url="https://idp.test/short",
            client_id="cid-short",
            client_secret="s",
            http_client=client,
        )
        get_client_credentials_token(**kwargs)
        get_client_credentials_token(**kwargs)

        assert client.post.call_count == 2

    @pytest.mark.parametrize(
        "missing",
        [
            dict(token_url=None, client_id="c", client_secret="s"),
            dict(token_url="https://idp.test", client_id=None, client_secret="s"),
            dict(token_url="https://idp.test", client_id="c", client_secret=None),
        ],
    )
    def test_missing_required_inputs_raise(self, missing):
        with pytest.raises(OAuthClientCredentialsError):
            get_client_credentials_token(**missing)

    def test_response_without_access_token_raises(self):
        client = MagicMock()
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"error": "invalid_client"}
        client.post.return_value = response

        with pytest.raises(OAuthClientCredentialsError):
            get_client_credentials_token(
                token_url="https://idp.test",
                client_id="c",
                client_secret="s",
                http_client=client,
            )

    def test_none_response_raises(self):
        client = MagicMock()
        client.post.return_value = None
        with pytest.raises(OAuthClientCredentialsError):
            get_client_credentials_token(
                token_url="https://idp.test",
                client_id="c",
                client_secret="s",
                http_client=client,
            )

    def test_http_status_error_maps_to_oauth_error(self):
        request = httpx.Request("POST", "https://idp.test")
        response = httpx.Response(403, text="forbidden", request=request)
        client = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403", request=request, response=response
        )
        client.post.return_value = resp

        with pytest.raises(OAuthClientCredentialsError) as excinfo:
            get_client_credentials_token(
                token_url="https://idp.test",
                client_id="c",
                client_secret="s",
                http_client=client,
            )
        assert excinfo.value.status_code == 403

    def test_request_error_maps_to_oauth_error(self):
        client = MagicMock()
        client.post.side_effect = httpx.RequestError(
            "connection refused", request=httpx.Request("POST", "https://idp.test")
        )
        with pytest.raises(OAuthClientCredentialsError):
            get_client_credentials_token(
                token_url="https://idp.test",
                client_id="c",
                client_secret="s",
                http_client=client,
            )

    def test_different_secret_does_not_share_cache(self):
        # Regression: same token_url+client_id but a different client_secret must
        # not be served the first credential's cached token.
        c1 = _token_client(access_token="tok-A")
        c2 = _token_client(access_token="tok-B")
        first = get_client_credentials_token(
            token_url="https://idp.test/t",
            client_id="cid",
            client_secret="secret-A",
            http_client=c1,
        )
        second = get_client_credentials_token(
            token_url="https://idp.test/t",
            client_id="cid",
            client_secret="secret-B",
            http_client=c2,
        )
        assert first == "tok-A"
        assert second == "tok-B"
        c1.post.assert_called_once()
        c2.post.assert_called_once()


class TestCustomOAuthJSONConfig:
    def test_registered_with_oauth_auth_and_no_static_endpoint(self):
        provider = JSONProviderRegistry.get("custom_oauth")
        assert provider is not None
        assert provider.auth == "oauth2_client_credentials"
        assert provider.base_url is None
        assert provider.api_key_env is None

    def test_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _api_key, api_base = get_llm_provider(
            model="custom_oauth/gpt-4o",
            custom_llm_provider=None,
            api_base="https://gateway.test/v1",
            api_key=None,
        )
        assert model == "gpt-4o"
        assert provider == "custom_oauth"
        assert api_base == "https://gateway.test/v1"

    def test_provider_config_manager_returns_dynamic_config(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="gpt-4o", provider=LlmProviders.CUSTOM_OAUTH
        )
        assert config is not None
        assert config.custom_llm_provider == "custom_oauth"

    def test_get_complete_url_requires_api_base(self):
        config = create_config_class(JSONProviderRegistry.get("custom_oauth"))()
        with pytest.raises(ValueError):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model="gpt-4o",
                optional_params={},
                litellm_params={},
            )


class TestValidateEnvironment:
    def test_oauth_bearer_injected_from_litellm_params(self):
        config = create_config_class(JSONProviderRegistry.get("custom_oauth"))()
        client = _token_client(access_token="tok-ve")

        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=client,
        ):
            headers = config.validate_environment(
                headers={},
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={
                    "oauth_token_url": "https://idp.test/token",
                    "oauth_client_id": "cid-ve",
                    "oauth_client_secret": "secret-ve",
                    "oauth_scope": "scope-ve",
                },
            )

        assert headers["Authorization"] == "Bearer tok-ve"
        assert headers["Content-Type"] == "application/json"
        data = client.post.call_args.kwargs["data"]
        assert data["client_id"] == "cid-ve"
        assert data["client_secret"] == "secret-ve"
        assert data["scope"] == "scope-ve"

    def test_oauth_reads_from_env_when_litellm_params_absent(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_OAUTH_TOKEN_URL", "https://idp.env/token")
        monkeypatch.setenv("CUSTOM_OAUTH_CLIENT_ID", "cid-env")
        monkeypatch.setenv("CUSTOM_OAUTH_CLIENT_SECRET", "secret-env")
        config = create_config_class(JSONProviderRegistry.get("custom_oauth"))()
        client = _token_client(access_token="tok-env")

        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=client,
        ):
            headers = config.validate_environment(
                headers={},
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
            )

        assert headers["Authorization"] == "Bearer tok-env"
        assert client.post.call_args.args[0] == "https://idp.env/token"
        assert client.post.call_args.kwargs["data"]["client_id"] == "cid-env"

    def test_non_oauth_provider_still_uses_api_key_bearer(self):
        # Regression: existing JSON providers keep the inherited api_key -> Bearer path.
        config = create_config_class(JSONProviderRegistry.get("publicai"))()
        headers = config.validate_environment(
            headers={},
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-publicai",
        )
        assert headers["Authorization"] == "Bearer sk-publicai"


class TestCompletionRouting:
    def test_custom_oauth_routes_to_base_handler_with_oauth_params(self):
        """Regression: completion() must route custom_oauth through
        base_llm_http_handler (the default openai-compatible path needs a static
        api_key and would never run the OAuth validate_environment), and the
        per-model oauth_* must reach litellm_params. Mocked at the handler so it
        asserts routing + propagation without real network."""
        import litellm
        import litellm.main as main_mod
        from litellm.types.utils import ModelResponse

        fake = ModelResponse()
        with patch.object(
            main_mod.base_llm_http_handler, "completion", return_value=fake
        ) as handler:
            litellm.completion(
                model="custom_oauth/m",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://gateway.test/v1",
                oauth_token_url="https://idp.test/token",
                oauth_client_id="cid",
                oauth_client_secret="sec",
                oauth_scope="sc",
            )

        handler.assert_called_once()
        kwargs = handler.call_args.kwargs
        assert kwargs["custom_llm_provider"] == "custom_oauth"
        litellm_params = kwargs["litellm_params"]
        assert litellm_params["oauth_token_url"] == "https://idp.test/token"
        assert litellm_params["oauth_client_id"] == "cid"
        assert litellm_params["oauth_client_secret"] == "sec"
        assert litellm_params["oauth_scope"] == "sc"

    def test_custom_oauth_propagates_handler_error(self):
        """The custom_oauth branch logs and re-raises handler failures."""
        import litellm
        import litellm.main as main_mod

        with patch.object(
            main_mod.base_llm_http_handler,
            "completion",
            side_effect=RuntimeError("upstream boom"),
        ):
            with pytest.raises(Exception):
                litellm.completion(
                    model="custom_oauth/m",
                    messages=[{"role": "user", "content": "hi"}],
                    api_base="https://gateway.test/v1",
                    oauth_token_url="https://idp.test/token",
                    oauth_client_id="cid",
                    oauth_client_secret="sec",
                )
