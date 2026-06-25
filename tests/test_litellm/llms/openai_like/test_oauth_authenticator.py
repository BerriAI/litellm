"""Tests for the oauth_client_credentials flag: OAuth2 client_credentials token
fetch + caching, the litellm_params resolver, and bearer injection on the
OpenAI-compatible completion path.

Regression coverage for https://github.com/BerriAI/litellm/issues/12367
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.openai_like.oauth_authenticator import (
    OAuthClientCredentialsError,
    _token_cache,
    get_client_credentials_token,
    resolve_client_credentials_token,
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
        with pytest.raises(OAuthClientCredentialsError) as exc_info:
            get_client_credentials_token(**missing)

        message = exc_info.value.message
        assert "litellm_params" in message
        assert "os.environ/" in message
        assert "CUSTOM_OAUTH_" not in message

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

    def test_non_json_200_body_maps_to_oauth_error(self):
        # Regression: a 200 with a non-JSON body (e.g. an HTML login page from an
        # intermediate proxy) makes response.json() raise JSONDecodeError (a
        # ValueError subclass, not a ValidationError). It must surface as the
        # documented OAuthClientCredentialsError, not a raw JSONDecodeError.
        request = httpx.Request("POST", "https://idp.test")
        response = httpx.Response(200, text="<html>login</html>", request=request)
        client = MagicMock()
        client.post.return_value = response

        with pytest.raises(OAuthClientCredentialsError) as exc_info:
            get_client_credentials_token(
                token_url="https://idp.test",
                client_id="c",
                client_secret="s",
                http_client=client,
            )
        assert exc_info.value.status_code == 500
        assert "access_token" in exc_info.value.message

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


class TestResolveClientCredentialsToken:
    def test_flag_on_with_creds_returns_token(self):
        client = _token_client(access_token="tok-r")
        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=client,
        ):
            token = resolve_client_credentials_token(
                {
                    "oauth_client_credentials": True,
                    "oauth_token_url": "https://idp.test/token",
                    "oauth_client_id": "cid-r",
                    "oauth_client_secret": "secret-r",
                    "oauth_scope": "scope-r",
                }
            )

        assert token == "tok-r"
        data = client.post.call_args.kwargs["data"]
        assert data["client_id"] == "cid-r"
        assert data["client_secret"] == "secret-r"
        assert data["scope"] == "scope-r"

    def test_flag_absent_returns_none_without_fetch(self):
        # Creds present but no flag -> OAuth must not engage; the deployment keeps
        # its configured api_key. Kills a mutant that triggers on creds presence.
        client = _token_client()
        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=client,
        ):
            result = resolve_client_credentials_token(
                {
                    "oauth_token_url": "https://idp.test/token",
                    "oauth_client_id": "cid",
                    "oauth_client_secret": "secret",
                }
            )

        assert result is None
        client.post.assert_not_called()

    def test_flag_false_returns_none(self):
        assert (
            resolve_client_credentials_token({"oauth_client_credentials": False})
            is None
        )

    def test_flag_on_missing_cred_raises(self):
        with pytest.raises(OAuthClientCredentialsError):
            resolve_client_credentials_token(
                {
                    "oauth_client_credentials": True,
                    "oauth_token_url": "https://idp.test/token",
                    "oauth_client_id": "cid",
                }
            )

    def test_env_vars_are_not_consulted(self, monkeypatch):
        # Creds live ONLY in env; the resolver reads litellm_params exclusively, so
        # with the flag on but creds absent from litellm_params it must raise and
        # never mint a token from env. This is the exfiltration path the security
        # review closed: env-configured creds bypassing the clear-on-override.
        monkeypatch.setenv("CUSTOM_OAUTH_TOKEN_URL", "https://idp.env/token")
        monkeypatch.setenv("CUSTOM_OAUTH_CLIENT_ID", "cid-env")
        monkeypatch.setenv("CUSTOM_OAUTH_CLIENT_SECRET", "secret-env")
        client = _token_client(access_token="tok-env")

        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OAuthClientCredentialsError):
                resolve_client_credentials_token({"oauth_client_credentials": True})

        client.post.assert_not_called()

    def test_no_token_after_clientside_base_override_clear(self):
        # End-to-end with the router: a client api_base override clears the
        # deployment's oauth_client_credentials flag + creds, so the resolver
        # returns None (graceful fallback to the configured api_key) and no admin
        # bearer is minted for the client-controlled upstream.
        from litellm.router_utils.clientside_credential_handler import (
            get_dynamic_litellm_params,
        )

        cleared = get_dynamic_litellm_params(
            {
                "oauth_client_credentials": True,
                "oauth_token_url": "https://idp.internal/token",
                "oauth_client_id": "admin-id",
                "oauth_client_secret": "admin-secret",
                "oauth_scope": "admin-scope",
            },
            {"api_base": "https://client.example/v1"},
        )

        client = _token_client(access_token="tok-leak")
        with patch(
            "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
            return_value=client,
        ):
            assert resolve_client_credentials_token(cleared) is None

        client.post.assert_not_called()


class TestCompletionInjection:
    def test_flag_injects_minted_token_as_api_key(self):
        # End-to-end through litellm.completion on a plain openai/ model: the flag
        # plus oauth_* in litellm_params mint a bearer that overrides the
        # configured api_key, which the OpenAI SDK then sends as Authorization:
        # Bearer. Fails if the injection in _complete_custom_openai is removed or
        # the flag stops reaching litellm_params.
        import litellm
        import litellm.main as main_mod
        from litellm.types.utils import ModelResponse

        token_client = _token_client(access_token="tok-e2e")
        fake = ModelResponse()
        with (
            patch(
                "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
                return_value=token_client,
            ),
            patch.object(
                main_mod.openai_chat_completions, "completion", return_value=fake
            ) as chat,
        ):
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://gateway.test/v1",
                api_key="sk-should-be-overridden",
                oauth_client_credentials=True,
                oauth_token_url="https://idp.test/token",
                oauth_client_id="cid-e2e",
                oauth_client_secret="secret-e2e",
                oauth_scope="scope-e2e",
            )

        token_client.post.assert_called_once()
        assert chat.call_args.kwargs["api_key"] == "tok-e2e"

    def test_no_flag_keeps_configured_api_key(self):
        # Without the flag, even with oauth_* present, no token is fetched and the
        # configured api_key is sent unchanged.
        import litellm
        import litellm.main as main_mod
        from litellm.types.utils import ModelResponse

        token_client = _token_client(access_token="should-not-be-used")
        fake = ModelResponse()
        with (
            patch(
                "litellm.llms.openai_like.oauth_authenticator._get_httpx_client",
                return_value=token_client,
            ),
            patch.object(
                main_mod.openai_chat_completions, "completion", return_value=fake
            ) as chat,
        ):
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://gateway.test/v1",
                api_key="sk-static",
                oauth_token_url="https://idp.test/token",
                oauth_client_id="cid",
                oauth_client_secret="secret",
            )

        token_client.post.assert_not_called()
        assert chat.call_args.kwargs["api_key"] == "sk-static"
