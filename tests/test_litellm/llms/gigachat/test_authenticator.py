"""
Unit tests for GigaChat OAuth authenticator.

Tests get_access_token and get_access_token_async covering token resolution
from litellm_params/env, credential validation, caching, and error handling.
"""

import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.gigachat import authenticator
from litellm.llms.gigachat.authenticator import (
    GigaChatAuthError,
    TOKEN_EXPIRY_BUFFER_MS,
    get_access_token,
    get_access_token_async,
)


AUTH_MODULE = "litellm.llms.gigachat.authenticator"


def _future_expires_at_ms(offset_seconds: float = 3600) -> int:
    return int(time.time() * 1000 + offset_seconds * 1000)


def _past_expires_at_ms(offset_seconds: float = 3600) -> int:
    return int(time.time() * 1000 - offset_seconds * 1000)


@pytest.fixture(autouse=True)
def _isolate_token_cache():
    """Each test gets a fresh module-level token cache to avoid cross-test leakage."""
    with patch(f"{AUTH_MODULE}._token_cache", new=MagicMock()):
        authenticator._token_cache.get_cache.return_value = None
        authenticator._token_cache.set_cache = MagicMock()
        yield


class TestGetAccessTokenSync:
    def test_returns_token_from_litellm_params(self):
        token = get_access_token(litellm_params={"gigachat_access_token": "param-token"})
        assert token == "param-token"
        authenticator._token_cache.get_cache.assert_not_called()

    @patch(f"{AUTH_MODULE}.get_secret_str")
    def test_returns_token_from_env(self, mock_get_secret):
        mock_get_secret.return_value = "env-access-token"
        token = get_access_token()
        assert token == "env-access-token"

    @patch(f"{AUTH_MODULE}._get_credentials", return_value=None)
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_raises_when_no_credentials(self, mock_get_secret, mock_get_creds):
        with pytest.raises(GigaChatAuthError) as exc_info:
            get_access_token()
        assert exc_info.value.status_code == 401
        assert "credentials not provided" in exc_info.value.message

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value=None)
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_raises_when_no_credentials_even_with_other_resolvers(
        self, mock_get_secret, mock_get_creds, mock_scope, mock_auth_url, mock_request
    ):
        with pytest.raises(GigaChatAuthError) as exc_info:
            get_access_token()
        assert exc_info.value.status_code == 401
        mock_request.assert_not_called()

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds-from-env")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_requests_new_token_and_caches(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        token = "fresh-token"
        expires_at = _future_expires_at_ms()
        mock_request.return_value = (token, expires_at)

        result = get_access_token()

        assert result == token
        mock_request.assert_called_once_with("creds-from-env", "GIGACHAT_API_PERS", "https://auth.example.com")
        authenticator._token_cache.set_cache.assert_called_once()
        call_args = authenticator._token_cache.set_cache.call_args
        assert call_args.args[1] == (token, expires_at)
        assert call_args.kwargs["ttl"] > 0

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_does_not_cache_when_no_expiry(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        mock_request.return_value = ("token-no-exp", 0)

        result = get_access_token()

        assert result == "token-no-exp"
        authenticator._token_cache.set_cache.assert_not_called()

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_does_not_cache_when_ttl_non_positive(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        expires_at = int(time.time() * 1000) + TOKEN_EXPIRY_BUFFER_MS - 1000
        mock_request.return_value = ("token", expires_at)

        result = get_access_token()

        assert result == "token"
        authenticator._token_cache.set_cache.assert_not_called()

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_returns_cached_valid_token(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        cached_token = "cached-token"
        cached_expires_at = _future_expires_at_ms(offset_seconds=7200)
        authenticator._token_cache.get_cache.return_value = (cached_token, cached_expires_at)

        result = get_access_token(credentials="creds")

        assert result == cached_token
        mock_request.assert_not_called()
        authenticator._token_cache.set_cache.assert_not_called()

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_requests_new_token_when_cache_expired(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        cached_token = "stale-token"
        cached_expires_at = _past_expires_at_ms(offset_seconds=10)
        authenticator._token_cache.get_cache.return_value = (cached_token, cached_expires_at)

        new_token = "refreshed-token"
        mock_request.return_value = (new_token, _future_expires_at_ms())

        result = get_access_token(credentials="creds")

        assert result == new_token
        mock_request.assert_called_once()

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://default-auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="env-creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_litellm_params_override_scope_and_auth_url(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        mock_request.return_value = ("token", _future_expires_at_ms())

        get_access_token(
            litellm_params={
                "gigachat_scope": "GIGACHAT_API_CORP",
                "gigachat_auth_url": "https://params-auth.example.com",
            }
        )

        mock_request.assert_called_once_with(
            "env-creds", "GIGACHAT_API_CORP", "https://params-auth.example.com"
        )

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://default-auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="env-creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_explicit_args_override_everything(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        mock_request.return_value = ("token", _future_expires_at_ms())

        get_access_token(
            credentials="explicit-creds",
            scope="EXPLICIT_SCOPE",
            auth_url="https://explicit.example.com",
            litellm_params={
                "gigachat_scope": "PARAM_SCOPE",
                "gigachat_auth_url": "https://params.example.com",
            },
        )

        mock_request.assert_called_once_with(
            "explicit-creds", "EXPLICIT_SCOPE", "https://explicit.example.com"
        )

    @patch(f"{AUTH_MODULE}._request_token_sync")
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    def test_propagates_auth_error_from_request(self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request):
        mock_request.side_effect = GigaChatAuthError(status_code=403, message="forbidden")

        with pytest.raises(GigaChatAuthError) as exc_info:
            get_access_token()
        assert exc_info.value.status_code == 403
        assert exc_info.value.message == "forbidden"


class TestGetAccessTokenAsync:
    @pytest.mark.asyncio
    async def test_returns_token_from_litellm_params(self):
        token = await get_access_token_async(
            litellm_params={"gigachat_access_token": "param-token"}
        )
        assert token == "param-token"
        authenticator._token_cache.get_cache.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}.get_secret_str")
    async def test_returns_token_from_env(self, mock_get_secret):
        mock_get_secret.return_value = "env-access-token"
        token = await get_access_token_async()
        assert token == "env-access-token"

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._get_credentials", return_value=None)
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_raises_when_no_credentials(self, mock_get_secret, mock_get_creds):
        with pytest.raises(GigaChatAuthError) as exc_info:
            await get_access_token_async()
        assert exc_info.value.status_code == 401
        assert "credentials not provided" in exc_info.value.message

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._request_token_async", new_callable=AsyncMock)
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds-from-env")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_requests_new_token_and_caches(
        self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request
    ):
        token = "fresh-token-async"
        expires_at = _future_expires_at_ms()
        mock_request.return_value = (token, expires_at)

        result = await get_access_token_async()

        assert result == token
        mock_request.assert_called_once_with(
            "creds-from-env", "GIGACHAT_API_PERS", "https://auth.example.com"
        )
        authenticator._token_cache.set_cache.assert_called_once()
        call_args = authenticator._token_cache.set_cache.call_args
        assert call_args.args[1] == (token, expires_at)
        assert call_args.kwargs["ttl"] > 0

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._request_token_async", new_callable=AsyncMock)
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_does_not_cache_when_no_expiry(
        self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request
    ):
        mock_request.return_value = ("token-no-exp", 0)

        result = await get_access_token_async()

        assert result == "token-no-exp"
        authenticator._token_cache.set_cache.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._request_token_async", new_callable=AsyncMock)
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_returns_cached_valid_token(
        self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request
    ):
        cached_token = "cached-token-async"
        cached_expires_at = _future_expires_at_ms(offset_seconds=7200)
        authenticator._token_cache.get_cache.return_value = (cached_token, cached_expires_at)

        result = await get_access_token_async(credentials="creds")

        assert result == cached_token
        mock_request.assert_not_called()
        authenticator._token_cache.set_cache.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._request_token_async", new_callable=AsyncMock)
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_requests_new_token_when_cache_expired(
        self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request
    ):
        cached_expires_at = _past_expires_at_ms(offset_seconds=10)
        authenticator._token_cache.get_cache.return_value = ("stale", cached_expires_at)

        new_token = "refreshed-token-async"
        mock_request.return_value = (new_token, _future_expires_at_ms())

        result = await get_access_token_async(credentials="creds")

        assert result == new_token
        mock_request.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._request_token_async", new_callable=AsyncMock)
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://default-auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="env-creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_litellm_params_override_scope_and_auth_url(
        self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request
    ):
        mock_request.return_value = ("token", _future_expires_at_ms())

        await get_access_token_async(
            litellm_params={
                "gigachat_scope": "GIGACHAT_API_CORP",
                "gigachat_auth_url": "https://params-auth.example.com",
            }
        )

        mock_request.assert_called_once_with(
            "env-creds", "GIGACHAT_API_CORP", "https://params-auth.example.com"
        )

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._request_token_async", new_callable=AsyncMock)
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://default-auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="env-creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_explicit_args_override_everything(
        self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request
    ):
        mock_request.return_value = ("token", _future_expires_at_ms())

        await get_access_token_async(
            credentials="explicit-creds",
            scope="EXPLICIT_SCOPE",
            auth_url="https://explicit.example.com",
            litellm_params={
                "gigachat_scope": "PARAM_SCOPE",
                "gigachat_auth_url": "https://params.example.com",
            },
        )

        mock_request.assert_called_once_with(
            "explicit-creds", "EXPLICIT_SCOPE", "https://explicit.example.com"
        )

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}._request_token_async", new_callable=AsyncMock)
    @patch(f"{AUTH_MODULE}._get_auth_url", return_value="https://auth.example.com")
    @patch(f"{AUTH_MODULE}._get_scope", return_value="GIGACHAT_API_PERS")
    @patch(f"{AUTH_MODULE}._get_credentials", return_value="creds")
    @patch(f"{AUTH_MODULE}.get_secret_str", return_value=None)
    async def test_propagates_auth_error_from_request(
        self, mock_get_secret, mock_creds, mock_scope, mock_auth_url, mock_request
    ):
        mock_request.side_effect = GigaChatAuthError(status_code=403, message="forbidden")

        with pytest.raises(GigaChatAuthError) as exc_info:
            await get_access_token_async()
        assert exc_info.value.status_code == 403
        assert exc_info.value.message == "forbidden"


class TestRequestTokenSyncErrorMapping:
    @patch(f"{AUTH_MODULE}._get_http_client")
    def test_http_status_error_maps_to_auth_error(self, mock_get_client):
        client = MagicMock()
        request = httpx.Request("POST", "https://auth.example.com")
        response = httpx.Response(status_code=401, content=b"bad creds", request=request)
        http_error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
        client.post.side_effect = http_error
        mock_get_client.return_value = client

        from litellm.llms.gigachat.authenticator import _request_token_sync

        with pytest.raises(GigaChatAuthError) as exc_info:
            _request_token_sync("creds", "GIGACHAT_API_PERS", "https://auth.example.com")
        assert exc_info.value.status_code == 401
        assert "bad creds" in exc_info.value.message

    @patch(f"{AUTH_MODULE}._get_http_client")
    def test_request_error_maps_to_auth_error(self, mock_get_client):
        client = MagicMock()
        client.post.side_effect = httpx.ConnectError("connection refused")
        mock_get_client.return_value = client

        from litellm.llms.gigachat.authenticator import _request_token_sync

        with pytest.raises(GigaChatAuthError) as exc_info:
            _request_token_sync("creds", "GIGACHAT_API_PERS", "https://auth.example.com")
        assert exc_info.value.status_code == 500
        assert "connection refused" in exc_info.value.message


class TestRequestTokenAsyncErrorMapping:
    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}.get_async_httpx_client")
    async def test_http_status_error_maps_to_auth_error(self, mock_get_client):
        client = MagicMock()
        request = httpx.Request("POST", "https://auth.example.com")
        response = httpx.Response(status_code=401, content=b"bad creds", request=request)
        http_error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
        client.post = AsyncMock(side_effect=http_error)
        mock_get_client.return_value = client

        from litellm.llms.gigachat.authenticator import _request_token_async

        with pytest.raises(GigaChatAuthError) as exc_info:
            await _request_token_async("creds", "GIGACHAT_API_PERS", "https://auth.example.com")
        assert exc_info.value.status_code == 401
        assert "bad creds" in exc_info.value.message

    @pytest.mark.asyncio
    @patch(f"{AUTH_MODULE}.get_async_httpx_client")
    async def test_request_error_maps_to_auth_error(self, mock_get_client):
        client = MagicMock()
        client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_get_client.return_value = client

        from litellm.llms.gigachat.authenticator import _request_token_async

        with pytest.raises(GigaChatAuthError) as exc_info:
            await _request_token_async("creds", "GIGACHAT_API_PERS", "https://auth.example.com")
        assert exc_info.value.status_code == 500
        assert "connection refused" in exc_info.value.message


class TestParseTokenResponse:
    def _make_response(self, body: dict) -> httpx.Response:
        import json

        return httpx.Response(
            status_code=200,
            content=json.dumps(body).encode("utf-8"),
            request=httpx.Request("POST", "https://auth.example.com"),
        )

    def test_parses_tok_exp_fields(self):
        from litellm.llms.gigachat.authenticator import _parse_token_response

        token, expires_at = _parse_token_response(
            self._make_response({"tok": "abc", "exp": 1700000000000})
        )
        assert token == "abc"
        assert expires_at == 1700000000000

    def test_parses_access_token_expires_at_fields(self):
        from litellm.llms.gigachat.authenticator import _parse_token_response

        token, expires_at = _parse_token_response(
            self._make_response({"access_token": "xyz", "expires_at": 1700000000000})
        )
        assert token == "xyz"
        assert expires_at == 1700000000000

    def test_parses_string_expires_at(self):
        from litellm.llms.gigachat.authenticator import _parse_token_response

        token, expires_at = _parse_token_response(
            self._make_response({"tok": "abc", "exp": "1700000000000"})
        )
        assert token == "abc"
        assert expires_at == 1700000000000
        assert isinstance(expires_at, int)

    def test_raises_when_no_access_token(self):
        from litellm.llms.gigachat.authenticator import _parse_token_response

        with pytest.raises(GigaChatAuthError) as exc_info:
            _parse_token_response(self._make_response({"exp": 1700000000000}))
        assert exc_info.value.status_code == 500
        assert "Invalid token response" in exc_info.value.message
