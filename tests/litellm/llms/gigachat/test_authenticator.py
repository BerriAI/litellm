"""
Tests for litellm.llms.gigachat.authenticator
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../../"))

from litellm.llms.gigachat.authenticator import (
    GIGACHAT_AUTH_URL,
    GIGACHAT_SCOPE,
    GigaChatAuthError,
    _get_auth_url,
    _get_credentials,
    _get_scope,
    _parse_token_response,
    _request_token_async,
    _request_token_sync,
    get_access_token,
    get_access_token_async,
)


class TestParseTokenResponse:
    def test_parse_with_tok_and_exp(self):
        response = MagicMock()
        response.json.return_value = {"tok": "token123", "exp": 1234567890000}
        token, expires_at = _parse_token_response(response)
        assert token == "token123"
        assert expires_at == 1234567890000

    def test_parse_with_access_token_and_expires_at(self):
        response = MagicMock()
        response.json.return_value = {
            "access_token": "token456",
            "expires_at": 9876543210000,
        }
        token, expires_at = _parse_token_response(response)
        assert token == "token456"
        assert expires_at == 9876543210000

    def test_parse_with_string_expires_at(self):
        response = MagicMock()
        response.json.return_value = {
            "access_token": "token789",
            "expires_at": "1234567890000",
        }
        token, expires_at = _parse_token_response(response)
        assert token == "token789"
        assert expires_at == 1234567890000

    def test_parse_prefers_tok_over_access_token(self):
        response = MagicMock()
        response.json.return_value = {
            "tok": "preferred",
            "access_token": "fallback",
            "exp": 111111,
        }
        token, expires_at = _parse_token_response(response)
        assert token == "preferred"
        assert expires_at == 111111

    def test_parse_missing_token_raises(self):
        response = MagicMock()
        response.json.return_value = {"expires_at": 1234567890000}
        with pytest.raises(GigaChatAuthError) as exc_info:
            _parse_token_response(response)
        assert "Invalid token response" in str(exc_info.value)
        assert exc_info.value.status_code == 500


class TestGetCredentials:
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_credentials_from_gigachat_credentials(self, mock_get_secret):
        mock_get_secret.side_effect = lambda key: "cred123" if key == "GIGACHAT_CREDENTIALS" else None
        result = _get_credentials()
        assert result == "cred123"

    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_credentials_fallback_to_api_key(self, mock_get_secret):
        mock_get_secret.side_effect = lambda key: (
            "apikey456" if key == "GIGACHAT_API_KEY" else None
        )
        result = _get_credentials()
        assert result == "apikey456"

    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_credentials_returns_none(self, mock_get_secret):
        mock_get_secret.return_value = None
        result = _get_credentials()
        assert result is None


class TestGetAuthUrl:
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_auth_url_from_env(self, mock_get_secret):
        mock_get_secret.return_value = "https://custom.auth.url"
        result = _get_auth_url()
        assert result == "https://custom.auth.url"

    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_auth_url_default(self, mock_get_secret):
        mock_get_secret.return_value = None
        result = _get_auth_url()
        assert result == GIGACHAT_AUTH_URL


class TestGetScope:
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_scope_from_env(self, mock_get_secret):
        mock_get_secret.return_value = "CUSTOM_SCOPE"
        result = _get_scope()
        assert result == "CUSTOM_SCOPE"

    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_scope_default(self, mock_get_secret):
        mock_get_secret.return_value = None
        result = _get_scope()
        assert result == GIGACHAT_SCOPE


class TestRequestTokenSync:
    @patch("litellm.llms.gigachat.authenticator.uuid.uuid4")
    @patch("litellm.llms.gigachat.authenticator._get_http_client")
    def test_request_token_success(self, mock_get_client, mock_uuid):
        mock_uuid.return_value = "test-uuid-123"
        mock_response = MagicMock()
        mock_response.json.return_value = {"tok": "newtoken", "exp": 9999999999999}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        token, expires_at = _request_token_sync("creds", "SCOPE", "https://auth.url")

        assert token == "newtoken"
        assert expires_at == 9999999999999
        mock_client.post.assert_called_once_with(
            "https://auth.url",
            headers={
                "Authorization": "Basic creds",
                "RqUID": "test-uuid-123",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": "SCOPE"},
            timeout=30,
        )

    @patch("litellm.llms.gigachat.authenticator._get_http_client")
    def test_request_token_http_status_error(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.text = "Unauthorized"
        mock_response.status_code = 401
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_response,
        )

        with pytest.raises(GigaChatAuthError) as exc_info:
            _request_token_sync("creds", "SCOPE", "https://auth.url")
        assert exc_info.value.status_code == 401
        assert "Unauthorized" in str(exc_info.value)

    @patch("litellm.llms.gigachat.authenticator._get_http_client")
    def test_request_token_request_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.RequestError("Connection refused")
        mock_get_client.return_value = mock_client

        with pytest.raises(GigaChatAuthError) as exc_info:
            _request_token_sync("creds", "SCOPE", "https://auth.url")
        assert exc_info.value.status_code == 500
        assert "Connection refused" in str(exc_info.value)


class TestRequestTokenAsync:
    @patch("litellm.llms.gigachat.authenticator.uuid.uuid4")
    @patch("litellm.llms.gigachat.authenticator.get_async_httpx_client")
    @pytest.mark.asyncio
    async def test_request_token_async_success(self, mock_get_client, mock_uuid):
        mock_uuid.return_value = "test-uuid-456"
        mock_response = MagicMock()
        mock_response.json.return_value = {"tok": "async_token", "exp": 8888888888888}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        token, expires_at = await _request_token_async("creds", "SCOPE", "https://auth.url")

        assert token == "async_token"
        assert expires_at == 8888888888888
        mock_client.post.assert_awaited_once_with(
            "https://auth.url",
            headers={
                "Authorization": "Basic creds",
                "RqUID": "test-uuid-456",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": "SCOPE"},
            timeout=30,
        )

    @patch("litellm.llms.gigachat.authenticator.get_async_httpx_client")
    @pytest.mark.asyncio
    async def test_request_token_async_http_status_error(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.text = "Forbidden"
        mock_response.status_code = 403
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden",
            request=MagicMock(),
            response=mock_response,
        )

        with pytest.raises(GigaChatAuthError) as exc_info:
            await _request_token_async("creds", "SCOPE", "https://auth.url")
        assert exc_info.value.status_code == 403
        assert "Forbidden" in str(exc_info.value)

    @patch("litellm.llms.gigachat.authenticator.get_async_httpx_client")
    @pytest.mark.asyncio
    async def test_request_token_async_request_error(self, mock_get_client):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.RequestError("Timeout")
        mock_get_client.return_value = mock_client

        with pytest.raises(GigaChatAuthError) as exc_info:
            await _request_token_async("creds", "SCOPE", "https://auth.url")
        assert exc_info.value.status_code == 500
        assert "Timeout" in str(exc_info.value)


class TestGetAccessToken:
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_access_token_from_litellm_params(self, mock_get_secret):
        result = get_access_token(
            credentials=None,
            litellm_params={"gigachat_access_token": "param_token"},
        )
        assert result == "param_token"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_access_token_from_env(self, mock_get_secret):
        mock_get_secret.return_value = "env_token"
        result = get_access_token(
            credentials=None,
            litellm_params={},
        )
        assert result == "env_token"

    @patch("litellm.llms.gigachat.authenticator._request_token_sync")
    @patch("litellm.llms.gigachat.authenticator._token_cache")
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_access_token_from_cache_valid(self, mock_get_secret, mock_cache, mock_request):
        mock_get_secret.side_effect = lambda key: "creds" if key == "GIGACHAT_CREDENTIALS" else None
        mock_cache.get_cache.return_value = ("cached_token", 9999999999999)

        with patch("time.time", return_value=1000):
            result = get_access_token(credentials="creds", litellm_params={})

        assert result == "cached_token"
        mock_request.assert_not_called()

    @patch("litellm.llms.gigachat.authenticator._request_token_sync")
    @patch("litellm.llms.gigachat.authenticator._token_cache")
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_access_token_from_cache_expired(self, mock_get_secret, mock_cache, mock_request):
        mock_get_secret.side_effect = lambda key: "creds" if key == "GIGACHAT_CREDENTIALS" else None
        # token expired: 1,050,000 - 60,000 = 990,000 <= 1,000,000
        mock_cache.get_cache.return_value = ("expired_token", 1050000)
        mock_request.return_value = ("new_token", 2000000)

        with patch("time.time", return_value=1000):
            result = get_access_token(credentials="creds", litellm_params={})

        assert result == "new_token"
        mock_request.assert_called_once()
        mock_cache.set_cache.assert_called_once()

    @patch("litellm.llms.gigachat.authenticator._request_token_sync")
    @patch("litellm.llms.gigachat.authenticator._token_cache")
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_access_token_requests_new_and_caches(self, mock_get_secret, mock_cache, mock_request):
        mock_get_secret.side_effect = lambda key: "creds" if key == "GIGACHAT_CREDENTIALS" else None
        mock_cache.get_cache.return_value = None
        mock_request.return_value = ("fresh_token", 9999999999999)

        with patch("time.time", return_value=1000):
            result = get_access_token(credentials="creds", litellm_params={})

        assert result == "fresh_token"
        mock_request.assert_called_once_with("creds", GIGACHAT_SCOPE, GIGACHAT_AUTH_URL)
        mock_cache.set_cache.assert_called_once()
        # check cache key includes first 16 chars of credentials
        args, kwargs = mock_cache.set_cache.call_args
        assert args[0] == "gigachat_token:creds"
        assert args[1] == ("fresh_token", 9999999999999)

    def test_get_access_token_no_credentials_raises(self):
        with patch("litellm.llms.gigachat.authenticator.get_secret_str", return_value=None):
            with pytest.raises(GigaChatAuthError) as exc_info:
                get_access_token(credentials=None, litellm_params={})
        assert exc_info.value.status_code == 401
        assert "credentials not provided" in str(exc_info.value)

    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_access_token_custom_scope_and_auth_url(self, mock_get_secret):
        mock_get_secret.return_value = None
        with patch("litellm.llms.gigachat.authenticator._request_token_sync") as mock_request:
            mock_request.return_value = ("token", 9999999999999)
            with patch("litellm.llms.gigachat.authenticator._token_cache") as mock_cache:
                mock_cache.get_cache.return_value = None
                with patch("time.time", return_value=1000):
                    result = get_access_token(
                        credentials="creds",
                        scope="CUSTOM_SCOPE",
                        auth_url="https://custom.auth",
                        litellm_params={},
                    )
        assert result == "token"
        mock_request.assert_called_once_with("creds", "CUSTOM_SCOPE", "https://custom.auth")

    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    def test_get_access_token_scope_from_litellm_params(self, mock_get_secret):
        mock_get_secret.return_value = None
        with patch("litellm.llms.gigachat.authenticator._request_token_sync") as mock_request:
            mock_request.return_value = ("token", 9999999999999)
            with patch("litellm.llms.gigachat.authenticator._token_cache") as mock_cache:
                mock_cache.get_cache.return_value = None
                with patch("time.time", return_value=1000):
                    result = get_access_token(
                        credentials="creds",
                        litellm_params={"gigachat_scope": "PARAM_SCOPE", "gigachat_auth_url": "https://param.auth"},
                    )
        assert result == "token"
        mock_request.assert_called_once_with("creds", "PARAM_SCOPE", "https://param.auth")


class TestGetAccessTokenAsync:
    @pytest.mark.asyncio
    async def test_get_access_token_async_from_litellm_params(self):
        result = await get_access_token_async(
            credentials=None,
            litellm_params={"gigachat_access_token": "async_param_token"},
        )
        assert result == "async_param_token"

    @pytest.mark.asyncio
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    async def test_get_access_token_async_from_env(self, mock_get_secret):
        mock_get_secret.return_value = "async_env_token"
        result = await get_access_token_async(
            credentials=None,
            litellm_params={},
        )
        assert result == "async_env_token"

    @pytest.mark.asyncio
    @patch("litellm.llms.gigachat.authenticator._request_token_async")
    @patch("litellm.llms.gigachat.authenticator._token_cache")
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    async def test_get_access_token_async_from_cache_valid(self, mock_get_secret, mock_cache, mock_request):
        mock_get_secret.side_effect = lambda key: "creds" if key == "GIGACHAT_CREDENTIALS" else None
        mock_cache.get_cache.return_value = ("cached_async_token", 9999999999999)

        with patch("time.time", return_value=1000):
            result = await get_access_token_async(credentials="creds", litellm_params={})

        assert result == "cached_async_token"
        mock_request.assert_not_called()

    @pytest.mark.asyncio
    @patch("litellm.llms.gigachat.authenticator._request_token_async")
    @patch("litellm.llms.gigachat.authenticator._token_cache")
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    async def test_get_access_token_async_requests_new_and_caches(self, mock_get_secret, mock_cache, mock_request):
        mock_get_secret.side_effect = lambda key: "creds" if key == "GIGACHAT_CREDENTIALS" else None
        mock_cache.get_cache.return_value = None
        mock_request.return_value = ("fresh_async_token", 9999999999999)

        with patch("time.time", return_value=1000):
            result = await get_access_token_async(credentials="creds", litellm_params={})

        assert result == "fresh_async_token"
        mock_request.assert_awaited_once_with("creds", GIGACHAT_SCOPE, GIGACHAT_AUTH_URL)
        mock_cache.set_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_access_token_async_no_credentials_raises(self):
        with patch("litellm.llms.gigachat.authenticator.get_secret_str", return_value=None):
            with pytest.raises(GigaChatAuthError) as exc_info:
                await get_access_token_async(credentials=None, litellm_params={})
        assert exc_info.value.status_code == 401
        assert "credentials not provided" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("litellm.llms.gigachat.authenticator.get_secret_str")
    async def test_get_access_token_async_custom_params(self, mock_get_secret):
        mock_get_secret.return_value = None
        with patch("litellm.llms.gigachat.authenticator._request_token_async") as mock_request:
            mock_request.return_value = ("token", 9999999999999)
            with patch("litellm.llms.gigachat.authenticator._token_cache") as mock_cache:
                mock_cache.get_cache.return_value = None
                with patch("time.time", return_value=1000):
                    result = await get_access_token_async(
                        credentials="creds",
                        scope="CUSTOM",
                        auth_url="https://custom",
                        litellm_params={},
                    )
        assert result == "token"
        mock_request.assert_awaited_once_with("creds", "CUSTOM", "https://custom")
