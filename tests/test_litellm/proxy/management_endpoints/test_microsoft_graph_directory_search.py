import asyncio
import os
import sys
import time

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.management_endpoints import microsoft_graph_directory_search as mgds


@pytest.fixture(autouse=True)
def _reset_token_cache():
    """Every test starts with a clean in-memory token cache."""
    mgds._microsoft_graph_token_cache.clear()
    yield
    mgds._microsoft_graph_token_cache.clear()


def _set_directory_env(
    monkeypatch,
    enabled="true",
    tenant="tenant-1",
    client_id="client-1",
    client_secret="secret-1",
):
    monkeypatch.setenv("MICROSOFT_DIRECTORY_SEARCH_ENABLED", enabled)
    if tenant is not None:
        monkeypatch.setenv("MICROSOFT_DIRECTORY_TENANT", tenant)
    if client_id is not None:
        monkeypatch.setenv("MICROSOFT_DIRECTORY_CLIENT_ID", client_id)
    if client_secret is not None:
        monkeypatch.setenv("MICROSOFT_DIRECTORY_CLIENT_SECRET", client_secret)


class TestEscapeFilterValue:
    def test_escapes_single_quote(self):
        assert mgds._escape_microsoft_graph_filter_value("O'Brien") == "O''Brien"

    def test_no_quotes_unchanged(self):
        assert mgds._escape_microsoft_graph_filter_value("alice") == "alice"

    def test_multiple_quotes(self):
        assert mgds._escape_microsoft_graph_filter_value("'''") == "''''''"


class TestIsMicrosoftDirectorySearchConfigured:
    def test_fully_configured(self, monkeypatch):
        _set_directory_env(monkeypatch)
        assert mgds.is_microsoft_directory_search_configured() is True

    def test_flag_disabled(self, monkeypatch):
        _set_directory_env(monkeypatch, enabled="false")
        assert mgds.is_microsoft_directory_search_configured() is False

    def test_flag_enabled_missing_client_id(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_DIRECTORY_SEARCH_ENABLED", "true")
        monkeypatch.setenv("MICROSOFT_DIRECTORY_TENANT", "tenant-1")
        monkeypatch.delenv("MICROSOFT_DIRECTORY_CLIENT_ID", raising=False)
        monkeypatch.delenv("MICROSOFT_CLIENT_ID", raising=False)
        monkeypatch.setenv("MICROSOFT_DIRECTORY_CLIENT_SECRET", "secret-1")
        assert mgds.is_microsoft_directory_search_configured() is False

    def test_falls_back_to_generic_microsoft_env_vars(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_DIRECTORY_SEARCH_ENABLED", "true")
        monkeypatch.delenv("MICROSOFT_DIRECTORY_TENANT", raising=False)
        monkeypatch.delenv("MICROSOFT_DIRECTORY_CLIENT_ID", raising=False)
        monkeypatch.delenv("MICROSOFT_DIRECTORY_CLIENT_SECRET", raising=False)
        monkeypatch.setenv("MICROSOFT_TENANT", "generic-tenant")
        monkeypatch.setenv("MICROSOFT_CLIENT_ID", "generic-client")
        monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "generic-secret")
        assert mgds.is_microsoft_directory_search_configured() is True

    def test_directory_specific_env_vars_take_precedence(self, monkeypatch):
        monkeypatch.setenv("MICROSOFT_DIRECTORY_TENANT", "directory-tenant")
        monkeypatch.setenv("MICROSOFT_TENANT", "generic-tenant")
        assert mgds._get_microsoft_directory_tenant() == "directory-tenant"

    def test_explicitly_empty_directory_var_does_not_fall_back_to_generic(
        self, monkeypatch
    ):
        """An explicitly-set-but-empty MICROSOFT_DIRECTORY_TENANT (e.g. an
        unresolved template variable) must not silently resolve to the
        generic MICROSOFT_TENANT used for SSO login."""
        monkeypatch.setenv("MICROSOFT_DIRECTORY_TENANT", "")
        monkeypatch.setenv("MICROSOFT_TENANT", "generic-tenant")
        assert mgds._get_microsoft_directory_tenant() is None


class TestGetMicrosoftGraphAccessToken:
    @pytest.mark.asyncio
    async def test_missing_credentials_raises_500(self, monkeypatch):
        monkeypatch.delenv("MICROSOFT_DIRECTORY_TENANT", raising=False)
        monkeypatch.delenv("MICROSOFT_TENANT", raising=False)
        monkeypatch.delenv("MICROSOFT_DIRECTORY_CLIENT_ID", raising=False)
        monkeypatch.delenv("MICROSOFT_CLIENT_ID", raising=False)
        monkeypatch.delenv("MICROSOFT_DIRECTORY_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("MICROSOFT_CLIENT_SECRET", raising=False)

        with pytest.raises(Exception) as exc_info:
            await mgds._get_microsoft_graph_access_token()
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_cold_cache_fetches_and_caches_token(self, monkeypatch, mocker):
        _set_directory_env(monkeypatch)
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "access_token": "brand-new-token",
            "expires_in": 3600,
        }
        mock_client = mocker.MagicMock()

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        token = await mgds._get_microsoft_graph_access_token()

        assert token == "brand-new-token"
        assert mgds._microsoft_graph_token_cache["access_token"] == "brand-new-token"

    @pytest.mark.asyncio
    async def test_warm_cache_skips_http_call(self, monkeypatch, mocker):
        _set_directory_env(monkeypatch)
        mgds._microsoft_graph_token_cache["access_token"] = "cached-token"
        mgds._microsoft_graph_token_cache["expires_at"] = time.time() + 3600

        mock_get_client = mocker.patch.object(mgds, "get_async_httpx_client")

        token = await mgds._get_microsoft_graph_access_token()

        assert token == "cached-token"
        mock_get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_cache_refetches(self, monkeypatch, mocker):
        _set_directory_env(monkeypatch)
        mgds._microsoft_graph_token_cache["access_token"] = "stale-token"
        # Already within the 60s expiry skew, so treated as expired.
        mgds._microsoft_graph_token_cache["expires_at"] = time.time() + 1

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "access_token": "refreshed-token",
            "expires_in": 3600,
        }
        mock_client = mocker.MagicMock()

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        token = await mgds._get_microsoft_graph_access_token()

        assert token == "refreshed-token"

    @pytest.mark.asyncio
    async def test_force_refresh_ignores_warm_cache(self, monkeypatch, mocker):
        _set_directory_env(monkeypatch)
        mgds._microsoft_graph_token_cache["access_token"] = "cached-token"
        mgds._microsoft_graph_token_cache["expires_at"] = time.time() + 3600

        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "access_token": "forced-refresh-token",
            "expires_in": 3600,
        }
        mock_client = mocker.MagicMock()

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        token = await mgds._get_microsoft_graph_access_token(force_refresh=True)

        assert token == "forced-refresh-token"

    @pytest.mark.asyncio
    async def test_missing_access_token_in_response_raises_500(
        self, monkeypatch, mocker
    ):
        _set_directory_env(monkeypatch)
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"expires_in": 3600}
        mock_client = mocker.MagicMock()

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        with pytest.raises(Exception) as exc_info:
            await mgds._get_microsoft_graph_access_token()
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_non_int_expires_in_falls_back_to_default(self, monkeypatch, mocker):
        _set_directory_env(monkeypatch)
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "access_token": "token-with-bad-expiry",
            "expires_in": "not-a-number",
        }
        mock_client = mocker.MagicMock()

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_client.post = mock_post
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        before = time.time()
        token = await mgds._get_microsoft_graph_access_token()
        assert token == "token-with-bad-expiry"
        # Falls back to the 3600s default rather than raising.
        assert mgds._microsoft_graph_token_cache["expires_at"] >= before + 3600

    @pytest.mark.asyncio
    async def test_concurrent_cold_cache_fetches_only_once(self, monkeypatch, mocker):
        """Concurrent callers on a cold cache must converge on a single
        Graph token request instead of each independently fetching."""
        _set_directory_env(monkeypatch)
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "access_token": "shared-token",
            "expires_in": 3600,
        }
        mock_client = mocker.MagicMock()
        post_call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal post_call_count
            post_call_count += 1
            await asyncio.sleep(0.01)
            return mock_response

        mock_client.post = mock_post
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        tokens = await asyncio.gather(
            *[mgds._get_microsoft_graph_access_token() for _ in range(5)]
        )

        assert tokens == ["shared-token"] * 5
        assert post_call_count == 1

    @pytest.mark.asyncio
    async def test_logs_azure_error_body_on_token_request_failure(
        self, monkeypatch, mocker
    ):
        """A bad/rotated client secret must not fail silently - the Azure
        error body (invalid_client, AADSTS...) should be logged."""
        _set_directory_env(monkeypatch)
        mock_400_response = mocker.MagicMock(
            status_code=400,
            text='{"error": "invalid_client", "error_description": "AADSTS7000215: Invalid client secret."}',
        )
        invalid_client_error = httpx.HTTPStatusError(
            "400", request=mocker.MagicMock(), response=mock_400_response
        )
        mock_client = mocker.MagicMock()

        async def mock_post(*args, **kwargs):
            raise invalid_client_error

        mock_client.post = mock_post
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)
        log_spy = mocker.patch.object(mgds.verbose_proxy_logger, "error")

        with pytest.raises(httpx.HTTPStatusError):
            await mgds._get_microsoft_graph_access_token()

        log_spy.assert_called_once()
        assert "AADSTS7000215" in log_spy.call_args.args[2]


class TestParseMicrosoftDirectoryUsers:
    def test_prefers_mail_over_upn(self):
        result = mgds._parse_microsoft_directory_users(
            {
                "value": [
                    {
                        "id": "1",
                        "displayName": "Alice",
                        "mail": "alice@example.com",
                        "userPrincipalName": "alice_upn@example.com",
                    }
                ]
            }
        )
        assert result[0].email == "alice@example.com"

    def test_falls_back_to_user_principal_name_when_mail_missing(self):
        """Guest/unlicensed AD accounts often have a null `mail` attribute."""
        result = mgds._parse_microsoft_directory_users(
            {
                "value": [
                    {
                        "id": "2",
                        "displayName": "Bob",
                        "mail": None,
                        "userPrincipalName": "bob@example.com",
                    }
                ]
            }
        )
        assert len(result) == 1
        assert result[0].email == "bob@example.com"

    def test_skips_record_missing_email_and_upn(self):
        result = mgds._parse_microsoft_directory_users(
            {"value": [{"id": "3", "displayName": "No Email", "mail": None}]}
        )
        assert result == []

    def test_skips_record_missing_id(self):
        result = mgds._parse_microsoft_directory_users(
            {"value": [{"displayName": "No Id", "mail": "noid@example.com"}]}
        )
        assert result == []

    def test_skips_non_dict_entries(self):
        result = mgds._parse_microsoft_directory_users({"value": ["not-a-dict", None]})
        assert result == []

    def test_empty_value_list(self):
        assert mgds._parse_microsoft_directory_users({}) == []


class TestGetMicrosoftGraphEndpoint:
    def test_default_endpoint(self, monkeypatch):
        monkeypatch.delenv("MICROSOFT_GRAPH_ENDPOINT", raising=False)
        assert (
            mgds._get_microsoft_graph_endpoint() == "https://graph.microsoft.com/v1.0"
        )

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(
            "MICROSOFT_GRAPH_ENDPOINT", "https://graph.microsoft.us/v1.0"
        )
        assert mgds._get_microsoft_graph_endpoint() == "https://graph.microsoft.us/v1.0"

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(
            "MICROSOFT_GRAPH_ENDPOINT", "https://graph.microsoft.us/v1.0/"
        )
        assert mgds._get_microsoft_graph_endpoint() == "https://graph.microsoft.us/v1.0"


class TestFetchMicrosoftDirectoryUsers:
    @pytest.mark.asyncio
    async def test_builds_request_with_escaped_filter_and_auth_header(
        self, monkeypatch, mocker
    ):
        monkeypatch.delenv("MICROSOFT_GRAPH_ENDPOINT", raising=False)
        mock_response = mocker.MagicMock()
        mock_client = mocker.MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        get_spy = mocker.patch.object(mock_client, "get", wraps=mock_get)
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        await mgds._fetch_microsoft_directory_users("O'Brien", "token-abc")

        get_spy.assert_called_once()
        _, kwargs = get_spy.call_args
        assert kwargs["headers"] == {
            "Authorization": "Bearer token-abc",
            "ConsistencyLevel": "eventual",
        }
        params = kwargs["params"]
        assert "startswith(displayName,'O''Brien')" in params["$filter"]
        assert "startswith(mail,'O''Brien')" in params["$filter"]
        assert "startswith(userPrincipalName,'O''Brien')" in params["$filter"]
        assert params["$select"] == "id,displayName,mail,userPrincipalName"
        assert params["$top"] == str(mgds.MICROSOFT_DIRECTORY_SEARCH_MAX_RESULTS)
        assert params["$count"] == "true"
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_url_uses_configured_graph_endpoint(
        self, monkeypatch, mocker
    ):
        monkeypatch.setenv(
            "MICROSOFT_GRAPH_ENDPOINT", "https://graph.microsoft.us/v1.0"
        )
        mock_response = mocker.MagicMock()
        mock_client = mocker.MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        get_spy = mocker.patch.object(mock_client, "get", wraps=mock_get)
        mocker.patch.object(mgds, "get_async_httpx_client", return_value=mock_client)

        await mgds._fetch_microsoft_directory_users("alice", "token-abc")

        args, _ = get_spy.call_args
        assert args[0] == "https://graph.microsoft.us/v1.0/users"


class TestSearchMicrosoftDirectoryUsers:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch, mocker):
        mocker.patch.object(
            mgds, "_get_microsoft_graph_access_token", return_value="token-123"
        )
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "value": [
                {
                    "id": "1",
                    "displayName": "Alice",
                    "mail": "alice@example.com",
                }
            ]
        }
        mocker.patch.object(
            mgds, "_fetch_microsoft_directory_users", return_value=mock_response
        )

        users = await mgds._search_microsoft_directory_users("alice")

        assert len(users) == 1
        assert users[0].email == "alice@example.com"

    @pytest.mark.asyncio
    async def test_retries_once_on_401_with_fresh_token(self, mocker):
        get_token = mocker.patch.object(
            mgds,
            "_get_microsoft_graph_access_token",
            side_effect=["stale-token", "fresh-token"],
        )
        mock_401_response = mocker.MagicMock(status_code=401)
        unauthorized_error = httpx.HTTPStatusError(
            "401", request=mocker.MagicMock(), response=mock_401_response
        )
        mock_success_response = mocker.MagicMock()
        mock_success_response.json.return_value = {"value": []}

        mocker.patch.object(
            mgds,
            "_fetch_microsoft_directory_users",
            side_effect=[unauthorized_error, mock_success_response],
        )

        users = await mgds._search_microsoft_directory_users("alice")

        assert users == []
        assert get_token.call_count == 2
        get_token.assert_called_with(force_refresh=True)

    @pytest.mark.asyncio
    async def test_non_401_http_status_error_propagates(self, mocker):
        mocker.patch.object(
            mgds, "_get_microsoft_graph_access_token", return_value="token-123"
        )
        mock_500_response = mocker.MagicMock(status_code=500)
        server_error = httpx.HTTPStatusError(
            "500", request=mocker.MagicMock(), response=mock_500_response
        )
        mocker.patch.object(
            mgds, "_fetch_microsoft_directory_users", side_effect=server_error
        )

        with pytest.raises(httpx.HTTPStatusError):
            await mgds._search_microsoft_directory_users("alice")
