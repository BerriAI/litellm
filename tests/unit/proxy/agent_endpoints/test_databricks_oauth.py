"""
Unit tests for Databricks App OAuth M2M support for A2A agents.

Covers config parsing (including os.environ/ resolution and validation),
workspace token-URL construction, client_credentials token fetching, caching
with expiry buffering, and the public ``resolve_databricks_app_auth_header``
helper.
"""

import base64
from unittest.mock import MagicMock, create_autospec, patch

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.agent_endpoints.databricks_oauth import (
    DatabricksAppOAuthConfig,
    DatabricksAppOAuthTokenCache,
    parse_databricks_oauth_config,
    resolve_databricks_app_auth_header,
)


def _expected_basic_auth(client_id: str, client_secret: str) -> str:
    token = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return f"Basic {token}"


def _mock_http_handler(access_token="tok-abc", expires_in=3600, post_error=None):
    """Return a mock that mirrors litellm's ``AsyncHTTPHandler`` contract.

    Two properties of the real handler matter for these tests and were the
    source of a runtime bug the original suite missed:

    1. ``post`` does not accept an ``auth`` kwarg. ``create_autospec`` enforces
       the real signature, so reintroducing HTTP Basic via ``auth=`` fails with
       ``TypeError`` instead of silently passing.
    2. ``post`` calls ``raise_for_status`` internally and raises
       ``httpx.HTTPStatusError`` itself on non-2xx; callers never inspect the
       returned response's status. Error-path tests therefore raise from
       ``post`` rather than from ``response.raise_for_status``.
    """
    handler = create_autospec(AsyncHTTPHandler, instance=True)
    if post_error is not None:
        handler.post.side_effect = post_error
    else:
        response = MagicMock()
        response.json = MagicMock(
            return_value={"access_token": access_token, "expires_in": expires_in}
        )
        handler.post.return_value = response
    return handler


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_parse_returns_none_without_block():
    assert parse_databricks_oauth_config(None) is None
    assert parse_databricks_oauth_config({}) is None
    assert parse_databricks_oauth_config({"other": "value"}) is None


def test_parse_builds_config_and_token_url():
    config = parse_databricks_oauth_config(
        {
            "databricks_oauth": {
                "client_id": "cid",
                "client_secret": "secret",
                "workspace_url": "https://dbc-abc.cloud.databricks.com",
            }
        }
    )
    assert config == DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc-abc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )


def test_parse_strips_serving_endpoints_and_trailing_slash():
    config = parse_databricks_oauth_config(
        {
            "databricks_oauth": {
                "client_id": "cid",
                "client_secret": "secret",
                "workspace_url": "https://dbc-abc.cloud.databricks.com/serving-endpoints/",
            }
        }
    )
    assert config is not None
    assert config.token_url == "https://dbc-abc.cloud.databricks.com/oidc/v1/token"


def test_parse_custom_scope():
    config = parse_databricks_oauth_config(
        {
            "databricks_oauth": {
                "client_id": "cid",
                "client_secret": "secret",
                "workspace_url": "https://dbc-abc.cloud.databricks.com",
                "scope": "custom-scope",
            }
        }
    )
    assert config is not None
    assert config.scope == "custom-scope"


@pytest.mark.parametrize(
    "missing_field", ["client_id", "client_secret", "workspace_url"]
)
def test_parse_raises_on_missing_field(missing_field):
    block = {
        "client_id": "cid",
        "client_secret": "secret",
        "workspace_url": "https://dbc-abc.cloud.databricks.com",
    }
    block.pop(missing_field)
    with pytest.raises(ValueError, match=missing_field):
        parse_databricks_oauth_config({"databricks_oauth": block})


def test_parse_raises_on_non_mapping_block():
    with pytest.raises(ValueError, match="mapping"):
        parse_databricks_oauth_config({"databricks_oauth": "not-a-dict"})


def test_parse_resolves_os_environ_references(monkeypatch):
    monkeypatch.setenv("MY_DBX_CLIENT_ID", "env-cid")
    monkeypatch.setenv("MY_DBX_SECRET", "env-secret")
    config = parse_databricks_oauth_config(
        {
            "databricks_oauth": {
                "client_id": "os.environ/MY_DBX_CLIENT_ID",
                "client_secret": "os.environ/MY_DBX_SECRET",
                "workspace_url": "https://dbc-abc.cloud.databricks.com",
            }
        }
    )
    assert config is not None
    assert config.client_id == "env-cid"
    assert config.client_secret == "env-secret"


# ---------------------------------------------------------------------------
# Token fetching + caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_token_posts_client_credentials_with_basic_auth():
    cache = DatabricksAppOAuthTokenCache()
    config = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    client = _mock_http_handler(access_token="tok-1")

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        token = await cache.async_get_token(config)

    assert token == "tok-1"
    client.post.assert_awaited_once()
    call = client.post.call_args
    assert call.args[0] == config.token_url
    assert call.kwargs["data"] == {
        "grant_type": "client_credentials",
        "scope": "all-apis",
    }
    # Databricks authenticates the client with HTTP Basic; it must be sent as a
    # header because litellm's AsyncHTTPHandler.post has no ``auth`` parameter.
    assert call.kwargs["headers"]["Authorization"] == _expected_basic_auth(
        "cid", "secret"
    )
    assert "auth" not in call.kwargs


@pytest.mark.asyncio
async def test_token_is_cached_across_calls():
    cache = DatabricksAppOAuthTokenCache()
    config = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    client = _mock_http_handler(access_token="tok-cached")

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        first = await cache.async_get_token(config)
        second = await cache.async_get_token(config)

    assert first == second == "tok-cached"
    client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_distinct_clients_do_not_share_token():
    cache = DatabricksAppOAuthTokenCache()
    config_a = DatabricksAppOAuthConfig(
        client_id="cid-a",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    config_b = DatabricksAppOAuthConfig(
        client_id="cid-b",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )

    clients = [_mock_http_handler("tok-a"), _mock_http_handler("tok-b")]

    def _next_client(*args, **kwargs):
        return clients.pop(0)

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        side_effect=_next_client,
    ):
        token_a = await cache.async_get_token(config_a)
        token_b = await cache.async_get_token(config_b)

    assert token_a == "tok-a"
    assert token_b == "tok-b"


@pytest.mark.asyncio
async def test_ttl_applies_expiry_buffer():
    cache = DatabricksAppOAuthTokenCache()
    config = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    client = _mock_http_handler(access_token="tok", expires_in=600)

    captured = {}
    real_set = cache.set_cache

    def _spy_set(key, value, **kwargs):
        captured["ttl"] = kwargs.get("ttl")
        return real_set(key, value, **kwargs)

    with (
        patch(
            "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
            return_value=client,
        ),
        patch.object(cache, "set_cache", side_effect=_spy_set),
    ):
        await cache.async_get_token(config)

    assert captured["ttl"] == 600 - 60


@pytest.mark.asyncio
async def test_missing_access_token_raises():
    cache = DatabricksAppOAuthTokenCache()
    config = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    client = _mock_http_handler()
    client.post.return_value.json.return_value = {"not_a_token": "x"}

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="access_token"):
            await cache.async_get_token(config)


def _config():
    return DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )


@pytest.mark.asyncio
async def test_http_status_error_raises_value_error():
    cache = DatabricksAppOAuthTokenCache()
    request = httpx.Request("POST", _config().token_url)
    error_response = httpx.Response(status_code=401, request=request)
    client = _mock_http_handler(
        post_error=httpx.HTTPStatusError(
            "unauthorized", request=request, response=error_response
        )
    )

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="status 401"):
            await cache.async_get_token(_config())


@pytest.mark.asyncio
async def test_transport_error_raises_value_error():
    cache = DatabricksAppOAuthTokenCache()
    client = _mock_http_handler(post_error=httpx.ConnectError("boom"))

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="token request failed"):
            await cache.async_get_token(_config())


@pytest.mark.asyncio
async def test_non_object_json_body_raises():
    cache = DatabricksAppOAuthTokenCache()
    client = _mock_http_handler()
    client.post.return_value.json.return_value = ["not", "an", "object"]

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="non-object JSON"):
            await cache.async_get_token(_config())


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in", [None, "not-a-number"])
async def test_invalid_expires_in_falls_back_to_default_ttl(expires_in):
    cache = DatabricksAppOAuthTokenCache()
    client = _mock_http_handler(expires_in=expires_in)

    captured = {}
    real_set = cache.set_cache

    def _spy_set(key, value, **kwargs):
        captured["ttl"] = kwargs.get("ttl")
        return real_set(key, value, **kwargs)

    with (
        patch(
            "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
            return_value=client,
        ),
        patch.object(cache, "set_cache", side_effect=_spy_set),
    ):
        await cache.async_get_token(_config())

    # default TTL (3600) minus the 60s expiry buffer
    assert captured["ttl"] == 3600 - 60


@pytest.mark.asyncio
async def test_short_lived_token_not_cached():
    """A token whose lifetime is below the refresh buffer is never cached and
    leaves no per-key lock behind."""
    cache = DatabricksAppOAuthTokenCache()
    config = _config()
    client = _mock_http_handler(access_token="short", expires_in=30)

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        await cache.async_get_token(config)
        await cache.async_get_token(config)

    assert cache.get_cache(config.cache_key) is None
    assert config.cache_key not in cache._locks
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_rotated_secret_forces_new_token():
    """Rotating client_secret changes the cache key so a fresh token is minted."""
    cache = DatabricksAppOAuthTokenCache()
    old = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="old-secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    rotated = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="new-secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    assert old.cache_key != rotated.cache_key

    clients = [_mock_http_handler("old-token"), _mock_http_handler("new-token")]

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        side_effect=lambda *a, **k: clients.pop(0),
    ):
        assert await cache.async_get_token(old) == "old-token"
        assert await cache.async_get_token(rotated) == "new-token"


@pytest.mark.asyncio
async def test_lock_pruned_when_token_evicted():
    """The per-key lock is removed when its cached token is deleted/evicted."""
    cache = DatabricksAppOAuthTokenCache()
    config = _config()
    client = _mock_http_handler("tok")

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        await cache.async_get_token(config)

    assert config.cache_key in cache._locks

    cache.delete_cache(config.cache_key)

    assert config.cache_key not in cache._locks


@pytest.mark.asyncio
async def test_flush_cache_clears_locks():
    """flush_cache drops the per-key locks alongside the cached tokens."""
    cache = DatabricksAppOAuthTokenCache()
    config = _config()
    client = _mock_http_handler("tok")

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        await cache.async_get_token(config)

    assert config.cache_key in cache._locks

    cache.flush_cache()

    assert cache._locks == {}
    assert cache.get_cache(config.cache_key) is None


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_returns_none_when_not_configured():
    assert await resolve_databricks_app_auth_header(None) is None
    assert await resolve_databricks_app_auth_header({"foo": "bar"}) is None


@pytest.mark.asyncio
async def test_resolve_returns_bearer_header():
    from litellm.proxy.agent_endpoints.databricks_oauth import (
        databricks_app_oauth_token_cache,
    )

    databricks_app_oauth_token_cache.flush_cache()

    litellm_params = {
        "databricks_oauth": {
            "client_id": "resolve-cid",
            "client_secret": "secret",
            "workspace_url": "https://resolve.cloud.databricks.com",
        }
    }
    client = _mock_http_handler(access_token="resolved-token")

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        header = await resolve_databricks_app_auth_header(litellm_params)

    assert header == {"Authorization": "Bearer resolved-token"}
