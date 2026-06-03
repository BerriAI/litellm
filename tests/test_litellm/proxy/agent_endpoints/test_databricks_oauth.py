"""
Unit tests for Databricks App OAuth M2M support for A2A agents.

Covers config parsing (including os.environ/ resolution and validation),
workspace token-URL construction, client_credentials token fetching, caching
with expiry buffering, and the public ``resolve_databricks_app_auth_header``
helper.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.agent_endpoints.databricks_oauth import (
    DatabricksAppOAuthConfig,
    DatabricksAppOAuthTokenCache,
    parse_databricks_oauth_config,
    resolve_databricks_app_auth_header,
)


def _mock_httpx_client(access_token="tok-abc", expires_in=3600):
    """Return a mock async httpx client whose POST yields an OAuth token body."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(
        return_value={"access_token": access_token, "expires_in": expires_in}
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


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
async def test_fetch_token_posts_client_credentials():
    cache = DatabricksAppOAuthTokenCache()
    config = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    client = _mock_httpx_client(access_token="tok-1")

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
    assert call.kwargs["auth"] == ("cid", "secret")


@pytest.mark.asyncio
async def test_token_is_cached_across_calls():
    cache = DatabricksAppOAuthTokenCache()
    config = DatabricksAppOAuthConfig(
        client_id="cid",
        client_secret="secret",
        token_url="https://dbc.cloud.databricks.com/oidc/v1/token",
        scope="all-apis",
    )
    client = _mock_httpx_client(access_token="tok-cached")

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

    responses = iter([_mock_httpx_client("tok-a"), _mock_httpx_client("tok-b")])
    clients = list(responses)

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
    client = _mock_httpx_client(access_token="tok", expires_in=600)

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
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"not_a_token": "x"})
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        with pytest.raises(ValueError, match="access_token"):
            await cache.async_get_token(config)


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
    client = _mock_httpx_client(access_token="resolved-token")

    with patch(
        "litellm.proxy.agent_endpoints.databricks_oauth.get_async_httpx_client",
        return_value=client,
    ):
        header = await resolve_databricks_app_auth_header(litellm_params)

    assert header == {"Authorization": "Bearer resolved-token"}
