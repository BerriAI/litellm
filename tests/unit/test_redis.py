import inspect
import json
import os
import ssl
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
import redis.asyncio as async_redis
from redis.asyncio.connection import AbstractConnection
from redis.credentials import CredentialProvider
from redis.exceptions import RedisClusterException

import litellm
from litellm._redis import (
    _AWS_IAM_KWARG_NAMES,
    _coerce_redis_kwargs_types,
    _credential_provider_auth_kwargs,
    _get_redis_client_logic,
    _get_redis_cluster_kwargs,
    _get_redis_env_kwarg_mapping,
    _get_redis_kwargs,
    _get_redis_url_kwargs,
    _pretty_print_redis_config,
    _uses_tls,
    get_redis_async_client,
    get_redis_client,
    get_redis_connection_pool,
    get_redis_url_from_environment,
)
from litellm._redis_credential_provider import (
    AzureADCredentialProvider,
    ElastiCacheIAMCredentialProvider,
    GCPIAMCredentialProvider,
    _token_cache,
)
from litellm.caching.redis_cache import RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.constants import REDIS_CLUSTER_HEALTH_CHECK_INTERVAL
from litellm.proxy._types import CoordinationRedisParams


class _StubCredentialProvider(CredentialProvider):
    def __init__(self, token: str = "stub-token") -> None:
        self._token = token

    def get_credentials(self):
        return (self._token,)

    async def get_credentials_async(self):
        return (self._token,)


class _HostileCredentialProvider(CredentialProvider):
    def __init__(self, secret: str) -> None:
        self._payload = secret

    def get_credentials(self):
        return (self._payload,)

    async def get_credentials_async(self):
        return (self._payload,)

    def __repr__(self):
        raise AssertionError("provider repr must never be invoked")

    def __str__(self):
        raise AssertionError("provider str must never be invoked")

    def __reduce__(self):
        raise AssertionError("provider must never be serialized")

    def __getstate__(self):
        raise AssertionError("provider state must never be inspected")


def _gcp_marker_callback() -> MagicMock:
    callback = MagicMock()
    callback._gcp_service_account = "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"
    return callback


@pytest.fixture
def clean_redis_environment(monkeypatch):
    for var in (
        "REDIS_URL",
        "REDIS_CLUSTER_NODES",
        "REDIS_SENTINEL_NODES",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        *_get_redis_env_kwarg_mapping(),
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def clear_llm_client_cache():
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(autouse=True)
def clear_gcp_iam_token_cache():
    """Reset the module-level GCP IAM token cache between tests."""
    _token_cache.clear()
    yield
    _token_cache.clear()


def test_redis_allowlists_include_credential_provider():
    assert "credential_provider" in _get_redis_kwargs()
    assert "credential_provider" in _get_redis_url_kwargs()
    assert "credential_provider" in _get_redis_cluster_kwargs()


def test_credential_provider_is_not_environment_derived():
    mapping = _get_redis_env_kwarg_mapping()
    assert "REDIS_CREDENTIAL_PROVIDER" not in mapping
    assert "credential_provider" not in mapping.values()


_AWS_IAM_SETTINGS = {
    "aws_iam_auth",
    "aws_iam_user_name",
    "aws_iam_cache_name",
    "aws_iam_region",
    "aws_iam_serverless",
}


def test_aws_iam_settings_are_environment_derived():
    allowed = _get_redis_kwargs()
    mapping = _get_redis_env_kwarg_mapping()

    assert _AWS_IAM_SETTINGS <= allowed
    assert set(_AWS_IAM_KWARG_NAMES) == _AWS_IAM_SETTINGS
    assert {f for f in CoordinationRedisParams.model_fields if f.startswith("aws_iam_")} == _AWS_IAM_SETTINGS
    assert mapping["REDIS_AWS_IAM_AUTH"] == "aws_iam_auth"
    assert mapping["REDIS_AWS_IAM_USER_NAME"] == "aws_iam_user_name"
    assert mapping["REDIS_AWS_IAM_CACHE_NAME"] == "aws_iam_cache_name"
    assert mapping["REDIS_AWS_IAM_REGION"] == "aws_iam_region"
    assert mapping["REDIS_AWS_IAM_SERVERLESS"] == "aws_iam_serverless"


def test_sync_direct_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()

    client = get_redis_client(host="redis-host", port=6379, credential_provider=provider)

    assert client.connection_pool.connection_kwargs["credential_provider"] is provider


def test_sync_direct_provider_supersedes_static_credentials(clean_redis_environment):
    provider = _StubCredentialProvider()

    client = get_redis_client(
        host="redis-host",
        port=6379,
        username="redis-user",
        password="redis-password",
        credential_provider=provider,
    )
    connection = client.connection_pool.make_connection()

    assert connection.credential_provider is provider
    assert connection.username is None
    assert connection.password is None


def test_sync_direct_provider_supersedes_environment_credentials(clean_redis_environment, monkeypatch):
    provider = _StubCredentialProvider()
    monkeypatch.setenv("REDIS_USERNAME", "redis-user")
    monkeypatch.setenv("REDIS_PASSWORD", "redis-password")

    client = get_redis_client(host="redis-host", port=6379, credential_provider=provider)
    connection = client.connection_pool.make_connection()

    assert connection.credential_provider is provider
    assert connection.username is None
    assert connection.password is None


def test_sync_url_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()

    client = get_redis_client(url="redis://redis-host:6379", credential_provider=provider)

    assert client.connection_pool.connection_kwargs["credential_provider"] is provider


def test_async_direct_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()

    client = get_redis_async_client(host="redis-host", port=6379, credential_provider=provider)

    assert client.connection_pool.connection_kwargs["credential_provider"] is provider


def test_async_url_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()

    client = get_redis_async_client(url="redis://redis-host:6379", credential_provider=provider)

    assert client.connection_pool.connection_kwargs["credential_provider"] is provider


def test_sync_url_credentials_do_not_replace_explicit_provider(clean_redis_environment):
    provider = _StubCredentialProvider()

    client = get_redis_client(
        url="redis://url-user:url-pass@redis-host:6379",
        credential_provider=provider,
    )
    connection = client.connection_pool.make_connection()

    assert connection.credential_provider is provider
    assert connection.username is None
    assert connection.password is None


def test_async_url_credentials_do_not_replace_explicit_provider(clean_redis_environment):
    provider = _StubCredentialProvider()

    client = get_redis_async_client(
        url="redis://url-user:url-pass@redis-host:6379",
        credential_provider=provider,
    )
    connection = client.connection_pool.make_connection()

    assert connection.credential_provider is provider
    assert connection.username is None
    assert connection.password is None


def test_async_host_port_pool_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()

    pool = get_redis_connection_pool(host="redis-host", port=6379, credential_provider=provider)

    assert pool is not None
    assert pool.connection_kwargs["credential_provider"] is provider


def test_async_url_pool_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()

    pool = get_redis_connection_pool(url="redis://redis-host:6379", credential_provider=provider)

    assert pool is not None
    assert pool.connection_kwargs["credential_provider"] is provider


def test_async_url_pool_strips_userinfo_for_the_provider(clean_redis_environment):
    provider = _StubCredentialProvider()

    pool = get_redis_connection_pool(url="rediss://url-user:url-pass@redis-host:6379/3", credential_provider=provider)

    connection = pool.make_connection()
    assert connection.credential_provider is provider
    assert connection.username is None
    assert connection.password is None
    assert connection.db == 3


def test_sync_cluster_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()
    startup_nodes = [{"host": "cluster-node", "port": 6379}]

    with patch("redis.RedisCluster", autospec=True) as mock_cluster_cls:
        get_redis_client(startup_nodes=startup_nodes, credential_provider=provider, password="redis-secret")

    cluster_kwargs = mock_cluster_cls.call_args.kwargs
    assert cluster_kwargs["credential_provider"] is provider
    assert "password" not in cluster_kwargs
    assert [(node.host, node.port) for node in cluster_kwargs["startup_nodes"]] == [("cluster-node", 6379)]


def test_sync_cluster_authenticates_with_azure_credentials(clean_redis_environment, monkeypatch):
    monkeypatch.setenv("REDIS_USERNAME", "identity-object-id")
    credential = MagicMock()
    credential.get_token.return_value = SimpleNamespace(token="azure-access-token")

    with (
        patch("azure.identity.DefaultAzureCredential", return_value=credential),
        patch("redis.RedisCluster", autospec=True) as cluster,
    ):
        get_redis_client(
            startup_nodes=[{"host": "cluster-node", "port": 6379}],
            azure_redis_ad_token=True,
            password="stale-password",
        )

    kwargs = cluster.call_args.kwargs
    provider = kwargs.get("credential_provider")
    assert isinstance(provider, AzureADCredentialProvider)
    assert provider.get_credentials() == ("identity-object-id", "azure-access-token")
    assert "username" not in kwargs
    assert "password" not in kwargs
    assert "redis_connect_func" not in kwargs
    credential.get_token.assert_called_once_with("https://redis.azure.com/.default")


def test_sync_cluster_authenticates_with_gcp_credentials(clean_redis_environment):
    with patch("redis.RedisCluster", autospec=True) as cluster:
        get_redis_client(
            startup_nodes=[{"host": "cluster-node", "port": 6379}],
            redis_connect_func=_gcp_marker_callback(),
            username="stale-user",
            password="stale-password",
        )

    kwargs = cluster.call_args.kwargs
    assert isinstance(kwargs.get("credential_provider"), GCPIAMCredentialProvider)
    assert "username" not in kwargs
    assert "password" not in kwargs
    assert "redis_connect_func" not in kwargs


def test_async_cluster_preserves_credential_provider_identity(clean_redis_environment):
    provider = _StubCredentialProvider()
    startup_nodes = [{"host": "cluster-node", "port": 6379}]

    client = get_redis_async_client(startup_nodes=startup_nodes, credential_provider=provider)

    assert client.connection_kwargs["credential_provider"] is provider
    assert client.connection_kwargs["socket_keepalive"] is True
    assert client.connection_kwargs["health_check_interval"] == REDIS_CLUSTER_HEALTH_CHECK_INTERVAL


def test_explicit_provider_skips_automatic_auth_and_callback(clean_redis_environment, monkeypatch):
    provider = _StubCredentialProvider()
    monkeypatch.setenv("REDIS_GCP_SERVICE_ACCOUNT", "service-account@example.com")
    monkeypatch.setenv("REDIS_AZURE_AD_TOKEN", "true")

    with (
        patch(  # test-quality-ok: an auto-auth callback built here is popped again by the provider branch, so the builders are the only place the wasted work is visible
            "litellm._redis.create_gcp_iam_redis_connect_func"
        ) as mock_gcp,
        patch(  # test-quality-ok: same as above, and reaching this one also builds an Azure credential the caller never asked for
            "litellm._redis.create_azure_ad_redis_connect_func"
        ) as mock_azure,
    ):
        redis_kwargs = _get_redis_client_logic(
            host="redis-host",
            port=6379,
            credential_provider=provider,
            redis_connect_func=_gcp_marker_callback(),
        )

    mock_gcp.assert_not_called()
    mock_azure.assert_not_called()
    assert redis_kwargs["credential_provider"] is provider
    assert "redis_connect_func" not in redis_kwargs


@pytest.mark.parametrize(
    "overrides",
    [
        {"gcp_ssl_ca_certs": "/tmp/ca.pem"},
        {"gcp_service_account": "sa@example.com", "gcp_ssl_ca_certs": "/tmp/ca.pem"},
    ],
    ids=["certs-without-service-account", "both-alongside-a-provider"],
)
def test_gcp_kwargs_never_survive_client_logic(clean_redis_environment, overrides):
    redis_kwargs = _get_redis_client_logic(
        host="redis-host",
        port=6379,
        credential_provider=_StubCredentialProvider() if "gcp_service_account" in overrides else None,
        **overrides,
    )

    assert "gcp_service_account" not in redis_kwargs
    assert "gcp_ssl_ca_certs" not in redis_kwargs


def test_aws_iam_environment_settings_install_provider(clean_redis_environment, monkeypatch):
    monkeypatch.setenv("REDIS_AWS_IAM_AUTH", "true")
    monkeypatch.setenv("REDIS_AWS_IAM_USER_NAME", "iam-user")
    monkeypatch.setenv("REDIS_AWS_IAM_CACHE_NAME", "cache.example.com")
    monkeypatch.setenv("REDIS_AWS_IAM_REGION", "us-east-1")
    monkeypatch.setenv("REDIS_AWS_IAM_SERVERLESS", "1")
    monkeypatch.setenv("REDIS_SSL", "1")

    redis_kwargs = _get_redis_client_logic(host="cache.example.com", port=6379)

    provider = redis_kwargs["credential_provider"]
    assert isinstance(provider, ElastiCacheIAMCredentialProvider)
    assert provider._is_serverless is True
    assert not _AWS_IAM_SETTINGS & redis_kwargs.keys()


@pytest.mark.parametrize(
    "transport",
    [
        pytest.param({"host": "cache.example.com", "port": 6379}, id="host_without_ssl"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": False}, id="host_ssl_false"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "false"}, id="host_ssl_false_string"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "0"}, id="host_ssl_zero_string"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "no"}, id="host_ssl_no_string"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "off"}, id="host_ssl_off_string"),
        pytest.param({"url": "redis://cache.example.com:6379", "ssl": True}, id="plaintext_url"),
        pytest.param(
            {"startup_nodes": [{"host": "cache.example.com", "port": 6379}]},
            id="cluster_without_ssl",
        ),
        pytest.param(
            {
                "url": "rediss://cache.example.com:6379",
                "startup_nodes": [{"host": "cache.example.com", "port": 6379}],
            },
            id="cluster_without_ssl_ignores_url_scheme",
        ),
        pytest.param(
            {
                "sentinel_nodes": [("sentinel.example.com", 26379)],
                "service_name": "cache",
            },
            id="sentinel_without_ssl",
        ),
    ],
)
def test_aws_iam_auth_rejects_non_tls_connections(clean_redis_environment, transport):
    with pytest.raises(ValueError, match="requires TLS"):
        _get_redis_client_logic(
            **transport,
            aws_iam_auth=True,
            aws_iam_user_name="iam-user",
            aws_iam_cache_name="cache.example.com",
            aws_iam_region="us-east-1",
        )


@pytest.mark.parametrize(
    "transport",
    [
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": True}, id="host"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "true"}, id="host_ssl_true_string"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "True"}, id="host_ssl_true_capitalized"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "1"}, id="host_ssl_one_string"),
        pytest.param({"host": "cache.example.com", "port": 6379, "ssl": "yes"}, id="host_ssl_yes_string"),
        pytest.param(
            {"startup_nodes": [{"host": "cache.example.com", "port": 6379}], "ssl": "1"},
            id="cluster_ssl_one_string",
        ),
        pytest.param({"url": "rediss://cache.example.com:6379"}, id="url"),
        pytest.param(
            {
                "startup_nodes": [{"host": "cache.example.com", "port": 6379}],
                "ssl": True,
            },
            id="cluster",
        ),
        pytest.param(
            {
                "sentinel_nodes": [("sentinel.example.com", 26379)],
                "service_name": "cache",
                "ssl": True,
            },
            id="sentinel",
        ),
    ],
)
def test_aws_iam_auth_accepts_tls_connections(clean_redis_environment, transport):
    redis_kwargs = _get_redis_client_logic(
        **transport,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="us-east-1",
    )

    assert isinstance(redis_kwargs["credential_provider"], ElastiCacheIAMCredentialProvider)


@pytest.mark.parametrize("ssl", ["true", "True", "TRUE", "1", "yes", "YES", "false", "0", "no", "off", "", "maybe"])
def test_tls_detection_agrees_with_the_ssl_kwarg_coercion(ssl):
    assert _uses_tls({"ssl": ssl}) is _coerce_redis_kwargs_types({"ssl": ssl})["ssl"]


def test_aws_iam_settings_are_removed_for_url_and_static_credentials(clean_redis_environment):
    redis_kwargs = _get_redis_client_logic(
        url="rediss://url-user:url-pass@cache.example.com:6380",
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="us-east-1",
        username="static-user",
        password="static-password",
    )

    assert isinstance(redis_kwargs["credential_provider"], ElastiCacheIAMCredentialProvider)
    assert redis_kwargs["url"] == "rediss://cache.example.com:6380"
    assert "username" not in redis_kwargs
    assert "password" not in redis_kwargs
    assert not _AWS_IAM_SETTINGS & redis_kwargs.keys()


@pytest.mark.parametrize("missing", ["aws_iam_user_name", "aws_iam_cache_name", "aws_iam_region"])
def test_aws_iam_missing_setting_fails_closed(clean_redis_environment, missing):
    settings = {
        "aws_iam_auth": True,
        "aws_iam_user_name": "iam-user",
        "aws_iam_cache_name": "cache.example.com",
        "aws_iam_region": "us-east-1",
        "ssl": True,
    }
    settings[missing] = None

    with pytest.raises(ValueError, match=missing):
        _get_redis_client_logic(host="cache.example.com", port=6379, **settings)


@pytest.mark.parametrize("region_var", ["AWS_REGION", "AWS_DEFAULT_REGION"])
def test_aws_iam_region_falls_back_to_environment(clean_redis_environment, monkeypatch, region_var):
    monkeypatch.setenv(region_var, "sa-east-1")

    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        ssl=True,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
    )

    provider = redis_kwargs["credential_provider"]
    assert isinstance(provider, ElastiCacheIAMCredentialProvider)
    assert provider._region == "sa-east-1"


def test_aws_iam_region_prefers_aws_region_over_default_region(clean_redis_environment, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "sa-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        ssl=True,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
    )

    assert redis_kwargs["credential_provider"]._region == "sa-east-1"


def test_aws_iam_region_prefers_explicit_over_environment(clean_redis_environment, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "sa-east-1")

    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        ssl=True,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="explicit-region",
    )

    assert redis_kwargs["credential_provider"]._region == "explicit-region"


def test_aws_iam_settings_map_to_distinct_provider_fields(clean_redis_environment):
    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        ssl=True,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user-value",
        aws_iam_cache_name="iam-cache-value",
        aws_iam_region="iam-region-value",
    )

    provider = redis_kwargs["credential_provider"]
    assert isinstance(provider, ElastiCacheIAMCredentialProvider)
    assert provider._user_name == "iam-user-value"
    assert provider._cache_name == "iam-cache-value"
    assert provider._region == "iam-region-value"


@pytest.mark.parametrize("aws_iam_auth", [None, False, "", "false", "0", "no", "off"])
def test_aws_iam_auth_disabled_does_not_install_provider(clean_redis_environment, aws_iam_auth):
    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        aws_iam_auth=aws_iam_auth,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="us-east-1",
    )

    assert "credential_provider" not in redis_kwargs
    assert not _AWS_IAM_SETTINGS & redis_kwargs.keys()


@pytest.mark.parametrize("aws_iam_auth", [True, "true", "True", "TRUE", "1", "yes"])
def test_aws_iam_auth_enabled_by_any_truthy_flag(clean_redis_environment, aws_iam_auth):
    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        ssl=True,
        aws_iam_auth=aws_iam_auth,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache-name",
        aws_iam_region="us-east-1",
    )

    assert isinstance(redis_kwargs["credential_provider"], ElastiCacheIAMCredentialProvider)


@pytest.mark.parametrize(
    "aws_iam_serverless, expected",
    [
        pytest.param(None, False, id="unset"),
        pytest.param(False, False, id="bool_false"),
        pytest.param("false", False, id="string_false"),
        pytest.param("0", False, id="string_zero"),
        pytest.param(True, True, id="bool_true"),
        pytest.param("true", True, id="string_true"),
        pytest.param("1", True, id="string_one"),
    ],
)
def test_aws_iam_serverless_flag_reaches_the_provider(clean_redis_environment, aws_iam_serverless, expected):
    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        ssl=True,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache-name",
        aws_iam_region="us-east-1",
        aws_iam_serverless=aws_iam_serverless,
    )

    provider = redis_kwargs["credential_provider"]
    assert isinstance(provider, ElastiCacheIAMCredentialProvider)
    assert provider._is_serverless is expected
    assert "aws_iam_serverless" not in redis_kwargs


def test_explicit_provider_wins_over_aws_iam(clean_redis_environment):
    provider = _StubCredentialProvider()

    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        credential_provider=provider,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="us-east-1",
    )

    assert redis_kwargs["credential_provider"] is provider
    assert "aws_iam_auth" not in redis_kwargs


def test_gcp_wins_over_aws_iam(clean_redis_environment):
    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        gcp_service_account="sa@example.com",
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="us-east-1",
    )

    assert "credential_provider" not in redis_kwargs
    assert redis_kwargs["redis_connect_func"]._gcp_service_account == "sa@example.com"


def test_azure_wins_over_aws_iam(clean_redis_environment):
    redis_kwargs = _get_redis_client_logic(
        host="cache.example.com",
        port=6379,
        azure_redis_ad_token="true",
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="us-east-1",
    )

    assert "credential_provider" not in redis_kwargs
    assert redis_kwargs["redis_connect_func"]._azure_redis_ad_token is True


def test_async_cluster_installs_aws_iam_provider(clean_redis_environment):
    startup_nodes = [{"host": "cluster-node", "port": 6379}]

    client = get_redis_async_client(
        startup_nodes=startup_nodes,
        ssl=True,
        aws_iam_auth=True,
        aws_iam_user_name="iam-user",
        aws_iam_cache_name="cache.example.com",
        aws_iam_region="us-east-1",
    )

    assert isinstance(client.connection_kwargs["credential_provider"], ElastiCacheIAMCredentialProvider)


def test_provider_keeps_the_rest_of_the_url_intact(clean_redis_environment):
    provider = _StubCredentialProvider()

    redis_kwargs = _get_redis_client_logic(
        url="rediss://url-user:url-pass@redis-host:6379/3?protocol=3",
        credential_provider=provider,
    )

    assert redis_kwargs["url"] == "rediss://redis-host:6379/3?protocol=3"


def test_provider_free_url_is_left_untouched(clean_redis_environment):
    url = "redis://url-user:url-pass@redis-host:6379/3"

    redis_kwargs = _get_redis_client_logic(url=url)

    assert redis_kwargs["url"] == url


def test_credential_provider_auth_kwargs_supersedes_credentials_an_explicit_provider_replaces():
    provider = _StubCredentialProvider()

    auth_kwargs = _credential_provider_auth_kwargs(
        {
            "host": "redis-host",
            "port": 6379,
            "credential_provider": provider,
            "redis_connect_func": _gcp_marker_callback(),
            "username": "url-user",
            "password": "url-pass",
        }
    )

    assert auth_kwargs["credential_provider"] is provider
    assert auth_kwargs["host"] == "redis-host"
    assert auth_kwargs["port"] == 6379
    assert "redis_connect_func" not in auth_kwargs
    assert "username" not in auth_kwargs
    assert "password" not in auth_kwargs


def test_credential_provider_auth_kwargs_leaves_provider_free_kwargs_alone():
    redis_kwargs = {"host": "redis-host", "port": 6379, "username": "url-user", "password": "url-pass"}

    assert _credential_provider_auth_kwargs(redis_kwargs) == redis_kwargs


@pytest.mark.asyncio
async def test_redis_cache_test_connection_uses_shared_factory(clean_redis_environment):
    provider = _StubCredentialProvider()

    with (
        patch("redis.Redis", autospec=True),
        patch("redis.asyncio.BlockingConnectionPool", autospec=True),
        patch("redis.asyncio.Redis", autospec=True) as mock_async_redis,
    ):
        mock_async_redis.return_value.ping = AsyncMock(return_value=True)
        mock_async_redis.return_value.aclose = AsyncMock()
        cache = RedisCache(host="redis-host", port=6379, credential_provider=provider, password="redis-secret")
        result = await cache.test_connection()

    client_kwargs = mock_async_redis.call_args.kwargs
    assert result["status"] == "success"
    assert client_kwargs["credential_provider"] is provider
    assert "password" not in client_kwargs


@pytest.mark.asyncio
async def test_redis_cluster_cache_test_connection_uses_shared_factory(clean_redis_environment):
    provider = _StubCredentialProvider()
    recorder = MagicMock()

    class _StubAsyncCluster:
        def __init__(self, **kwargs):
            recorder(**kwargs)

        async def ping(self):
            return True

        async def aclose(self):
            return None

    with (
        patch("redis.RedisCluster", autospec=True),
        patch("redis.asyncio.cluster.RedisCluster", _StubAsyncCluster),
    ):
        cache = RedisClusterCache(startup_nodes=[{"host": "redis-host", "port": 6379}], credential_provider=provider)
        result = await cache.test_connection()

    cluster_kwargs = recorder.call_args.kwargs
    assert result["status"] == "success"
    assert cluster_kwargs["credential_provider"] is provider


def test_redis_cache_key_does_not_inspect_provider(clear_llm_client_cache):
    provider = _HostileCredentialProvider("synthetic-secret")
    second_provider = _StubCredentialProvider("another-token")

    with (
        patch("redis.Redis", autospec=True),
        patch("redis.asyncio.BlockingConnectionPool", autospec=True),
    ):
        cache = RedisCache(host="redis-host", port=6379, credential_provider=provider)
        second_cache = RedisCache(host="redis-host", port=6379, credential_provider=second_provider)

    first_key = cache._get_async_client_cache_key()
    assert first_key == cache._get_async_client_cache_key()
    assert first_key != second_cache._get_async_client_cache_key()


def test_pretty_print_never_expands_credential_provider(capsys):
    secret = "aaaa-UNIQUE-SENTINEL-bbbb"

    with patch(  # test-quality-ok: enable the debug-only printer without changing process-wide logger state
        "litellm._redis.verbose_logger.isEnabledFor", return_value=True
    ):
        _pretty_print_redis_config(
            redis_kwargs={
                "host": "redis-host",
                "port": 6379,
                "credential_provider": _HostileCredentialProvider(secret),
            }
        )

    output = capsys.readouterr().out
    assert secret not in output
    assert "UNIQUE" not in output
    assert "_payload" not in output
    assert "credential_provider" in output


def test_redis_cache_key_does_not_serialize_connect_func():
    def connect(connection):
        return None

    cache = RedisCache.__new__(RedisCache)
    cache.redis_kwargs = {"host": "redis-host", "port": 6379, "redis_connect_func": connect}

    first_key = cache._get_async_client_cache_key()
    assert first_key == cache._get_async_client_cache_key()


def test_redis_cache_key_keys_opaque_kwargs_by_identity():

    class _Opaque:
        pass

    first = RedisCache.__new__(RedisCache)
    first.redis_kwargs = {"host": "redis-host", "retry": _Opaque()}
    second = RedisCache.__new__(RedisCache)
    second.redis_kwargs = {"host": "redis-host", "retry": _Opaque()}

    assert first._get_async_client_cache_key() == first._get_async_client_cache_key()
    assert first._get_async_client_cache_key() != second._get_async_client_cache_key()


def test_get_redis_url_from_environment_single_url(monkeypatch):
    """Test when REDIS_URL is directly provided"""
    # Set the environment variable
    monkeypatch.setenv("REDIS_URL", "redis://redis-server:6379/0")

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL matches the expected value
    assert redis_url == "redis://redis-server:6379/0"


def test_get_redis_url_from_environment_host_port(monkeypatch):
    """Test when REDIS_HOST and REDIS_PORT are provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    # Ensure authentication variables are not set
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL matches the expected value
    assert redis_url == "redis://redis-server:6379"


def test_get_redis_url_from_environment_with_ssl(monkeypatch):
    """Test when SSL is enabled"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_SSL", "true")
    # Ensure authentication variables are not set
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL uses rediss:// protocol
    assert redis_url == "rediss://redis-server:6379"


def test_get_redis_url_from_environment_with_username_password(monkeypatch):
    """Test when username and password are provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_USERNAME", "user")
    monkeypatch.setenv("REDIS_PASSWORD", "password")

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL includes username:password@
    assert redis_url == "redis://user:password@redis-server:6379"


def test_get_redis_url_from_environment_with_password_only(monkeypatch):
    """Test when only password is provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_PASSWORD", "password")
    # Ensure username is not set
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL includes :password@
    assert redis_url == "redis://password@redis-server:6379"


def test_get_redis_url_from_environment_with_all_options(monkeypatch):
    """Test when all options are provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_USERNAME", "user")
    monkeypatch.setenv("REDIS_PASSWORD", "password")
    monkeypatch.setenv("REDIS_SSL", "true")

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL includes all components
    assert redis_url == "rediss://user:password@redis-server:6379"


def test_get_redis_url_from_environment_missing_host_port(monkeypatch):
    """Test error when required variables are missing"""
    # Make sure these environment variables don't exist
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT") as excinfo:
        get_redis_url_from_environment()

    # Check the error message
    assert "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified" in str(excinfo.value)


def test_get_redis_url_from_environment_missing_port(monkeypatch):
    """Test error when only REDIS_HOST is provided but REDIS_PORT is missing"""
    # Make sure REDIS_URL doesn't exist and set only REDIS_HOST
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis-server")

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT") as excinfo:
        get_redis_url_from_environment()

    # Check the error message
    assert "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified" in str(excinfo.value)


def test_max_connections_in_cluster_kwargs():
    """Test that max_connections is included in Redis cluster kwargs"""
    kwargs = _get_redis_cluster_kwargs()
    assert "max_connections" in kwargs, "max_connections should be in available Redis cluster kwargs"


def test_socket_timeouts_in_cluster_kwargs():
    """Test that Redis cluster clients can receive socket timeout configuration"""
    kwargs = _get_redis_cluster_kwargs()
    assert "socket_timeout" in kwargs
    assert "socket_connect_timeout" in kwargs


def test_reconnect_kwargs_in_cluster_kwargs():
    """Health check and keepalive must survive the cluster kwarg allow-list so
    operators can tune Redis cluster reconnection behavior via config."""
    kwargs = _get_redis_cluster_kwargs()
    assert "health_check_interval" in kwargs
    assert "socket_keepalive" in kwargs


def test_retry_attempts_in_cluster_kwargs():
    """cluster_error_retry_attempts must survive the cluster kwarg allow-list so
    operators can bound worst-case retry latency on a Redis Cluster: it was being
    silently dropped because the allow-list was built from redis.RedisCluster's
    decorated __init__ without unwrapping it, so getfullargspec saw an empty
    (self, *args, **kwargs) wrapper signature."""
    kwargs = _get_redis_cluster_kwargs()
    assert "cluster_error_retry_attempts" in kwargs


def test_async_only_kwargs_in_cluster_kwargs_when_async_client_requested():
    """decode_responses is on the async cluster client's constructor and not the sync
    one, on every redis-py the matrix covers. Introspecting the sync class regardless
    of which client is actually built silently drops it for every async cluster caller."""
    sync_kwargs = _get_redis_cluster_kwargs()
    async_kwargs = _get_redis_cluster_kwargs(async_redis.RedisCluster)

    assert "decode_responses" not in sync_kwargs
    assert "decode_responses" in async_kwargs


@patch(  # test-quality-ok: redis-py >= 6 keeps no cluster_error_retry_attempts attribute on the built client, so the constructor call is the only place the value is observable
    "litellm.caching.redis_cluster_node_isolation.get_litellm_async_redis_cluster_class"
)
def test_async_cluster_forwards_retry_attempts(mock_get_cluster_class):
    """Regression: cluster_error_retry_attempts must reach the constructed async
    cluster client. Silently dropping it removes an operator's only lever for
    bounding a stuck node's worst-case retry latency, and the client falls back
    to redis-py's own default (3 retries) instead."""
    mock_cluster_cls = mock_get_cluster_class.return_value
    get_redis_async_client(
        startup_nodes=[{"host": "cluster-node", "port": 6379}],
        cluster_error_retry_attempts=2,
    )

    call_kwargs = mock_cluster_cls.call_args[1]
    assert call_kwargs["cluster_error_retry_attempts"] == 2


def test_async_cluster_passes_async_only_kwargs():
    """Regression: decode_responses is an async-cluster-only constructor arg. When
    the allow-list came from the sync class it was filtered out and values came
    back as bytes instead of str."""
    client = get_redis_async_client(
        startup_nodes=[{"host": "cluster-node", "port": 6379}],
        decode_responses=True,
    )

    assert client.connection_kwargs["decode_responses"] is True


@pytest.mark.parametrize("cluster_client", [redis.RedisCluster, async_redis.RedisCluster], ids=["sync", "async"])
def test_cluster_kwargs_exclude_variadic_parameters(cluster_client):
    """*args / **kwargs are signature placeholders, not connection settings, and
    must never land in the allow-list regardless of which cluster client is
    introspected."""
    variadic = {
        name
        for name, param in inspect.signature(cluster_client).parameters.items()
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    }

    leaked = variadic & set(_get_redis_cluster_kwargs(cluster_client))
    assert not leaked, f"variadic params leaked into the allow-list: {leaked}"


@patch("litellm.caching.redis_cluster_node_isolation.get_litellm_async_redis_cluster_class")
def test_async_cluster_sets_reconnect_defaults(mock_get_cluster_class):
    """
    The async RedisCluster client must be built with a periodic health check and
    TCP keepalive so a connection silently dropped by a cluster restart (e.g.
    ElastiCache Serverless maintenance) is revalidated and reconnected before
    reuse instead of stalling in re-initialization. Regression for LIT-4083.
    """
    mock_cluster_cls = mock_get_cluster_class.return_value
    get_redis_async_client(startup_nodes=[{"host": "cluster-node", "port": 6379}])

    mock_cluster_cls.assert_called_once()
    call_kwargs = mock_cluster_cls.call_args[1]
    assert call_kwargs["health_check_interval"] == REDIS_CLUSTER_HEALTH_CHECK_INTERVAL
    assert call_kwargs["health_check_interval"] > 0
    assert call_kwargs["socket_keepalive"] is True


@patch("litellm.caching.redis_cluster_node_isolation.get_litellm_async_redis_cluster_class")
def test_async_cluster_reconnect_defaults_are_overridable(mock_get_cluster_class):
    """An explicit health_check_interval / socket_keepalive from config must win
    over the built-in reconnect defaults."""
    mock_cluster_cls = mock_get_cluster_class.return_value
    get_redis_async_client(
        startup_nodes=[{"host": "cluster-node", "port": 6379}],
        health_check_interval=7,
        socket_keepalive=False,
    )

    call_kwargs = mock_cluster_cls.call_args[1]
    assert call_kwargs["health_check_interval"] == 7
    assert call_kwargs["socket_keepalive"] is False


def test_get_redis_async_client_with_connection_pool():
    """Test that connection_pool parameter is properly passed to Redis client"""
    # Create a mock connection pool
    mock_pool = MagicMock(spec=async_redis.BlockingConnectionPool)

    # Mock the Redis client creation
    with (
        patch("litellm._redis.async_redis.Redis") as mock_redis,
        patch("litellm._redis._get_redis_client_logic") as mock_logic,
    ):
        # Configure mock to return basic redis kwargs
        mock_logic.return_value = {"host": "localhost", "port": 6379, "db": 0}

        # Call get_redis_async_client with connection_pool
        get_redis_async_client(connection_pool=mock_pool)

        # Verify Redis was called with connection_pool in kwargs
        call_kwargs = mock_redis.call_args[1]
        assert "connection_pool" in call_kwargs, "connection_pool should be passed to Redis client"
        assert call_kwargs["connection_pool"] == mock_pool, "connection_pool should match the provided pool"


def test_get_redis_async_client_without_connection_pool():
    """Test that Redis client works without connection_pool parameter"""
    with (
        patch("litellm._redis.async_redis.Redis") as mock_redis,
        patch("litellm._redis._get_redis_client_logic") as mock_logic,
    ):
        # Configure mock to return basic redis kwargs
        mock_logic.return_value = {"host": "localhost", "port": 6379, "db": 0}

        # Call get_redis_async_client without connection_pool
        get_redis_async_client()

        # Verify Redis was called without connection_pool in kwargs
        call_kwargs = mock_redis.call_args[1]
        assert "connection_pool" not in call_kwargs, "connection_pool should not be in kwargs when not provided"


def test_gcp_iam_credential_provider_get_credentials():
    """GCPIAMCredentialProvider.get_credentials() returns a token tuple."""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"

    with patch(
        "litellm._redis_credential_provider.generate_gcp_iam_access_token",
        return_value="tok-1",
    ) as mock_gen:
        provider = GCPIAMCredentialProvider(service_account)
        creds = provider.get_credentials()

    assert creds == ("tok-1",)
    mock_gen.assert_called_once_with(service_account)


def test_gcp_iam_credential_provider_caches_token():
    """
    Repeated calls to get_credentials() reuse the cached token and only call
    _generate_gcp_iam_access_token once, avoiding redundant blocking I/O.
    """
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"

    with patch(
        "litellm._redis_credential_provider.generate_gcp_iam_access_token",
        return_value="tok-cached",
    ) as mock_gen:
        provider = GCPIAMCredentialProvider(service_account)
        results = [provider.get_credentials() for _ in range(5)]

    assert all(r == ("tok-cached",) for r in results)
    # Token must be fetched exactly once regardless of how many connections are established
    mock_gen.assert_called_once_with(service_account)


def test_gcp_iam_credential_provider_refreshes_on_expiry():
    """
    get_credentials() fetches a new token after the cached one expires,
    ensuring connections always authenticate with a valid token.
    """
    import time

    import litellm._redis_credential_provider as cred_module

    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"

    with patch(
        "litellm._redis_credential_provider.generate_gcp_iam_access_token",
        side_effect=["tok-1", "tok-2"],
    ) as mock_gen:
        provider = GCPIAMCredentialProvider(service_account)

        # First call — populates cache
        assert provider.get_credentials() == ("tok-1",)

        # Artificially expire the cached token
        cred_module._token_cache[service_account] = ("tok-1", time.monotonic() - 1)

        # Second call — cache miss, must refresh
        assert provider.get_credentials() == ("tok-2",)

    assert mock_gen.call_count == 2


def test_gcp_iam_credential_provider_cache_shared_across_instances():
    """
    Multiple GCPIAMCredentialProvider instances for the same service account
    share one cached token so concurrent Redis connections don't each trigger
    a blocking IAM round-trip.
    """
    service_account = "projects/-/serviceAccounts/shared@project.iam.gserviceaccount.com"

    with patch(
        "litellm._redis_credential_provider.generate_gcp_iam_access_token",
        return_value="tok-shared",
    ) as mock_gen:
        p1 = GCPIAMCredentialProvider(service_account)
        p2 = GCPIAMCredentialProvider(service_account)

        assert p1.get_credentials() == ("tok-shared",)
        assert p2.get_credentials() == ("tok-shared",)

    # Only one network call despite two provider instances
    mock_gen.assert_called_once()


def test_get_redis_async_client_gcp_cluster_uses_credential_provider():
    """
    When startup_nodes + gcp_service_account are provided, the async cluster client
    must be constructed with a GCPIAMCredentialProvider — not a static password.
    This ensures that the 1-hour IAM token expiry does not cause auth failures.
    """
    startup_nodes = [{"host": "redis-node-1", "port": 6379}]

    mock_connect_func = MagicMock()
    mock_connect_func._gcp_service_account = "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"

    redis_kwargs = {
        "startup_nodes": startup_nodes,
        "redis_connect_func": mock_connect_func,
    }

    with (
        patch(
            "litellm.caching.redis_cluster_node_isolation.get_litellm_async_redis_cluster_class"
        ) as mock_get_cluster_class,
        patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs),
    ):
        mock_cluster = mock_get_cluster_class.return_value
        get_redis_async_client()

    assert mock_cluster.called
    cluster_call_kwargs = mock_cluster.call_args[1]

    # Must use credential_provider, not a static password
    assert "credential_provider" in cluster_call_kwargs, (
        "async GCP cluster must use credential_provider for per-connection token refresh"
    )
    assert isinstance(cluster_call_kwargs["credential_provider"], GCPIAMCredentialProvider)
    assert "password" not in cluster_call_kwargs, "async GCP cluster must not use a static password (expires after 1h)"


@patch("litellm._redis.init_redis_cluster")
def test_sync_client_prefers_cluster_over_url(mock_init_cluster, monkeypatch):
    """
    Test get_redis_client returns RedisCluster when startup_nodes is present even if
    REDIS_URL is also set.
    """
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")
    mock_init_cluster.return_value = MagicMock(spec=redis.RedisCluster)

    startup_nodes = [{"host": "cluster-node.example.com", "port": 6379}]
    get_redis_client(startup_nodes=startup_nodes)

    mock_init_cluster.assert_called_once()
    call_kwargs = mock_init_cluster.call_args[0][0]
    assert "startup_nodes" in call_kwargs, "startup_nodes must be forwarded to init_redis_cluster"


@patch("litellm.caching.redis_cluster_node_isolation.get_litellm_async_redis_cluster_class")
def test_async_client_prefers_cluster_over_url(mock_get_cluster_class, monkeypatch):
    """
    Test (1) get_redis_async_client returns async RedisCluster when startup_nodes is present
    even if REDIS_URL is also set and (2) startup_nodes is forwarded to RedisCluster.
    """
    mock_cluster_cls = mock_get_cluster_class.return_value
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")

    startup_nodes = [{"host": "cluster-node.example.com", "port": 6379}]
    get_redis_async_client(startup_nodes=startup_nodes)

    mock_cluster_cls.assert_called_once()
    call_kwargs = mock_cluster_cls.call_args[1]
    assert "startup_nodes" in call_kwargs, "startup_nodes must be forwarded to async RedisCluster"
    assert len(call_kwargs["startup_nodes"]) == 1, "should forward exactly 1 cluster node"


@patch("litellm.caching.redis_cluster_node_isolation.get_litellm_async_redis_cluster_class")
def test_async_client_prefers_cluster_over_url_via_env_var(mock_get_cluster_class, monkeypatch):
    """
    Test get_redis_async_client returns async RedisCluster when REDIS_CLUSTER_NODES is set
    even if REDIS_URL is also set.
    """
    mock_cluster_cls = mock_get_cluster_class.return_value
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")
    monkeypatch.setenv(
        "REDIS_CLUSTER_NODES",
        json.dumps([{"host": "cluster-node.example.com", "port": 6379}]),
    )

    get_redis_async_client()

    mock_cluster_cls.assert_called_once()
    call_kwargs = mock_cluster_cls.call_args[1]
    assert "startup_nodes" in call_kwargs, "startup_nodes must be forwarded to async RedisCluster"


@patch("litellm._redis.init_redis_cluster")
def test_sync_client_prefers_cluster_over_url_via_env_var(mock_init_cluster, monkeypatch):
    """
    Test get_redis_client returns RedisCluster when REDIS_CLUSTER_NODES is set even if
    REDIS_URL is also set.
    """
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")
    monkeypatch.setenv(
        "REDIS_CLUSTER_NODES",
        json.dumps([{"host": "cluster-node.example.com", "port": 6379}]),
    )
    mock_init_cluster.return_value = MagicMock(spec=redis.RedisCluster)

    get_redis_client()

    mock_init_cluster.assert_called_once()
    call_kwargs = mock_init_cluster.call_args[0][0]
    assert "startup_nodes" in call_kwargs, "startup_nodes must be forwarded to init_redis_cluster"
    assert len(call_kwargs["startup_nodes"]) == 1


@patch("litellm._redis.redis.Sentinel")
def test_sync_sentinel_uses_sentinel_password_and_master_password(mock_sentinel_cls):
    """Sentinel auth must be passed to the sentinel, not the Redis master client."""
    mock_sentinel = MagicMock()
    mock_sentinel_cls.return_value = mock_sentinel

    get_redis_client(
        sentinel_nodes=[("sentinel-1", 26379)],
        sentinel_password="sentinel-secret",
        service_name="mymaster",
        password="redis-secret",
        username="redis-user",
        ssl=True,
        ssl_cert_reqs="required",
        ssl_check_hostname=True,
        ssl_ca_certs="/tmp/test-ca.pem",
        max_connections=17,
        socket_timeout=5,
    )

    mock_sentinel_cls.assert_called_once()
    sentinel_call_kwargs = mock_sentinel_cls.call_args[1]
    assert "password" not in sentinel_call_kwargs
    assert "username" not in sentinel_call_kwargs
    assert "ssl" not in sentinel_call_kwargs
    assert "ssl_cert_reqs" not in sentinel_call_kwargs
    assert "ssl_check_hostname" not in sentinel_call_kwargs
    assert "ssl_ca_certs" not in sentinel_call_kwargs
    assert "max_connections" not in sentinel_call_kwargs
    assert "socket_timeout" not in sentinel_call_kwargs
    assert sentinel_call_kwargs["sentinel_kwargs"] == {
        "password": "sentinel-secret",
        "username": "redis-user",
        "ssl": True,
        "ssl_cert_reqs": "required",
        "ssl_check_hostname": True,
        "ssl_ca_certs": "/tmp/test-ca.pem",
        "max_connections": 17,
        "socket_timeout": 5,
    }
    assert "service_name" not in sentinel_call_kwargs["sentinel_kwargs"]
    assert "sentinel_nodes" not in sentinel_call_kwargs["sentinel_kwargs"]
    assert "sentinel_password" not in sentinel_call_kwargs["sentinel_kwargs"]
    mock_sentinel.master_for.assert_called_once_with(
        "mymaster",
        password="redis-secret",
        username="redis-user",
        ssl=True,
        ssl_cert_reqs="required",
        ssl_check_hostname=True,
        ssl_ca_certs="/tmp/test-ca.pem",
        max_connections=17,
        socket_timeout=5,
    )


@patch("redis.Sentinel")
def test_sync_sentinel_keeps_provider_off_monitors_and_on_master(mock_sentinel_cls):
    provider = _StubCredentialProvider()
    mock_sentinel = MagicMock()
    mock_sentinel_cls.return_value = mock_sentinel

    get_redis_client(
        sentinel_nodes=[("sentinel-1", 26379)],
        sentinel_password="sentinel-secret",
        service_name="mymaster",
        password="redis-secret",
        credential_provider=provider,
    )

    sentinel_kwargs = mock_sentinel_cls.call_args.kwargs["sentinel_kwargs"]
    assert sentinel_kwargs["password"] == "sentinel-secret"
    assert "credential_provider" not in sentinel_kwargs
    assert mock_sentinel.master_for.call_args.kwargs["credential_provider"] is provider
    assert "password" not in mock_sentinel.master_for.call_args.kwargs


@patch("litellm._redis.async_redis.Sentinel")
def test_async_sentinel_uses_sentinel_password_and_master_password(
    mock_sentinel_cls,
):
    """Async sentinel auth must mirror the sync sentinel password routing."""
    mock_sentinel = MagicMock()
    mock_sentinel_cls.return_value = mock_sentinel

    get_redis_async_client(
        sentinel_nodes=[("sentinel-1", 26379)],
        sentinel_password="sentinel-secret",
        service_name="mymaster",
        password="redis-secret",
        username="redis-user",
        ssl=True,
        ssl_cert_reqs="required",
        ssl_check_hostname=True,
        ssl_ca_certs="/tmp/test-ca.pem",
        max_connections=17,
        socket_timeout=5,
    )

    mock_sentinel_cls.assert_called_once()
    sentinel_call_kwargs = mock_sentinel_cls.call_args[1]
    assert "password" not in sentinel_call_kwargs
    assert "username" not in sentinel_call_kwargs
    assert "ssl" not in sentinel_call_kwargs
    assert "ssl_cert_reqs" not in sentinel_call_kwargs
    assert "ssl_check_hostname" not in sentinel_call_kwargs
    assert "ssl_ca_certs" not in sentinel_call_kwargs
    assert "max_connections" not in sentinel_call_kwargs
    assert "socket_timeout" not in sentinel_call_kwargs
    assert sentinel_call_kwargs["sentinel_kwargs"] == {
        "password": "sentinel-secret",
        "username": "redis-user",
        "ssl": True,
        "ssl_cert_reqs": "required",
        "ssl_check_hostname": True,
        "ssl_ca_certs": "/tmp/test-ca.pem",
        "max_connections": 17,
        "socket_timeout": 5,
    }
    assert "service_name" not in sentinel_call_kwargs["sentinel_kwargs"]
    assert "sentinel_nodes" not in sentinel_call_kwargs["sentinel_kwargs"]
    assert "sentinel_password" not in sentinel_call_kwargs["sentinel_kwargs"]
    mock_sentinel.master_for.assert_called_once_with(
        "mymaster",
        password="redis-secret",
        username="redis-user",
        ssl=True,
        ssl_cert_reqs="required",
        ssl_check_hostname=True,
        ssl_ca_certs="/tmp/test-ca.pem",
        max_connections=17,
        socket_timeout=5,
    )


@patch("litellm._redis.init_redis_cluster")
def test_sync_client_preserves_password_for_cluster_when_url_also_set(mock_init_cluster, monkeypatch):
    """
    Test _get_redis_client_logic does not strip password from redis_kwargs when
    startup_nodes is present even if REDIS_URL is also set.
    """
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")
    monkeypatch.setenv("REDIS_PASSWORD", "secret")
    mock_init_cluster.return_value = MagicMock(spec=redis.RedisCluster)

    startup_nodes = [{"host": "cluster-node.example.com", "port": 6379}]
    get_redis_client(startup_nodes=startup_nodes)

    mock_init_cluster.assert_called_once()
    call_kwargs = mock_init_cluster.call_args[0][0]
    assert "password" in call_kwargs, "password must not be stripped when routing to cluster"
    assert call_kwargs["password"] == "secret"


def test_connection_pool_returns_none_for_cluster(monkeypatch):
    """Test get_redis_connection_pool returns None when startup_nodes is present."""
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")
    startup_nodes = [{"host": "cluster-node.example.com", "port": 6379}]
    result = get_redis_connection_pool(startup_nodes=startup_nodes)
    assert result is None, "connection pool must be None for cluster mode"


@patch("litellm._redis.redis.Redis.from_url")
def test_sync_client_url_used_when_no_cluster(mock_from_url, monkeypatch):
    """
    Test get_redis_client default to using URL path when no startup_nodes are provided.
    """
    monkeypatch.setenv("REDIS_URL", "redis://plain-host:6379")
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    get_redis_client()

    mock_from_url.assert_called_once()


@patch("litellm._redis.redis.Redis.from_url")
def test_explicit_host_outranks_environment_redis_url(mock_from_url, monkeypatch):
    """
    An explicitly configured host must win over REDIS_URL in the environment.

    Otherwise the url branch strips the caller's host/port and the client
    silently connects to whatever REDIS_URL names, so an explicit config block
    (or a connection test typed into the admin UI) targets the wrong server.
    """
    monkeypatch.setenv("REDIS_URL", "redis://env-host:6379")
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    client = get_redis_client(host="explicit-host", port=6380)

    mock_from_url.assert_not_called()
    assert client.connection_pool.connection_kwargs["host"] == "explicit-host"
    assert client.connection_pool.connection_kwargs["port"] == 6380


@patch("litellm._redis.redis.Redis.from_url")
def test_explicit_url_still_wins_over_environment_host(mock_from_url, monkeypatch):
    """An explicit url argument keeps taking the from_url path."""
    monkeypatch.setenv("REDIS_HOST", "env-host")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    get_redis_client(url="redis://explicit-host:6380")

    mock_from_url.assert_called_once()
    assert mock_from_url.call_args.kwargs["url"] == "redis://explicit-host:6380"


@patch("litellm._redis.redis.Redis.from_url")
def test_environment_redis_url_used_when_caller_names_no_target(mock_from_url, monkeypatch):
    """With no caller-supplied connection target, REDIS_URL still drives the client."""
    monkeypatch.setenv("REDIS_URL", "redis://env-host:6379")
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    get_redis_client()

    mock_from_url.assert_called_once()


def test_redis_client_resolves_environment_override_before_creation(
    monkeypatch: pytest.MonkeyPatch,
    clean_redis_environment: None,
    clear_llm_client_cache: None,
) -> None:
    monkeypatch.setenv("LITELLM_TEST_REDIS_PASSWORD", "resolved-redis-password")

    client: Final = get_redis_client(
        host="redis-host",
        port=6379,
        password="os.environ/LITELLM_TEST_REDIS_PASSWORD",
    )

    assert client.connection_pool.connection_kwargs["password"] == "resolved-redis-password"


@pytest.mark.parametrize("falsy_ssl", [False, None, 0, ""])
def test_connection_pool_falsy_ssl_uses_plain_connection(falsy_ssl, monkeypatch):
    """
    ssl=False must produce a plain (non-TLS) connection pool.

    The admin UI's coordination Redis form always sends ssl explicitly, so a
    presence check here turns ssl=False into an SSLConnection; the TLS
    handshake against a plaintext Redis then hangs until the ping timeout and
    every connection test from the UI fails.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    with patch("litellm._redis.async_redis.BlockingConnectionPool") as mock_pool:
        get_redis_connection_pool(host="plain-redis.example.com", port=6379, ssl=falsy_ssl)

    call_kwargs = mock_pool.call_args.kwargs
    assert call_kwargs.get("connection_class") is not async_redis.SSLConnection, (
        f"ssl={falsy_ssl!r} must not select SSLConnection"
    )
    assert "ssl" not in call_kwargs, "ssl must never leak into BlockingConnectionPool kwargs"


def test_connection_pool_ssl_true_uses_ssl_connection(monkeypatch):
    """ssl=True must still opt in to a TLS connection pool."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    with patch("litellm._redis.async_redis.BlockingConnectionPool") as mock_pool:
        get_redis_connection_pool(host="tls-redis.example.com", port=6380, ssl=True)

    call_kwargs = mock_pool.call_args.kwargs
    assert call_kwargs.get("connection_class") is async_redis.SSLConnection
    assert "ssl" not in call_kwargs, "ssl must be consumed, not forwarded to the pool"


def test_connection_pool_without_ssl_kwarg_uses_plain_connection(monkeypatch):
    """Omitting ssl entirely must keep the historical plain-connection default."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    with patch("litellm._redis.async_redis.BlockingConnectionPool") as mock_pool:
        get_redis_connection_pool(host="plain-redis.example.com", port=6379)

    call_kwargs = mock_pool.call_args.kwargs
    assert call_kwargs.get("connection_class") is not async_redis.SSLConnection
    assert "ssl" not in call_kwargs


def test_connection_pool_env_redis_ssl_false_uses_plain_connection(monkeypatch):
    """REDIS_SSL=false from the environment must not select SSLConnection."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)
    monkeypatch.setenv("REDIS_SSL", "false")

    pool = get_redis_connection_pool(host="plain-host", port=6379)

    assert pool is not None
    assert pool.connection_class is async_redis.Connection
    assert "ssl" not in pool.connection_kwargs


@pytest.mark.parametrize(
    "redis_config",
    [
        pytest.param({"host": "redis-host", "port": 6379}, id="host_port"),
        pytest.param({"url": "redis://redis-host:6379"}, id="url"),
    ],
)
def test_connection_pool_keeps_socket_timeout(redis_config, monkeypatch):
    """The async pool must carry socket_timeout however Redis was configured.

    The url branch used to rebuild pool kwargs from scratch as {timeout, url,
    max_connections}, dropping socket_timeout. redis-py then leaves both
    socket_timeout and socket_connect_timeout (which falls back to it) unset, so a
    Redis host that drops packets rather than refusing them blocks every caller
    indefinitely instead of failing fast.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    pool = get_redis_connection_pool(socket_timeout=5.0, **redis_config)

    assert pool is not None
    assert pool.connection_kwargs.get("socket_timeout") == 5.0


@pytest.mark.parametrize(
    "redis_config",
    [
        pytest.param({"host": "redis-host", "port": 6379}, id="host_port"),
        pytest.param({"url": "redis://redis-host:6379"}, id="url"),
    ],
)
def test_sync_client_keeps_socket_timeout(redis_config, monkeypatch):
    """The sync client is built during RedisCache.__init__ and blocks the caller.

    Without socket_timeout it stalls for the OS TCP timeout against an unreachable
    host, so merely constructing the cache stops the process.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    client = get_redis_client(socket_timeout=5.0, **redis_config)

    assert client.connection_pool.connection_kwargs.get("socket_timeout") == 5.0


@pytest.mark.parametrize(
    "redis_config",
    [
        pytest.param({"host": "redis-host", "port": 6379}, id="host_port"),
        pytest.param({"url": "redis://redis-host:6379"}, id="url"),
    ],
)
def test_async_client_keeps_socket_timeout(redis_config, monkeypatch):
    """Same invariant for the async client built without an injected pool."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    client = get_redis_async_client(socket_timeout=5.0, **redis_config)

    assert client.connection_pool.connection_kwargs.get("socket_timeout") == 5.0


def test_url_config_does_not_forward_ssl_kwarg(monkeypatch):
    """ssl stays consumed rather than forwarded on the url path.

    TLS is selected by the rediss:// scheme there; handing ssl=True to a redis://
    url yields a plain Connection that rejects the kwarg when it first connects.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    client = get_redis_client(url="redis://redis-host:6379", ssl=True)

    assert "ssl" not in client.connection_pool.connection_kwargs


@pytest.mark.parametrize(
    "client_only_kwarg",
    [
        pytest.param({"single_connection_client": True}, id="single_connection_client"),
        pytest.param({"auto_close_connection_pool": True}, id="auto_close_connection_pool"),
        pytest.param({"ssl_ca_certs": "/tmp/ca.pem"}, id="ssl_ca_certs"),
        pytest.param({"ssl": True}, id="ssl"),
    ],
)
def test_url_config_drops_kwargs_the_connection_cannot_accept(client_only_kwarg, monkeypatch):
    """Only kwargs the connection accepts may be forwarded on the url path.

    from_url hands its kwargs down to the connection class, so client-level settings and
    the SSLConnection-only ssl_* family raise TypeError the first time a connection is
    created. TLS on a url config comes from the rediss:// scheme instead.
    """
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_CLUSTER_NODES", raising=False)

    pool = get_redis_connection_pool(url="redis://redis-host:6379", socket_timeout=5.0, **client_only_kwarg)

    assert pool is not None
    pool.make_connection()
    assert pool.connection_kwargs.get("socket_timeout") == 5.0


def test_redis_uses_the_hiredis_response_parser():
    """The C parser must be the one redis-py actually picks.

    hiredis is declared in the `proxy` extra purely for speed; nothing imports it, so
    dropping it from pyproject.toml would silently fall back to the pure-Python parser
    with no other symptom. redis-py selects it at import time, so asserting on the
    selection is what catches that.
    """
    from redis._parsers import _HiredisParser
    from redis.connection import HIREDIS_AVAILABLE, DefaultParser

    assert HIREDIS_AVAILABLE, "hiredis is not installed; redis-py fell back to the pure-Python parser"
    assert DefaultParser is _HiredisParser, f"redis-py selected {DefaultParser.__name__}, expected _HiredisParser"

    client = get_redis_client(host="redis-host", port=6379)
    connection = client.connection_pool.make_connection()
    assert isinstance(connection._parser, _HiredisParser)


def test_init_arg_names_sees_through_decorated_inits():
    """redis-py >= 7.4 wraps AbstractConnection.__init__ with @deprecated_args, whose
    wrapper is declared (self, *args, **kwargs). Introspecting the wrapper directly
    yields no real parameters, which silently emptied the from_url allowlist and
    dropped socket_timeout from url-configured connections. The MRO walk must follow
    __wrapped__ to the true signature.
    """
    import functools

    from litellm._redis import _init_arg_names

    def deprecating(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            return fn(self, *args, **kwargs)

        return wrapper

    class Base:
        @deprecating
        def __init__(self, socket_timeout=None, socket_connect_timeout=None):
            pass

    class Concrete(Base):
        def __init__(self, host=None, **kwargs):
            super().__init__(**kwargs)

    names = _init_arg_names(Concrete)
    assert "socket_timeout" in names
    assert "socket_connect_timeout" in names
    assert "host" in names


def test_url_allowlist_always_carries_socket_timeouts():
    """The load-bearing invariant behind test_url_config_* against the INSTALLED
    redis-py, whatever its version: if a redis-py release changes how its __init__
    signatures are declared (7.4 did, via @deprecated_args), this is the first
    assertion that goes red.
    """
    from litellm._redis import _get_redis_url_kwargs

    allowed = _get_redis_url_kwargs()
    assert "socket_timeout" in allowed
    assert "socket_connect_timeout" in allowed


AZURE_AD_CONNECT_FUNC = {"_azure_credential": object()}
GCP_IAM_CONNECT_FUNC = {"_gcp_service_account": "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"}


@pytest.mark.parametrize(
    "markers, provider_cls",
    [
        (AZURE_AD_CONNECT_FUNC, AzureADCredentialProvider),
        (GCP_IAM_CONNECT_FUNC, GCPIAMCredentialProvider),
    ],
    ids=["azure_ad", "gcp_iam"],
)
def test_async_url_client_authenticates_through_credential_provider(markers, provider_cls):
    """A REDIS_URL config with Azure AD or GCP IAM must still reach the server with a credential.

    The url branch forwards redis_connect_func straight to the async connection, which runs
    its AUTH exchange with the blocking client API and dies, so the branch has to hand the
    connection a CredentialProvider instead.
    """
    redis_kwargs = {
        "url": "rediss://redis-host:6380",
        "redis_connect_func": SimpleNamespace(**markers),
    }

    with patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs):
        client = get_redis_async_client()

    connection_kwargs = client.connection_pool.connection_kwargs
    assert isinstance(connection_kwargs.get("credential_provider"), provider_cls)
    assert "redis_connect_func" not in connection_kwargs


@pytest.mark.parametrize(
    "markers, provider_cls",
    [
        (AZURE_AD_CONNECT_FUNC, AzureADCredentialProvider),
        (GCP_IAM_CONNECT_FUNC, GCPIAMCredentialProvider),
    ],
    ids=["azure_ad", "gcp_iam"],
)
def test_async_url_connection_pool_authenticates_through_credential_provider(markers, provider_cls):
    """Same for the pool-based path: every connection the pool hands out needs the provider."""
    redis_kwargs = {
        "url": "rediss://redis-host:6380",
        "redis_connect_func": SimpleNamespace(**markers),
    }

    with patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs):
        pool = get_redis_connection_pool()

    assert isinstance(pool.connection_kwargs.get("credential_provider"), provider_cls)
    assert "redis_connect_func" not in pool.connection_kwargs


def test_async_url_client_drops_username_alongside_credential_provider():
    """redis-py refuses a connection given both a username and a credential_provider, and
    AzureADCredentialProvider already carries REDIS_USERNAME, so the username must be dropped.
    """
    redis_kwargs = {
        "url": "rediss://redis-host:6380",
        "username": "redis-user",
        "redis_connect_func": SimpleNamespace(**AZURE_AD_CONNECT_FUNC),
    }

    with patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs):
        client = get_redis_async_client()

    pool = client.connection_pool
    assert "username" not in pool.connection_kwargs
    pool.connection_class(**pool.connection_kwargs)


@pytest.mark.parametrize("build_pool", [False, True], ids=["client", "pool"])
def test_async_url_keeps_a_coroutine_connect_func(build_pool):
    """redis-py awaits a coroutine redis_connect_func on an async connection, so one we cannot
    turn into a credential provider has to be left where it is rather than dropped.
    """

    async def connect(connection):
        return None

    redis_kwargs = {
        "url": "rediss://redis-host:6380",
        "redis_connect_func": connect,
    }

    with patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs):
        pool = get_redis_connection_pool() if build_pool else get_redis_async_client().connection_pool

    assert pool.connection_kwargs["redis_connect_func"] is connect
    assert "credential_provider" not in pool.connection_kwargs


def test_async_cluster_drops_a_connect_func_it_cannot_pass_on():
    """redis-py's async RedisCluster has no redis_connect_func parameter, so a connect func that
    is not translated into a credential provider has to be dropped rather than forwarded.
    """

    async def connect(connection):
        return None

    redis_kwargs = {
        "startup_nodes": [{"host": "cluster-node", "port": 6379}],
        "redis_connect_func": connect,
    }

    with patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs):
        client = get_redis_async_client()

    assert isinstance(client, async_redis.RedisCluster)


@pytest.mark.parametrize(
    "markers, provider_cls",
    [
        (AZURE_AD_CONNECT_FUNC, AzureADCredentialProvider),
        (GCP_IAM_CONNECT_FUNC, GCPIAMCredentialProvider),
    ],
    ids=["azure_ad", "gcp_iam"],
)
@pytest.mark.parametrize(
    "sentinel_password",
    [None, "sentinel-secret"],
    ids=["unauthenticated_monitors", "password_protected_monitors"],
)
def test_async_sentinel_keeps_the_credential_provider_off_the_monitors(markers, provider_cls, sentinel_password):
    """The Sentinel monitors are separate servers with their own password, so the data node's token
    never belongs on them: redis-py refuses it next to a Sentinel password, and sends it to an
    unauthenticated monitor as an AUTH the monitor rejects.
    """
    redis_kwargs = {
        "sentinel_nodes": [("sentinel-1", 26379)],
        "sentinel_password": sentinel_password,
        "service_name": "mymaster",
        "redis_connect_func": SimpleNamespace(**markers),
    }

    with patch("litellm._redis.async_redis.Sentinel") as mock_sentinel_cls:
        with patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs):
            get_redis_async_client()

    sentinel_kwargs = mock_sentinel_cls.call_args[1]["sentinel_kwargs"]
    assert sentinel_kwargs["password"] == sentinel_password
    assert "credential_provider" not in sentinel_kwargs

    monitor_connection = async_redis.Connection(host="sentinel-1", port=26379, **sentinel_kwargs)
    assert monitor_connection.credential_provider is None
    assert bool(monitor_connection.username or monitor_connection.password) is bool(sentinel_password)

    master_kwargs = mock_sentinel_cls.return_value.master_for.call_args[1]
    assert isinstance(master_kwargs["credential_provider"], provider_cls)
    assert "password" not in master_kwargs


def _tls_view(connection: AbstractConnection) -> dict[str, object] | None:
    if not isinstance(connection, async_redis.SSLConnection):
        return None
    context: Final = connection.ssl_context
    return {
        "ca_certs": context.ca_certs,
        "certfile": context.certfile,
        "keyfile": context.keyfile,
        "verify_certificate": context.cert_reqs != ssl.CERT_NONE,
        "check_hostname": context.check_hostname,
    }


def _auth_view(kwargs: dict[str, object]) -> dict[str, object]:
    provider: Final = kwargs.get("credential_provider")
    match provider:
        case None:
            return {"kind": "static", "username": kwargs.get("username"), "password": kwargs.get("password")}
        case GCPIAMCredentialProvider():
            return {"kind": "gcp_iam", "service_account": provider._gcp_service_account}
        case AzureADCredentialProvider():
            return {"kind": "azure_ad", "username": provider._username}
        case ElastiCacheIAMCredentialProvider():
            return {
                "kind": "aws_iam",
                "user_name": provider._user_name,
                "cache_name": provider._cache_name,
                "region": provider._region,
                "serverless": provider._is_serverless,
            }
        case _:
            return {"kind": "host"}


def _connection_view(connection: AbstractConnection, kwargs: dict[str, object]) -> dict[str, object]:
    return {
        "tls": _tls_view(connection),
        "auth": _auth_view(kwargs),
        "socket_timeout": connection.socket_timeout,
        "socket_connect_timeout": connection.socket_connect_timeout,
    }


def _resolution_view(kwargs: dict[str, object]) -> dict[str, object]:
    url: Final = _credential_provider_auth_kwargs(_get_redis_client_logic(**kwargs)).get("url")
    client: Final = get_redis_async_client(**kwargs)
    if isinstance(client, async_redis.RedisCluster):
        nodes: Final = list(client.nodes_manager.startup_nodes.values())
        return {
            "mode": "cluster",
            "nodes": [f"{node.host}:{node.port}" for node in nodes],
            **_connection_view(nodes[0].acquire_connection(), client.connection_kwargs),
        }
    pool: Final = client.connection_pool
    connection: Final = pool.make_connection()
    if isinstance(pool, async_redis.SentinelConnectionPool):
        sentinels: Final = [sentinel.connection_pool.connection_kwargs for sentinel in pool.sentinel_manager.sentinels]
        return {
            "mode": "sentinel",
            "nodes": [f"{sentinel['host']}:{sentinel['port']}" for sentinel in sentinels],
            "service_name": pool.service_name,
            "sentinel_password": sentinels[0].get("password"),
            "db": connection.db,
            **_connection_view(connection, pool.connection_kwargs),
        }
    match connection:
        case async_redis.UnixDomainSocketConnection():
            target = {"mode": "unix", "path": connection.path}
        case _ if url is not None:
            target = {"mode": "url", "url": url}
        case _:
            target = {"mode": "host", "host": connection.host, "port": connection.port, "db": connection.db}
    return {**target, **_connection_view(connection, pool.connection_kwargs)}


def _expected_view(fields: dict[str, object]) -> dict[str, object]:
    return {
        "auth": {"kind": "static", "username": None, "password": None},
        "tls": None,
        "socket_timeout": 5.0,
        "socket_connect_timeout": 5.0,
        **fields,
    }


@pytest.mark.skipif(
    redis.VERSION != (5, 3, 1),
    reason="the Rust resolver in litellm-rust/crates/config pins redis-py 5.3.1, per its AGENTS.md",
)
@pytest.mark.parametrize(
    ("kwargs", "environment", "expected"),
    [
        pytest.param(
            {"host": "redis"},
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="host_defaults",
        ),
        pytest.param(
            {"host": "redis", "port": "6380\n", "db": " 2 ", "socket_timeout": "1_0.5\n"},
            {},
            _expected_view(
                {
                    "db": 2,
                    "host": "redis",
                    "mode": "host",
                    "port": 6380,
                    "socket_timeout": 10.5,
                    "socket_connect_timeout": 10.5,
                }
            ),
            id="padded_and_underscored_numbers_parse_like_python_int_and_float",
        ),
        pytest.param(
            {},
            {"REDIS_HOST": "redis", "REDIS_PORT": "6_380", "REDIS_DB": "3\n"},
            _expected_view({"db": 3, "host": "redis", "mode": "host", "port": 6380}),
            id="padded_environment_numbers_parse_like_python_int",
        ),
        pytest.param(
            {"startup_nodes": []},
            {"REDIS_CLUSTER_NODES": '[{"host": "node-a", "port": 7000}]'},
            _expected_view({"mode": "cluster", "nodes": ["node-a:7000"]}),
            id="empty_cluster_nodes_use_the_environment",
        ),
        pytest.param(
            {"sentinel_nodes": [], "service_name": "primary"},
            {"REDIS_SENTINEL_NODES": '[["sentinel-a", 26379]]'},
            _expected_view(
                {
                    "db": 0,
                    "mode": "sentinel",
                    "nodes": ["sentinel-a:26379"],
                    "sentinel_password": None,
                    "service_name": "primary",
                }
            ),
            id="empty_sentinel_nodes_use_the_environment",
        ),
        pytest.param(
            {"sentinel_nodes": [["sentinel-a", 26379]], "service_name": ""},
            {"REDIS_SERVICE_NAME": "primary"},
            _expected_view(
                {
                    "db": 0,
                    "mode": "sentinel",
                    "nodes": ["sentinel-a:26379"],
                    "sentinel_password": None,
                    "service_name": "primary",
                }
            ),
            id="empty_service_name_uses_the_environment",
        ),
        pytest.param(
            {"host": "redis", "ssl_cert_reqs": "strict"},
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="plaintext_ignores_tls_requirements",
        ),
        pytest.param(
            {"host": "redis", "ssl": True, "ssl_ca_path": "/ca"},
            {},
            {
                "rejected": {
                    "message": "Redis.__init__() got an unexpected keyword argument 'ssl_ca_path'",
                    "type": "TypeError",
                }
            },
            id="ssl_ca_path_is_unsupported",
        ),
        pytest.param(
            {
                "db": 2,
                "host": "redis",
                "password": "pass",
                "port": 6380,
                "socket_timeout": 1.5,
                "ssl": True,
                "username": "user",
            },
            {},
            _expected_view(
                {
                    "auth": {"kind": "static", "password": "pass", "username": "user"},
                    "db": 2,
                    "host": "redis",
                    "mode": "host",
                    "port": 6380,
                    "socket_connect_timeout": 1.5,
                    "socket_timeout": 1.5,
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="host_with_every_basic_setting",
        ),
        pytest.param(
            {},
            {
                "REDIS_HOST": "environment",
                "REDIS_PASSWORD": "environment-secret",
                "REDIS_PORT": "6390",
                "REDIS_SSL": "True",
            },
            _expected_view(
                {
                    "auth": {"kind": "static", "password": "environment-secret", "username": None},
                    "db": 0,
                    "host": "environment",
                    "mode": "host",
                    "port": 6390,
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="environment_host_port_and_password",
        ),
        pytest.param(
            {"db": "3", "host": "redis", "port": "6380", "socket_timeout": "2.5"},
            {},
            _expected_view(
                {
                    "db": 3,
                    "host": "redis",
                    "mode": "host",
                    "port": 6380,
                    "socket_connect_timeout": 2.5,
                    "socket_timeout": 2.5,
                }
            ),
            id="string_numbers_are_coerced",
        ),
        pytest.param(
            {"host": "redis"},
            {"REDIS_SOCKET_TIMEOUT": "0.75"},
            _expected_view(
                {
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                    "socket_connect_timeout": 0.75,
                    "socket_timeout": 0.75,
                }
            ),
            id="socket_timeout_from_the_environment",
        ),
        pytest.param(
            {},
            {"REDIS_URL": "redis://environment:6379/1"},
            _expected_view({"mode": "url", "url": "redis://environment:6379/1"}),
            id="environment_url",
        ),
        pytest.param(
            {},
            {"REDIS_HOST": "ignored", "REDIS_URL": "redis://from-url:6379"},
            _expected_view({"mode": "url", "url": "redis://from-url:6379"}),
            id="environment_url_beats_the_environment_host",
        ),
        pytest.param(
            {"host": "configured"},
            {"REDIS_URL": "redis://environment:6379"},
            _expected_view({"db": 0, "host": "configured", "mode": "host", "port": 6379}),
            id="an_explicit_host_drops_the_environment_url",
        ),
        pytest.param(
            {"db": 4, "host": "dropped", "password": "dropped", "url": "rediss://cache:6380", "username": "user"},
            {},
            _expected_view(
                {
                    "auth": {"kind": "static", "password": None, "username": "user"},
                    "mode": "url",
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                    "url": "rediss://cache:6380",
                }
            ),
            id="a_url_drops_host_db_and_password_and_keeps_the_username",
        ),
        pytest.param(
            {
                "ssl_ca_certs": "/ca.pem",
                "ssl_cert_reqs": "none",
                "ssl_check_hostname": True,
                "url": "rediss://cache:6380",
            },
            {},
            _expected_view(
                {
                    "mode": "url",
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                    "url": "rediss://cache:6380",
                }
            ),
            id="a_url_ignores_tls_settings_outside_the_url",
        ),
        pytest.param(
            {
                "password": "cluster-secret",
                "startup_nodes": [{"host": "node-a", "port": 7000}],
                "url": "redis://single:6379",
            },
            {},
            _expected_view(
                {
                    "auth": {"kind": "static", "password": "cluster-secret", "username": None},
                    "mode": "cluster",
                    "nodes": ["node-a:7000"],
                }
            ),
            id="startup_nodes_beat_a_url",
        ),
        pytest.param(
            {},
            {"REDIS_CLUSTER_NODES": '[{"host": "node-a", "port": "7000"}, {"host": "node-b", "port": 7001}]'},
            _expected_view({"mode": "cluster", "nodes": ["node-a:7000", "node-b:7001"]}),
            id="environment_cluster_nodes",
        ),
        pytest.param(
            {
                "ssl": "true",
                "ssl_ca_certs": "/ca.pem",
                "ssl_check_hostname": True,
                "startup_nodes": [{"host": "node-a", "port": 7000}],
            },
            {},
            _expected_view(
                {
                    "mode": "cluster",
                    "nodes": ["node-a:7000"],
                    "tls": {
                        "ca_certs": "/ca.pem",
                        "certfile": None,
                        "check_hostname": True,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="cluster_tls_options",
        ),
        pytest.param(
            {
                "password": "data-secret",
                "sentinel_nodes": [["sentinel-a", 26379], ["sentinel-b", 26380]],
                "sentinel_password": "sentinel-secret",
                "service_name": "primary",
            },
            {},
            _expected_view(
                {
                    "auth": {"kind": "static", "password": "data-secret", "username": None},
                    "db": 0,
                    "mode": "sentinel",
                    "nodes": ["sentinel-a:26379", "sentinel-b:26380"],
                    "sentinel_password": "sentinel-secret",
                    "service_name": "primary",
                }
            ),
            id="sentinel_from_settings",
        ),
        pytest.param(
            {},
            {
                "REDIS_SENTINEL_NODES": '[["sentinel-a", 26379]]',
                "REDIS_SENTINEL_PASSWORD": "sentinel-secret",
                "REDIS_SERVICE_NAME": "primary",
            },
            _expected_view(
                {
                    "db": 0,
                    "mode": "sentinel",
                    "nodes": ["sentinel-a:26379"],
                    "sentinel_password": "sentinel-secret",
                    "service_name": "primary",
                }
            ),
            id="sentinel_from_the_environment",
        ),
        pytest.param(
            {"host": "ignored", "sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"},
            {},
            {
                "rejected": {
                    "message": "redis.asyncio.client.Redis() got multiple values for keyword argument 'host'",
                    "type": "TypeError",
                }
            },
            id="a_host_breaks_sentinel",
        ),
        pytest.param(
            {"sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary", "url": "redis://single:6379"},
            {},
            _expected_view({"mode": "url", "url": "redis://single:6379"}),
            id="a_url_beats_sentinels",
        ),
        pytest.param(
            {
                "aws_iam_auth": True,
                "credential_provider": True,
                "gcp_service_account": "account",
                "host": "redis",
                "password": "dropped",
            },
            {"REDIS_AZURE_AD_TOKEN": "true"},
            _expected_view({"auth": {"kind": "host"}, "db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="an_explicit_credential_provider_beats_cloud_auth",
        ),
        pytest.param(
            {"aws_iam_auth": True, "azure_redis_ad_token": "True", "host": "redis", "password": "dropped"},
            {"REDIS_GCP_SERVICE_ACCOUNT": "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"},
            _expected_view(
                {
                    "auth": {
                        "kind": "gcp_iam",
                        "service_account": "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com",
                    },
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                }
            ),
            id="gcp_beats_azure_and_aws",
        ),
        pytest.param(
            {
                "gcp_service_account": "account",
                "gcp_ssl_ca_certs": "/gcp-ca.pem",
                "host": "redis",
                "ssl": True,
                "ssl_ca_certs": "/configured-ca.pem",
            },
            {},
            _expected_view(
                {
                    "auth": {"kind": "gcp_iam", "service_account": "account"},
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                    "tls": {
                        "ca_certs": "/gcp-ca.pem",
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="gcp_ca_certs_replace_the_configured_ones_under_ssl",
        ),
        pytest.param(
            {
                "aws_iam_auth": True,
                "azure_client_id": "client",
                "azure_client_secret": "secret",
                "azure_redis_ad_token": True,
                "azure_tenant_id": "tenant",
                "host": "redis",
                "username": "dropped",
            },
            {"REDIS_USERNAME": "environment-user"},
            _expected_view(
                {
                    "auth": {"kind": "azure_ad", "username": "environment-user"},
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                }
            ),
            id="azure_beats_aws_and_takes_the_environment_username",
        ),
        pytest.param(
            {
                "azure_client_id": "client",
                "azure_client_secret": "secret",
                "azure_redis_ad_token": True,
                "azure_tenant_id": "tenant",
                "host": "redis",
            },
            {},
            _expected_view(
                {"auth": {"kind": "azure_ad", "username": None}, "db": 0, "host": "redis", "mode": "host", "port": 6379}
            ),
            id="azure_without_a_username_sends_only_the_token",
        ),
        pytest.param(
            {
                "azure_client_id": "client",
                "azure_client_secret": "secret",
                "azure_redis_ad_token": "1",
                "azure_tenant_id": "tenant",
                "host": "redis",
            },
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="azure_needs_the_literal_true",
        ),
        pytest.param(
            {
                "aws_iam_auth": "yes",
                "aws_iam_cache_name": "cache-1",
                "aws_iam_user_name": "iam-user",
                "host": "cache-1",
                "password": "dropped",
                "ssl": True,
            },
            {"AWS_DEFAULT_REGION": "us-east-1"},
            _expected_view(
                {
                    "auth": {
                        "cache_name": "cache-1",
                        "kind": "aws_iam",
                        "region": "us-east-1",
                        "serverless": False,
                        "user_name": "iam-user",
                    },
                    "db": 0,
                    "host": "cache-1",
                    "mode": "host",
                    "port": 6379,
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="aws_iam_over_tls_with_the_default_region",
        ),
        pytest.param(
            {
                "aws_iam_auth": True,
                "aws_iam_cache_name": "cache-1",
                "aws_iam_user_name": "iam-user",
                "host": "cache-1",
                "ssl": True,
            },
            {"AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "eu-central-1"},
            _expected_view(
                {
                    "auth": {
                        "cache_name": "cache-1",
                        "kind": "aws_iam",
                        "region": "eu-central-1",
                        "serverless": False,
                        "user_name": "iam-user",
                    },
                    "db": 0,
                    "host": "cache-1",
                    "mode": "host",
                    "port": 6379,
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="aws_region_beats_the_default_region",
        ),
        pytest.param(
            {
                "aws_iam_auth": True,
                "aws_iam_cache_name": "cache-1",
                "aws_iam_region": "eu-west-1",
                "aws_iam_serverless": True,
                "aws_iam_user_name": "iam-user",
                "url": "rediss://user:pass@cache-1:6379",
            },
            {},
            _expected_view(
                {
                    "auth": {
                        "cache_name": "cache-1",
                        "kind": "aws_iam",
                        "region": "eu-west-1",
                        "serverless": True,
                        "user_name": "iam-user",
                    },
                    "mode": "url",
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                    "url": "rediss://cache-1:6379",
                }
            ),
            id="aws_iam_serverless_over_a_rediss_url_drops_its_userinfo",
        ),
        pytest.param(
            {
                "aws_iam_auth": True,
                "aws_iam_cache_name": "cache-1",
                "aws_iam_region": "us-east-1",
                "aws_iam_user_name": "iam-user",
                "host": "cache-1",
            },
            {},
            {"rejected": {"message": "AWS ElastiCache IAM Redis authentication requires TLS", "type": "ValueError"}},
            id="aws_iam_without_tls_fails",
        ),
        pytest.param(
            {
                "aws_iam_auth": True,
                "aws_iam_cache_name": "cache-1",
                "aws_iam_user_name": "iam-user",
                "host": "cache-1",
                "ssl": True,
            },
            {},
            {
                "rejected": {
                    "message": "AWS ElastiCache IAM Redis authentication requires: aws_iam_region",
                    "type": "ValueError",
                }
            },
            id="aws_iam_without_a_region_fails",
        ),
        pytest.param(
            {
                "host": "redis",
                "ssl": True,
                "ssl_ca_certs": "/ca.pem",
                "ssl_cert_reqs": "none",
                "ssl_check_hostname": True,
            },
            {},
            _expected_view(
                {
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                    "tls": {
                        "ca_certs": "/ca.pem",
                        "certfile": None,
                        "check_hostname": True,
                        "keyfile": None,
                        "verify_certificate": False,
                    },
                }
            ),
            id="tls_options_from_settings",
        ),
        pytest.param(
            {"host": "redis"},
            {"REDIS_SSL": "true", "REDIS_SSL_CA_CERTS": "/env-ca.pem", "REDIS_SSL_CERT_REQS": "optional"},
            _expected_view(
                {
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                    "tls": {
                        "ca_certs": "/env-ca.pem",
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="tls_options_from_the_environment",
        ),
        pytest.param(
            {"host": "redis", "socket_connect_timeout": 0.25, "socket_timeout": 2.5},
            {},
            _expected_view(
                {
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                    "socket_connect_timeout": 0.25,
                    "socket_timeout": 2.5,
                }
            ),
            id="connect_timeout_is_separate_from_the_socket_timeout",
        ),
        pytest.param(
            {"host": "os.environ/COORD_HOST", "password": "os.environ/COORD_PASSWORD"},
            {"COORD_HOST": "referenced", "COORD_PASSWORD": "referenced-secret"},
            _expected_view(
                {
                    "auth": {"kind": "static", "password": "referenced-secret", "username": None},
                    "db": 0,
                    "host": "referenced",
                    "mode": "host",
                    "port": 6379,
                }
            ),
            id="environment_references_resolve",
        ),
        pytest.param(
            {"port": 6379},
            {},
            {"rejected": {"message": "Either 'host' or 'url' must be specified for redis.", "type": "ValueError"}},
            id="no_target_fails",
        ),
        pytest.param(
            {"host": "redis", "ssl": True, "ssl_certfile": "/client.pem", "ssl_keyfile": "/client.key"},
            {},
            _expected_view(
                {
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                    "tls": {
                        "ca_certs": None,
                        "certfile": "/client.pem",
                        "check_hostname": False,
                        "keyfile": "/client.key",
                        "verify_certificate": True,
                    },
                }
            ),
            id="client_certificates",
        ),
        pytest.param(
            {"url": "rediss://cache:6380/0?ssl_cert_reqs=none&ssl_ca_certs=/ca.pem&ssl_check_hostname=yes"},
            {},
            _expected_view(
                {
                    "mode": "url",
                    "tls": {
                        "ca_certs": "/ca.pem",
                        "certfile": None,
                        "check_hostname": True,
                        "keyfile": None,
                        "verify_certificate": False,
                    },
                    "url": "rediss://cache:6380/0?ssl_cert_reqs=none&ssl_ca_certs=/ca.pem&ssl_check_hostname=yes",
                }
            ),
            id="a_rediss_url_takes_tls_options_from_its_query",
        ),
        pytest.param(
            {"url": "redis://cache:6379/0?socket_timeout=2"},
            {},
            _expected_view(
                {
                    "mode": "url",
                    "socket_connect_timeout": 2.0,
                    "socket_timeout": 2.0,
                    "url": "redis://cache:6379/0?socket_timeout=2",
                }
            ),
            id="a_url_query_setting_without_a_redis_rs_equivalent",
        ),
        pytest.param(
            {"db": "99999999999999999999", "ssl_cert_reqs": "strict", "url": "rediss://cache:6380"},
            {},
            _expected_view(
                {
                    "mode": "url",
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                    "url": "rediss://cache:6380",
                }
            ),
            id="tls_settings_outside_a_url_never_fail_it",
        ),
        pytest.param(
            {"db": 99999999999, "startup_nodes": [{"host": "node-a", "port": 7000}]},
            {},
            {
                "rejected": {
                    "message": "Argument 'db' must be 0 or None in cluster mode",
                    "type": "RedisClusterException",
                }
            },
            id="db_is_not_read_for_a_cluster",
        ),
        pytest.param(
            {"db": True, "startup_nodes": [{"host": "node-a", "port": 7000}]},
            {},
            {
                "rejected": {
                    "message": "Argument 'db' must be 0 or None in cluster mode",
                    "type": "RedisClusterException",
                }
            },
            id="cluster_db_true",
        ),
        pytest.param(
            {"db": False, "startup_nodes": [{"host": "node-a", "port": 7000}]},
            {},
            _expected_view({"mode": "cluster", "nodes": ["node-a:7000"]}),
            id="cluster_db_false",
        ),
        pytest.param(
            {"db": 0.5, "startup_nodes": [{"host": "node-a", "port": 7000}]},
            {},
            {
                "rejected": {
                    "message": "Argument 'db' must be 0 or None in cluster mode",
                    "type": "RedisClusterException",
                }
            },
            id="cluster_db_fraction",
        ),
        pytest.param(
            {"db": 0.0, "startup_nodes": [{"host": "node-a", "port": 7000}]},
            {},
            _expected_view({"mode": "cluster", "nodes": ["node-a:7000"]}),
            id="cluster_db_float_zero",
        ),
        pytest.param(
            {"db": [1], "startup_nodes": [{"host": "node-a", "port": 7000}]},
            {},
            {
                "rejected": {
                    "message": "Argument 'db' must be 0 or None in cluster mode",
                    "type": "RedisClusterException",
                }
            },
            id="cluster_db_nonempty_list",
        ),
        pytest.param(
            {"db": [], "startup_nodes": [{"host": "node-a", "port": 7000}]},
            {},
            _expected_view({"mode": "cluster", "nodes": ["node-a:7000"]}),
            id="cluster_db_empty_list",
        ),
        pytest.param(
            {"host": "redis", "ssl": True, "ssl_ciphers": "HIGH"},
            {},
            _expected_view(
                {
                    "db": 0,
                    "host": "redis",
                    "mode": "host",
                    "port": 6379,
                    "tls": {
                        "ca_certs": None,
                        "certfile": None,
                        "check_hostname": False,
                        "keyfile": None,
                        "verify_certificate": True,
                    },
                }
            ),
            id="ssl_ciphers",
        ),
        pytest.param(
            {"host": "redis", "retry_on_timeout": True},
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="retry_on_timeout",
        ),
        pytest.param(
            {"host": "redis", "port": "abc"},
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="malformed_port",
        ),
        pytest.param(
            {"db": "first", "host": "redis"},
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="malformed_db",
        ),
        pytest.param(
            {"host": "redis", "socket_timeout": "slow"},
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="socket_timeout_not_a_number",
        ),
        pytest.param(
            {},
            {"REDIS_HOST": "redis", "REDIS_PORT": "abc"},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="malformed_port_from_the_environment",
        ),
        pytest.param(
            {"host": "redis", "protocol": 3},
            {},
            _expected_view({"db": 0, "host": "redis", "mode": "host", "port": 6379}),
            id="protocol_3",
        ),
        pytest.param(
            {"host": "redis"},
            {"REDIS_RETRY_ON_ERROR": "true"},
            {"rejected": {"message": "'bool' object is not iterable", "type": "TypeError"}},
            id="retry_on_error_from_the_environment",
        ),
        pytest.param(
            {"host": "redis", "unix_socket_path": "/tmp/redis.sock"},
            {},
            _expected_view({"mode": "unix", "path": "/tmp/redis.sock"}),
            id="unix_socket_path",
        ),
    ],
)
def test_redis_settings_resolve_like_the_rust_resolver_expects(monkeypatch, kwargs, environment, expected):
    for name in [name for name in os.environ if name.startswith(("REDIS_", "AZURE_", "AWS_"))]:
        monkeypatch.delenv(name)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    try:
        outcome = _resolution_view(kwargs)
    except (ValueError, TypeError, RedisClusterException) as error:
        outcome = {"rejected": {"type": type(error).__name__, "message": str(error)}}

    assert outcome == expected
