import json
from unittest.mock import MagicMock, patch

import pytest
import redis
import redis.asyncio as async_redis

from litellm._redis import (
    _get_credential_provider_from_connect_func,
    _get_redis_cluster_kwargs,
    create_azure_ad_redis_connect_func,
    get_redis_async_client,
    get_redis_client,
    get_redis_connection_pool,
    get_redis_url_from_environment,
)
from litellm.constants import REDIS_CLUSTER_HEALTH_CHECK_INTERVAL
from litellm._redis_credential_provider import (
    AzureADCredentialProvider,
    GCPIAMCredentialProvider,
    _token_cache,
)


@pytest.fixture(autouse=True)
def clear_gcp_iam_token_cache():
    """Reset the module-level GCP IAM token cache between tests."""
    _token_cache.clear()
    yield
    _token_cache.clear()


def _mock_azure_identity(mock_credential):
    mock_azure_identity = MagicMock()
    mock_azure_identity.DefaultAzureCredential = MagicMock(return_value=mock_credential)
    mock_azure_identity.ClientSecretCredential = MagicMock(return_value=mock_credential)
    mock_azure_identity.ManagedIdentityCredential = MagicMock(
        return_value=mock_credential
    )
    return mock_azure_identity


def test_azure_redis_workload_identity_uses_default_credential(monkeypatch):
    from litellm._redis import _build_azure_credential

    default_credential = object()
    mock_azure_identity = MagicMock()
    mock_azure_identity.DefaultAzureCredential = MagicMock(
        return_value=default_credential
    )
    mock_azure_identity.ClientSecretCredential = MagicMock()
    mock_azure_identity.ManagedIdentityCredential = MagicMock()
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-id")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-id")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", "/var/run/token")
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)

    with patch.dict(
        "sys.modules", {"azure.identity": mock_azure_identity, "azure": MagicMock()}
    ):
        credential = _build_azure_credential()

    assert credential is default_credential
    mock_azure_identity.DefaultAzureCredential.assert_called_once_with()
    mock_azure_identity.ManagedIdentityCredential.assert_not_called()


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
    with pytest.raises(ValueError) as excinfo:
        get_redis_url_from_environment()

    # Check the error message
    assert (
        "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified"
        in str(excinfo.value)
    )


def test_get_redis_url_from_environment_missing_port(monkeypatch):
    """Test error when only REDIS_HOST is provided but REDIS_PORT is missing"""
    # Make sure REDIS_URL doesn't exist and set only REDIS_HOST
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis-server")

    # Call the function and expect a ValueError
    with pytest.raises(ValueError) as excinfo:
        get_redis_url_from_environment()

    # Check the error message
    assert (
        "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified"
        in str(excinfo.value)
    )


def test_max_connections_in_cluster_kwargs():
    """Test that max_connections is included in Redis cluster kwargs"""
    kwargs = _get_redis_cluster_kwargs()
    assert (
        "max_connections" in kwargs
    ), "max_connections should be in available Redis cluster kwargs"


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


@patch("litellm._redis.async_redis.RedisCluster")
def test_async_cluster_sets_reconnect_defaults(mock_cluster_cls):
    """
    The async RedisCluster client must be built with a periodic health check and
    TCP keepalive so a connection silently dropped by a cluster restart (e.g.
    ElastiCache Serverless maintenance) is revalidated and reconnected before
    reuse instead of stalling in re-initialization. Regression for LIT-4083.
    """
    get_redis_async_client(startup_nodes=[{"host": "cluster-node", "port": 6379}])

    mock_cluster_cls.assert_called_once()
    call_kwargs = mock_cluster_cls.call_args[1]
    assert call_kwargs["health_check_interval"] == REDIS_CLUSTER_HEALTH_CHECK_INTERVAL
    assert call_kwargs["health_check_interval"] > 0
    assert call_kwargs["socket_keepalive"] is True


@patch("litellm._redis.async_redis.RedisCluster")
def test_async_cluster_reconnect_defaults_are_overridable(mock_cluster_cls):
    """An explicit health_check_interval / socket_keepalive from config must win
    over the built-in reconnect defaults."""
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
        assert (
            "connection_pool" in call_kwargs
        ), "connection_pool should be passed to Redis client"
        assert (
            call_kwargs["connection_pool"] == mock_pool
        ), "connection_pool should match the provided pool"


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
        assert (
            "connection_pool" not in call_kwargs
        ), "connection_pool should not be in kwargs when not provided"


def test_gcp_iam_credential_provider_get_credentials():
    """GCPIAMCredentialProvider.get_credentials() returns a token tuple."""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"

    with patch(
        "litellm._redis_credential_provider._generate_gcp_iam_access_token",
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
        "litellm._redis_credential_provider._generate_gcp_iam_access_token",
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
        "litellm._redis_credential_provider._generate_gcp_iam_access_token",
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
    service_account = (
        "projects/-/serviceAccounts/shared@project.iam.gserviceaccount.com"
    )

    with patch(
        "litellm._redis_credential_provider._generate_gcp_iam_access_token",
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
    mock_connect_func._gcp_service_account = (
        "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"
    )

    redis_kwargs = {
        "startup_nodes": startup_nodes,
        "redis_connect_func": mock_connect_func,
    }

    with (
        patch("litellm._redis.async_redis.RedisCluster") as mock_cluster,
        patch("litellm._redis._get_redis_client_logic", return_value=redis_kwargs),
    ):
        get_redis_async_client()

    assert mock_cluster.called
    cluster_call_kwargs = mock_cluster.call_args[1]

    # Must use credential_provider, not a static password
    assert (
        "credential_provider" in cluster_call_kwargs
    ), "async GCP cluster must use credential_provider for per-connection token refresh"
    assert isinstance(
        cluster_call_kwargs["credential_provider"], GCPIAMCredentialProvider
    )
    assert (
        "password" not in cluster_call_kwargs
    ), "async GCP cluster must not use a static password (expires after 1h)"


def test_redis_connect_func_rejects_mixed_credential_markers():
    mock_connect_func = MagicMock()
    mock_connect_func._gcp_service_account = "service-account"
    mock_connect_func._azure_credential = MagicMock()

    with pytest.raises(ValueError, match="both GCP and Azure"):
        _get_credential_provider_from_connect_func(mock_connect_func, {})


def test_redis_connect_func_without_markers_has_no_credential_provider():
    assert _get_credential_provider_from_connect_func(lambda _: None, {}) is None


def test_azure_ad_connect_func_sends_username_auth_argument():
    mock_credential = MagicMock()
    mock_credential.get_token.return_value.token = "access-token"
    mock_azure_identity = _mock_azure_identity(mock_credential)

    with patch.dict(
        "sys.modules", {"azure.identity": mock_azure_identity, "azure": MagicMock()}
    ):
        connect_func = create_azure_ad_redis_connect_func(username="redis-user")

    connection = MagicMock()
    connection.read_response.return_value = b"OK"

    connect_func(connection)

    connection.send_command.assert_called_once_with(
        "AUTH", "redis-user", "access-token", check_health=False
    )


def test_sync_url_client_azure_ad_auth_uses_credential_provider(monkeypatch):
    """URL-based Azure Redis clients must keep token auth instead of filtering
    it out before Redis.from_url."""
    mock_credential = MagicMock()
    mock_azure_identity = _mock_azure_identity(mock_credential)
    monkeypatch.setenv("REDIS_USERNAME", "managed-identity-object-id")

    with (
        patch.dict(
            "sys.modules", {"azure.identity": mock_azure_identity, "azure": MagicMock()}
        ),
        patch("litellm._redis.redis.Redis.from_url") as mock_from_url,
    ):
        get_redis_client(
            url="rediss://cache.redis.cache.windows.net:6380",
            azure_redis_ad_token="true",
            ssl=True,
        )

    call_kwargs = mock_from_url.call_args.kwargs
    assert call_kwargs["url"] == "rediss://cache.redis.cache.windows.net:6380"
    assert isinstance(call_kwargs["credential_provider"], AzureADCredentialProvider)
    assert "redis_connect_func" not in call_kwargs
    assert call_kwargs["credential_provider"].get_credentials() == (
        "managed-identity-object-id",
        mock_credential.get_token.return_value.token,
    )


def test_async_url_client_azure_ad_auth_uses_credential_provider(monkeypatch):
    mock_credential = MagicMock()
    mock_azure_identity = _mock_azure_identity(mock_credential)
    monkeypatch.setenv("REDIS_USERNAME", "managed-identity-object-id")

    with (
        patch.dict(
            "sys.modules", {"azure.identity": mock_azure_identity, "azure": MagicMock()}
        ),
        patch("litellm._redis.async_redis.Redis.from_url") as mock_from_url,
    ):
        get_redis_async_client(
            url="rediss://cache.redis.cache.windows.net:6380",
            azure_redis_ad_token="true",
            ssl=True,
        )

    call_kwargs = mock_from_url.call_args.kwargs
    assert isinstance(call_kwargs["credential_provider"], AzureADCredentialProvider)
    assert call_kwargs["credential_provider"].get_credentials() == (
        "managed-identity-object-id",
        mock_credential.get_token.return_value.token,
    )


def test_async_client_azure_ad_auth_uses_credential_provider(monkeypatch):
    mock_credential = MagicMock()
    mock_azure_identity = _mock_azure_identity(mock_credential)
    monkeypatch.setenv("REDIS_USERNAME", "managed-identity-object-id")

    with (
        patch.dict(
            "sys.modules", {"azure.identity": mock_azure_identity, "azure": MagicMock()}
        ),
        patch("litellm._redis.async_redis.Redis") as mock_redis,
    ):
        get_redis_async_client(
            host="cache.redis.cache.windows.net",
            port=6380,
            azure_redis_ad_token="true",
            ssl=True,
        )

    call_kwargs = mock_redis.call_args.kwargs
    assert isinstance(call_kwargs["credential_provider"], AzureADCredentialProvider)
    assert call_kwargs["credential_provider"].get_credentials() == (
        "managed-identity-object-id",
        mock_credential.get_token.return_value.token,
    )


def test_connection_pool_url_azure_ad_auth_uses_credential_provider(monkeypatch):
    """Connection pools built from Redis URLs need Azure AD credential_provider
    too, otherwise pooled connections authenticate with no token."""
    mock_credential = MagicMock()
    mock_azure_identity = _mock_azure_identity(mock_credential)
    monkeypatch.setenv("REDIS_USERNAME", "managed-identity-object-id")

    with (
        patch.dict(
            "sys.modules", {"azure.identity": mock_azure_identity, "azure": MagicMock()}
        ),
        patch(
            "litellm._redis.async_redis.BlockingConnectionPool.from_url"
        ) as mock_from_url,
    ):
        get_redis_connection_pool(
            url="rediss://cache.redis.cache.windows.net:6380",
            azure_redis_ad_token="true",
            ssl=True,
        )

    call_kwargs = mock_from_url.call_args.kwargs
    assert call_kwargs["url"] == "rediss://cache.redis.cache.windows.net:6380"
    assert isinstance(call_kwargs["credential_provider"], AzureADCredentialProvider)
    assert call_kwargs["credential_provider"].get_credentials() == (
        "managed-identity-object-id",
        mock_credential.get_token.return_value.token,
    )


def test_connection_pool_azure_ad_auth_uses_credential_provider(monkeypatch):
    mock_credential = MagicMock()
    mock_azure_identity = _mock_azure_identity(mock_credential)
    monkeypatch.setenv("REDIS_USERNAME", "managed-identity-object-id")

    with (
        patch.dict(
            "sys.modules", {"azure.identity": mock_azure_identity, "azure": MagicMock()}
        ),
        patch("litellm._redis.async_redis.BlockingConnectionPool") as mock_pool,
    ):
        get_redis_connection_pool(
            host="cache.redis.cache.windows.net",
            port=6380,
            azure_redis_ad_token="true",
            ssl=True,
        )

    call_kwargs = mock_pool.call_args.kwargs
    assert isinstance(call_kwargs["credential_provider"], AzureADCredentialProvider)
    assert call_kwargs["credential_provider"].get_credentials() == (
        "managed-identity-object-id",
        mock_credential.get_token.return_value.token,
    )


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
    assert (
        "startup_nodes" in call_kwargs
    ), "startup_nodes must be forwarded to init_redis_cluster"


@patch("litellm._redis.async_redis.RedisCluster")
def test_async_client_prefers_cluster_over_url(mock_cluster_cls, monkeypatch):
    """
    Test (1) get_redis_async_client returns async RedisCluster when startup_nodes is present
    even if REDIS_URL is also set and (2) startup_nodes is forwarded to RedisCluster.
    """
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")

    startup_nodes = [{"host": "cluster-node.example.com", "port": 6379}]
    get_redis_async_client(startup_nodes=startup_nodes)

    mock_cluster_cls.assert_called_once()
    call_kwargs = mock_cluster_cls.call_args[1]
    assert (
        "startup_nodes" in call_kwargs
    ), "startup_nodes must be forwarded to async RedisCluster"
    assert (
        len(call_kwargs["startup_nodes"]) == 1
    ), "should forward exactly 1 cluster node"


@patch("litellm._redis.async_redis.RedisCluster")
def test_async_client_prefers_cluster_over_url_via_env_var(
    mock_cluster_cls, monkeypatch
):
    """
    Test get_redis_async_client returns async RedisCluster when REDIS_CLUSTER_NODES is set
    even if REDIS_URL is also set.
    """
    monkeypatch.setenv("REDIS_URL", "redis://fallback-host:6379")
    monkeypatch.setenv(
        "REDIS_CLUSTER_NODES",
        json.dumps([{"host": "cluster-node.example.com", "port": 6379}]),
    )

    get_redis_async_client()

    mock_cluster_cls.assert_called_once()
    call_kwargs = mock_cluster_cls.call_args[1]
    assert (
        "startup_nodes" in call_kwargs
    ), "startup_nodes must be forwarded to async RedisCluster"


@patch("litellm._redis.init_redis_cluster")
def test_sync_client_prefers_cluster_over_url_via_env_var(
    mock_init_cluster, monkeypatch
):
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
    assert (
        "startup_nodes" in call_kwargs
    ), "startup_nodes must be forwarded to init_redis_cluster"
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
def test_sync_client_preserves_password_for_cluster_when_url_also_set(
    mock_init_cluster, monkeypatch
):
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
    assert (
        "password" in call_kwargs
    ), "password must not be stripped when routing to cluster"
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
