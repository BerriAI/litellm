import json
import os
from unittest.mock import MagicMock, patch

import pytest
import redis
import redis.asyncio as async_redis

from litellm._redis import (
    _get_redis_cluster_kwargs,
    get_redis_async_client,
    get_redis_client,
    get_redis_connection_pool,
    get_redis_url_from_environment,
)
from litellm._redis_credential_provider import (
    GCPIAMCredentialProvider,
    _token_cache,
)


@pytest.fixture(autouse=True)
def clear_gcp_iam_token_cache():
    """Reset the module-level GCP IAM token cache between tests."""
    _token_cache.clear()
    yield
    _token_cache.clear()


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
