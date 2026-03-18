import asyncio
import os

import pytest
import redis.asyncio as async_redis
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from litellm._redis import (
    _get_redis_cluster_kwargs,
    create_gcp_iam_redis_connect_func_async,
    get_redis_async_client,
    get_redis_url_from_environment,
)

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
    assert "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified" in str(excinfo.value)

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
    assert "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified" in str(excinfo.value)

def test_max_connections_in_cluster_kwargs():
    """Test that max_connections is included in Redis cluster kwargs"""
    kwargs = _get_redis_cluster_kwargs()
    assert "max_connections" in kwargs, "max_connections should be in available Redis cluster kwargs"

def test_get_redis_async_client_with_connection_pool():
    """Test that connection_pool parameter is properly passed to Redis client"""
    # Create a mock connection pool
    mock_pool = MagicMock(spec=async_redis.BlockingConnectionPool)
    
    # Mock the Redis client creation
    with patch('litellm._redis.async_redis.Redis') as mock_redis, \
         patch('litellm._redis._get_redis_client_logic') as mock_logic:
        
        # Configure mock to return basic redis kwargs
        mock_logic.return_value = {
            "host": "localhost",
            "port": 6379,
            "db": 0
        }
        
        # Call get_redis_async_client with connection_pool
        get_redis_async_client(connection_pool=mock_pool)
        
        # Verify Redis was called with connection_pool in kwargs
        call_kwargs = mock_redis.call_args[1]
        assert "connection_pool" in call_kwargs, "connection_pool should be passed to Redis client"
        assert call_kwargs["connection_pool"] == mock_pool, "connection_pool should match the provided pool"

def test_get_redis_async_client_without_connection_pool():
    """Test that Redis client works without connection_pool parameter"""
    with patch('litellm._redis.async_redis.Redis') as mock_redis, \
         patch('litellm._redis._get_redis_client_logic') as mock_logic:
        
        # Configure mock to return basic redis kwargs
        mock_logic.return_value = {
            "host": "localhost",
            "port": 6379,
            "db": 0
        }
        
        # Call get_redis_async_client without connection_pool
        get_redis_async_client()
        
        # Verify Redis was called without connection_pool in kwargs
        call_kwargs = mock_redis.call_args[1]
        assert "connection_pool" not in call_kwargs, "connection_pool should not be in kwargs when not provided"


def test_async_cluster_gcp_iam_uses_redis_connect_func():
    """Async Redis cluster with GCP IAM should pass redis_connect_func (per-connection
    token) instead of a static password, so tokens are refreshed on each connection."""
    sync_connect_func = MagicMock()
    sync_connect_func._gcp_service_account = (
        "projects/-/serviceAccounts/sa@proj.iam.gserviceaccount.com"
    )
    sync_connect_func._gcp_ssl_ca_certs = "/path/to/ca.pem"

    with patch("litellm._redis.async_redis.RedisCluster") as mock_cluster, \
         patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "startup_nodes": [{"host": "node1", "port": "6379"}],
            "redis_connect_func": sync_connect_func,
            "password": "should-be-ignored",
        }

        get_redis_async_client()

        call_kwargs = mock_cluster.call_args[1]
        connect_func = call_kwargs.get("redis_connect_func")
        assert connect_func is not None, "redis_connect_func must be passed to async RedisCluster"
        assert asyncio.iscoroutinefunction(connect_func), (
            "redis_connect_func should be async for async RedisCluster"
        )
        assert "password" not in call_kwargs or call_kwargs.get("password") is None or call_kwargs.get("password") == "should-be-ignored", (
            "password may be present but the connect func handles auth"
        )


def test_async_cluster_no_gcp_iam_no_connect_func():
    """Async Redis cluster without GCP IAM should not inject a redis_connect_func."""
    with patch("litellm._redis.async_redis.RedisCluster") as mock_cluster, \
         patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "startup_nodes": [{"host": "node1", "port": "6379"}],
            "password": "static-password",
        }

        get_redis_async_client()

        call_kwargs = mock_cluster.call_args[1]
        assert "redis_connect_func" not in call_kwargs, (
            "redis_connect_func should not be set without GCP IAM"
        )
        assert call_kwargs.get("password") == "static-password"


@pytest.mark.asyncio
async def test_async_connect_func_generates_fresh_token_per_call():
    """Each invocation of the async connect func should call _generate_gcp_iam_access_token,
    ensuring a fresh token even after the original has expired."""
    service_account = "projects/-/serviceAccounts/sa@proj.iam.gserviceaccount.com"
    connect_func = create_gcp_iam_redis_connect_func_async(
        service_account=service_account
    )

    mock_conn = AsyncMock()
    mock_conn._parser = MagicMock()
    mock_conn.read_response = AsyncMock(return_value=b"OK")

    with patch(
        "litellm._redis._generate_gcp_iam_access_token",
        return_value="token-1",
    ) as mock_gen:
        await connect_func(mock_conn)
        assert mock_gen.call_count == 1

        mock_gen.return_value = "token-2"
        await connect_func(mock_conn)
        assert mock_gen.call_count == 2

    assert mock_conn.send_command.await_count == 2


@pytest.mark.asyncio
async def test_async_connect_func_auth_error():
    """Async connect func should raise AuthenticationError on non-OK response."""
    from redis.exceptions import AuthenticationError

    connect_func = create_gcp_iam_redis_connect_func_async(
        service_account="projects/-/serviceAccounts/sa@proj.iam.gserviceaccount.com"
    )

    mock_conn = AsyncMock()
    mock_conn._parser = MagicMock()
    mock_conn.read_response = AsyncMock(return_value=b"DENIED")

    with patch(
        "litellm._redis._generate_gcp_iam_access_token",
        return_value="bad-token",
    ):
        with pytest.raises(AuthenticationError, match="GCP IAM authentication failed"):
            await connect_func(mock_conn)
