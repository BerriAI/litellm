from litellm._redis import (
    get_redis_url_from_environment,
    _get_redis_kwargs,
    _get_redis_url_kwargs,
    _get_redis_cluster_kwargs,
    get_redis_async_client,
)
import os
import pytest
from unittest.mock import MagicMock, patch
import redis.asyncio as async_redis
import inspect
from functools import wraps

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

# Tests for signature introspection functions
def test_get_redis_kwargs_returns_list():
    """Test that _get_redis_kwargs returns a list of parameter names"""
    kwargs = _get_redis_kwargs()
    assert isinstance(kwargs, list), "_get_redis_kwargs should return a list"
    assert len(kwargs) > 0, "_get_redis_kwargs should return non-empty list"

def test_get_redis_kwargs_excludes_args():
    """Test that _get_redis_kwargs excludes specific arguments"""
    kwargs = _get_redis_kwargs()
    # These should be excluded
    assert "self" not in kwargs, "'self' should be excluded from Redis kwargs"
    assert "connection_pool" not in kwargs, "'connection_pool' should be excluded from Redis kwargs"
    assert "retry" not in kwargs, "'retry' should be excluded from Redis kwargs"

def test_get_redis_kwargs_includes_custom_args():
    """Test that _get_redis_kwargs includes custom arguments"""
    kwargs = _get_redis_kwargs()
    # These should be explicitly included
    assert "url" in kwargs, "'url' should be in Redis kwargs"
    assert "redis_connect_func" in kwargs, "'redis_connect_func' should be in Redis kwargs"
    assert "gcp_service_account" in kwargs, "'gcp_service_account' should be in Redis kwargs"
    assert "gcp_ssl_ca_certs" in kwargs, "'gcp_ssl_ca_certs' should be in Redis kwargs"

def test_get_redis_url_kwargs_returns_list():
    """Test that _get_redis_url_kwargs returns a list of parameter names"""
    kwargs = _get_redis_url_kwargs()
    assert isinstance(kwargs, list), "_get_redis_url_kwargs should return a list"
    assert len(kwargs) > 0, "_get_redis_url_kwargs should return non-empty list"

def test_get_redis_url_kwargs_excludes_args():
    """Test that _get_redis_url_kwargs excludes specific arguments"""
    kwargs = _get_redis_url_kwargs()
    # These should be excluded
    assert "self" not in kwargs, "'self' should be excluded from Redis URL kwargs"
    assert "connection_pool" not in kwargs, "'connection_pool' should be excluded from Redis URL kwargs"
    assert "retry" not in kwargs, "'retry' should be excluded from Redis URL kwargs"

def test_get_redis_url_kwargs_includes_url():
    """Test that _get_redis_url_kwargs includes 'url' argument"""
    kwargs = _get_redis_url_kwargs()
    assert "url" in kwargs, "'url' should be in Redis URL kwargs"

def test_get_redis_cluster_kwargs_returns_list():
    """Test that _get_redis_cluster_kwargs returns a list of parameter names"""
    kwargs = _get_redis_cluster_kwargs()
    assert isinstance(kwargs, list), "_get_redis_cluster_kwargs should return a list"
    assert len(kwargs) > 0, "_get_redis_cluster_kwargs should return non-empty list"

def test_get_redis_cluster_kwargs_excludes_args():
    """Test that _get_redis_cluster_kwargs excludes specific arguments"""
    kwargs = _get_redis_cluster_kwargs()
    # These should be excluded
    assert "self" not in kwargs, "'self' should be excluded from Redis cluster kwargs"
    assert "connection_pool" not in kwargs, "'connection_pool' should be excluded from Redis cluster kwargs"
    assert "retry" not in kwargs, "'retry' should be excluded from Redis cluster kwargs"
    assert "host" not in kwargs, "'host' should be excluded from Redis cluster kwargs"
    assert "port" not in kwargs, "'port' should be excluded from Redis cluster kwargs"
    assert "startup_nodes" not in kwargs, "'startup_nodes' should be excluded from Redis cluster kwargs"

def test_get_redis_cluster_kwargs_includes_custom_args():
    """Test that _get_redis_cluster_kwargs includes all custom arguments"""
    kwargs = _get_redis_cluster_kwargs()
    # These should be explicitly included
    required_args = [
        "password",
        "username",
        "ssl",
        "ssl_cert_reqs",
        "ssl_check_hostname",
        "ssl_ca_certs",
        "redis_connect_func",
        "gcp_service_account",
        "gcp_ssl_ca_certs",
        "max_connections"
    ]
    for arg in required_args:
        assert arg in kwargs, f"'{arg}' should be in Redis cluster kwargs"

def test_max_connections_in_cluster_kwargs():
    """Test that max_connections is included in Redis cluster kwargs"""
    kwargs = _get_redis_cluster_kwargs()
    assert "max_connections" in kwargs, "max_connections should be in available Redis cluster kwargs"

def test_signature_vs_getfullargspec_with_redis_decorator():
    """
    Regression test for Redis 7.1+ compatibility.

    Redis-py 7.1 introduced a @deprecated_args decorator on Redis.__init__ that wraps
    the method. Even though it uses functools.wraps, inspect.getfullargspec() cannot
    properly introspect through the wrapper and returns an empty args list, while
    inspect.signature() correctly unwraps and returns all parameters.

    This test replicates the exact decorator pattern to document why we migrated
    from getfullargspec() to signature().
    """
    from functools import wraps
    from typing import Callable, TypeVar

    C = TypeVar("C", bound=Callable)

    # Replicate redis-py's @deprecated_args decorator pattern
    def deprecated_args(args_to_warn=None, reason="", version=""):
        def decorator(func: C) -> C:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    # Create a mock Redis class with the decorator (like redis-py 7.1+)
    class MockRedisWithDecorator:
        @deprecated_args(args_to_warn=["retry_on_timeout"], version="6.0.0")
        def __init__(self, host='localhost', port=6379, db=0, password=None):
            self.host = host
            self.port = port

    # Test getfullargspec - it fails to see through the wrapper
    spec = inspect.getfullargspec(MockRedisWithDecorator.__init__)
    getfullargspec_args = spec.args

    # Test signature - it properly unwraps and sees the real parameters
    sig = inspect.signature(MockRedisWithDecorator.__init__)
    signature_params = list(sig.parameters.keys())

    # Document the issue: getfullargspec sees only the wrapper's (*args, **kwargs)
    # which results in 0 named args, while signature sees the original 5 parameters
    assert len(getfullargspec_args) == 0, "getfullargspec incorrectly returns 0 args for decorated function"
    assert len(signature_params) == 5, "signature correctly returns 5 parameters"
    assert 'self' in signature_params, "signature finds 'self' parameter"
    assert 'host' in signature_params, "signature finds 'host' parameter"
    assert 'port' in signature_params, "signature finds 'port' parameter"
    assert 'host' not in getfullargspec_args, "getfullargspec fails to find 'host' parameter"

    # This is why we use signature() instead of getfullargspec() in:
    # - _get_redis_kwargs()
    # - _get_redis_url_kwargs()
    # - _get_redis_cluster_kwargs()

def test_get_redis_kwargs_works_with_actual_redis_class():
    """
    Integration test: Verify _get_redis_kwargs() works with real redis.Redis class.

    This catches regressions when redis-py library updates add decorators or
    change class structure that might break our introspection.
    """
    import redis

    # Get kwargs from the actual Redis class
    kwargs = _get_redis_kwargs()

    # Critical: Should NOT be empty
    assert len(kwargs) > 0, (
        "CRITICAL: _get_redis_kwargs() returned empty list! "
        "This likely means redis-py updated their decorators/class structure. "
        f"Redis version: {redis.__version__}"
    )

    # Should have at least these common Redis parameters
    expected_params = ['host', 'port', 'db', 'password']
    for param in expected_params:
        assert param in kwargs, (
            f"Expected parameter '{param}' missing from Redis kwargs. "
            f"Redis version: {redis.__version__}. "
            "This may indicate redis-py structural changes."
        )

    # Excluded params should NOT be present
    excluded_params = ['self', 'connection_pool', 'retry']
    for param in excluded_params:
        assert param not in kwargs, (
            f"Parameter '{param}' should be excluded but was found in kwargs"
        )

def test_get_redis_url_kwargs_works_with_actual_redis_class():
    """
    Integration test: Verify _get_redis_url_kwargs() works with real redis.Redis.from_url.

    This catches regressions when redis-py library updates might break our introspection.
    """
    import redis

    # Get kwargs from the actual Redis.from_url method
    kwargs = _get_redis_url_kwargs()

    # Critical: Should NOT be empty
    assert len(kwargs) > 0, (
        "CRITICAL: _get_redis_url_kwargs() returned empty list! "
        "This likely means redis-py updated their decorators/class structure. "
        f"Redis version: {redis.__version__}"
    )

    # Should definitely include 'url' parameter
    assert 'url' in kwargs, (
        "'url' parameter missing from Redis.from_url kwargs. "
        f"Redis version: {redis.__version__}"
    )

    # Excluded params should NOT be present
    excluded_params = ['self', 'connection_pool', 'retry']
    for param in excluded_params:
        assert param not in kwargs, (
            f"Parameter '{param}' should be excluded but was found in kwargs"
        )

def test_get_redis_cluster_kwargs_works_with_actual_redis_class():
    """
    Integration test: Verify _get_redis_cluster_kwargs() works with real redis.RedisCluster.

    This catches regressions when redis-py library updates might break our introspection.
    """
    import redis

    # Get kwargs from the actual RedisCluster class
    kwargs = _get_redis_cluster_kwargs()

    # Critical: Should NOT be empty
    assert len(kwargs) > 0, (
        "CRITICAL: _get_redis_cluster_kwargs() returned empty list! "
        "This likely means redis-py updated their decorators/class structure. "
        f"Redis version: {redis.__version__}"
    )

    # Should have these cluster-specific parameters we explicitly add
    expected_custom_params = [
        'max_connections', 'password', 'username', 'ssl',
        'ssl_cert_reqs', 'ssl_check_hostname', 'ssl_ca_certs'
    ]
    for param in expected_custom_params:
        assert param in kwargs, (
            f"Expected parameter '{param}' missing from RedisCluster kwargs. "
            f"Redis version: {redis.__version__}"
        )

    # Excluded params should NOT be present
    excluded_params = ['self', 'connection_pool', 'retry', 'host', 'port', 'startup_nodes']
    for param in excluded_params:
        assert param not in kwargs, (
            f"Parameter '{param}' should be excluded but was found in kwargs"
        )

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
