import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock

from litellm.caching.redis_cache import RedisCache
from litellm._redis import _create_gcp_iam_auth_function, _create_gcp_iam_async_auth_function


@pytest.fixture
def redis_no_ping():
    """Patch RedisCache initialization to prevent async ping tasks from being created"""
    with patch("asyncio.get_running_loop") as mock_get_loop:
        # Either raise an exception or return a mock that will handle the task creation
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.mark.parametrize("namespace", [None, "test"])
@pytest.mark.asyncio
async def test_redis_cache_async_increment(namespace, monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache(namespace=namespace)
    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()

    # Make sure the mock can be used as an async context manager
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    assert redis_cache is not None

    expected_key = "test:test" if namespace else "test"

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        # Call async_set_cache
        await redis_cache.async_increment(key=expected_key, value=1)

        # Verify that the set method was called on the mock Redis instance
        mock_redis_instance.incrbyfloat.assert_called_once_with(
            name=expected_key, amount=1
        )


@pytest.mark.asyncio
async def test_redis_client_init_with_socket_timeout(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "my-fake-host")
    redis_cache = RedisCache(socket_timeout=1.0)
    assert redis_cache.redis_kwargs["socket_timeout"] == 1.0
    client = redis_cache.init_async_client()
    assert client is not None
    assert client.connection_pool.connection_kwargs["socket_timeout"] == 1.0


@pytest.mark.asyncio
async def test_redis_cache_async_batch_get_cache(monkeypatch, redis_no_ping):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    # Create an AsyncMock for the Redis client
    mock_redis_instance = AsyncMock()

    # Make sure the mock can be used as an async context manager
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    # Setup the return value for mget
    mock_redis_instance.mget.return_value = [
        b'{"key1": "value1"}',
        None,
        b'{"key3": "value3"}',
    ]

    test_keys = ["key1", "key2", "key3"]

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        # Call async_batch_get_cache
        result = await redis_cache.async_batch_get_cache(key_list=test_keys)

        # Verify mget was called with the correct keys
        mock_redis_instance.mget.assert_called_once()

        # Check that results were properly decoded
        assert result["key1"] == {"key1": "value1"}
        assert result["key2"] is None
        assert result["key3"] == {"key3": "value3"}


@pytest.mark.asyncio
async def test_handle_lpop_count_for_older_redis_versions(monkeypatch):
    """Test the helper method that handles LPOP with count for Redis versions < 7.0"""
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    # Create RedisCache instance
    redis_cache = RedisCache()

    # Create a mock pipeline
    mock_pipeline = AsyncMock()
    # Set up execute to return different values each time
    mock_pipeline.execute.side_effect = [
        [b"value1"],  # First execute returns first value
        [b"value2"],  # Second execute returns second value
    ]

    # Test the helper method
    result = await redis_cache.handle_lpop_count_for_older_redis_versions(
        pipe=mock_pipeline, key="test_key", count=2
    )

    # Verify results
    assert result == [b"value1", b"value2"]
    assert mock_pipeline.lpop.call_count == 2
    assert mock_pipeline.execute.call_count == 2


@pytest.mark.asyncio
async def test_redis_cache_gcp_iam_authentication(monkeypatch, redis_no_ping):
    """Test Redis cache initialization with Google Cloud IAM authentication"""
    monkeypatch.setenv("REDIS_HOST", "my-fake-redis-host")
    
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    
    # Test RedisCache initialization with GCP IAM service account
    redis_cache = RedisCache(
        gcp_iam_service_account=service_account,
        ssl=True
    )
    
    # Verify that the service account parameter was passed through
    assert redis_cache.redis_kwargs["gcp_iam_service_account"] == service_account
    assert redis_cache.redis_kwargs["ssl"] is True


def test_create_gcp_iam_auth_function():
    """Test the creation of GCP IAM authentication function"""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    
    # Test that the function is created successfully
    auth_func = _create_gcp_iam_auth_function(service_account)
    assert callable(auth_func)
    
    # The function should be named iam_connect
    assert auth_func.__name__ == "iam_connect"


def test_create_gcp_iam_async_auth_function():
    """Test the creation of async GCP IAM authentication function"""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    
    # Test that the async function is created successfully
    auth_func = _create_gcp_iam_async_auth_function(service_account)
    assert callable(auth_func)
    
    # The function should be named iam_connect_async
    assert auth_func.__name__ == "iam_connect_async"


@patch('litellm._redis.iam_credentials_v1.IAMCredentialsClient')
def test_gcp_iam_auth_function_execution(mock_iam_client_class):
    """Test execution of GCP IAM authentication function"""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    
    # Mock the IAM credentials client and response
    mock_client = MagicMock()
    mock_iam_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.access_token = "mock-access-token"
    mock_client.generate_access_token.return_value = mock_response
    
    # Mock connection object
    mock_connection = MagicMock()
    mock_connection._parser = MagicMock()
    mock_connection.read_response.return_value = "OK"
    
    # Create and test the auth function
    auth_func = _create_gcp_iam_auth_function(service_account)
    
    with patch('litellm._redis.str_if_bytes', return_value="OK"):
        # This should not raise an exception
        auth_func(mock_connection)
    
    # Verify that the IAM client was called correctly
    mock_client.generate_access_token.assert_called_once()
    request_arg = mock_client.generate_access_token.call_args[0][0]
    assert request_arg.name == service_account
    assert request_arg.scope == ['https://www.googleapis.com/auth/cloud-platform']
    
    # Verify connection methods were called
    mock_connection._parser.on_connect.assert_called_once_with(mock_connection)
    mock_connection.send_command.assert_called_with("AUTH", "mock-access-token", check_health=False)
    mock_connection.read_response.assert_called_once()


@patch('litellm._redis.iam_credentials_v1.IAMCredentialsClient')
@pytest.mark.asyncio
async def test_gcp_iam_async_auth_function_execution(mock_iam_client_class):
    """Test execution of async GCP IAM authentication function"""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    
    # Mock the IAM credentials client and response
    mock_client = MagicMock()
    mock_iam_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.access_token = "mock-access-token"
    mock_client.generate_access_token.return_value = mock_response
    
    # Mock async connection object
    mock_connection = AsyncMock()
    mock_connection._parser = AsyncMock()
    mock_connection.read_response.return_value = "OK"
    
    # Create and test the async auth function
    auth_func = _create_gcp_iam_async_auth_function(service_account)
    
    with patch('litellm._redis.str_if_bytes', return_value="OK"):
        # This should not raise an exception
        await auth_func(mock_connection)
    
    # Verify that the IAM client was called correctly
    mock_client.generate_access_token.assert_called_once()
    request_arg = mock_client.generate_access_token.call_args[0][0]
    assert request_arg.name == service_account
    assert request_arg.scope == ['https://www.googleapis.com/auth/cloud-platform']
    
    # Verify async connection methods were called
    mock_connection._parser.on_connect.assert_called_once_with(mock_connection)
    mock_connection.send_command.assert_called_with("AUTH", "mock-access-token", check_health=False)
    mock_connection.read_response.assert_called_once()


def test_gcp_iam_auth_function_missing_import():
    """Test that appropriate error is raised when google-cloud-iam is not available"""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    
    auth_func = _create_gcp_iam_auth_function(service_account)
    mock_connection = MagicMock()
    
    # Mock ImportError for google.cloud.iam_credentials_v1
    with patch.dict('sys.modules', {'google.cloud.iam_credentials_v1': None}):
        with pytest.raises(ImportError, match="Google Cloud IAM authentication requires the 'google-cloud-iam' package"):
            auth_func(mock_connection)


@patch('litellm._redis.iam_credentials_v1.IAMCredentialsClient')
def test_gcp_iam_auth_function_authentication_error(mock_iam_client_class):
    """Test authentication error handling in GCP IAM auth function"""
    service_account = "projects/-/serviceAccounts/test@project.iam.gserviceaccount.com"
    
    # Mock the IAM credentials client and response
    mock_client = MagicMock()
    mock_iam_client_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.access_token = "mock-access-token"
    mock_client.generate_access_token.return_value = mock_response
    
    # Mock connection object with failed auth response
    mock_connection = MagicMock()
    mock_connection._parser = MagicMock()
    mock_connection.read_response.return_value = "ERROR"
    
    auth_func = _create_gcp_iam_auth_function(service_account)
    
    with patch('litellm._redis.str_if_bytes', return_value="ERROR"):
        with patch('litellm._redis.AuthenticationError') as mock_auth_error:
            mock_auth_error.side_effect = Exception("Invalid Username or Password")
            
            with pytest.raises(Exception, match="Invalid Username or Password"):
                auth_func(mock_connection)


@pytest.mark.asyncio
async def test_redis_cache_gcp_iam_environment_variable(monkeypatch, redis_no_ping):
    """Test Redis cache initialization with GCP IAM service account from environment variable"""
    service_account = "projects/-/serviceAccounts/env-test@project.iam.gserviceaccount.com"
    
    # Set environment variable
    monkeypatch.setenv("REDIS_GCP_IAM_SERVICE_ACCOUNT", service_account)
    monkeypatch.setenv("REDIS_HOST", "my-fake-redis-host")
    
    # Test RedisCache initialization without explicit parameter
    redis_cache = RedisCache()
    
    # Verify that the service account was picked up from environment
    assert redis_cache.redis_kwargs["gcp_iam_service_account"] == service_account
