import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import litellm
from litellm.caching import RedisCache
from litellm.proxy.proxy_server import app

client = TestClient(app)


# Mock successful Redis connection
@pytest.fixture
def mock_redis_success(mocker):
    async def mock_ping():
        return True

    async def mock_add_cache(*args, **kwargs):
        return None

    mock_cache = mocker.MagicMock()
    mock_cache.type = "redis"
    mock_cache.ping = mock_ping
    mock_cache.async_add_cache = mock_add_cache
    mock_cache.cache = RedisCache(
        host="localhost",
        port=6379,
        password="hello",
    )

    mocker.patch.object(litellm, "cache", mock_cache)
    return mock_cache


# Mock failed Redis connection
@pytest.fixture
def mock_redis_failure(mocker):
    async def mock_ping():
        raise Exception("invalid username-password pair")

    mock_cache = mocker.MagicMock()
    mock_cache.type = "redis"
    mock_cache.ping = mock_ping

    mocker.patch.object(litellm, "cache", mock_cache)
    return mock_cache


def test_cache_ping_success(mock_redis_success):
    """Test successful cache ping with regular response"""
    response = client.get("/cache/ping", headers={"Authorization": "Bearer sk-1234"})
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["cache_type"] == "redis"
    assert data["ping_response"] is True
    assert data["set_cache_response"] == "success"


def test_cache_ping_with_complex_objects(mock_redis_success, mocker):
    """Test cache ping with non-standard serializable objects"""

    # Mock complex objects in the cache parameters
    class ComplexObject:
        def __str__(self):
            return "complex_object"

    mock_redis_success.cache.complex_attr = ComplexObject()
    mock_redis_success.cache.datetime_attr = mocker.MagicMock()

    response = client.get("/cache/ping", headers={"Authorization": "Bearer sk-1234"})
    assert response.status_code == 200

    # Verify response is JSON serializable
    data = response.json()
    print("data=", json.dumps(data, indent=4))
    assert data["status"] == "healthy"
    assert "litellm_cache_params" in data

    # Verify complex objects were converted to strings
    cache_params = json.loads(data["litellm_cache_params"])
    assert isinstance(cache_params, dict)


def test_cache_ping_with_circular_reference(mock_redis_success):
    """Test cache ping with circular reference in cache parameters"""
    # Create circular reference
    circular_dict = {}
    circular_dict["self"] = circular_dict
    mock_redis_success.cache.circular_ref = circular_dict

    response = client.get("/cache/ping", headers={"Authorization": "Bearer sk-1234"})
    assert response.status_code == 200

    # Verify response is still JSON serializable
    data = response.json()
    assert data["status"] == "healthy"


def test_cache_ping_failure(mock_redis_failure):
    """Test cache ping failure with expected error fields"""
    response = client.get("/cache/ping", headers={"Authorization": "Bearer sk-1234"})
    assert response.status_code == 503

    data = response.json()
    print("data=", json.dumps(data, indent=4, default=str))

    assert "error" in data
    error = data["error"]

    # Verify error contains all expected fields
    assert "message" in error
    error_details = json.loads(error["message"])
    assert "message" in error_details
    assert "litellm_cache_params" in error_details
    assert "health_check_cache_params" in error_details
    assert "traceback" in error_details

    # Verify specific error message
    assert "invalid username-password pair" in error_details["message"]


def test_cache_ping_no_cache_initialized():
    """Test cache ping when no cache is initialized"""
    # Set cache to None
    original_cache = litellm.cache
    litellm.cache = None

    response = client.get("/cache/ping", headers={"Authorization": "Bearer sk-1234"})
    assert response.status_code == 503

    data = response.json()
    print("response data=", json.dumps(data, indent=4))
    assert "error" in data
    error = data["error"]

    # Verify error contains all expected fields
    assert "message" in error
    error_details = json.loads(error["message"])
    assert "Cache not initialized. litellm.cache is None" in error_details["message"]

    # Restore original cache
    litellm.cache = original_cache


def test_cache_ping_health_check_includes_only_cache_attributes(mock_redis_success):
    """
    Ensure that the /cache/ping endpoint only pulls HealthCheckCacheParams from litellm.cache.cache,
    and not from other attributes on litellm.cache.
    """
    # Add an unrelated field directly to the cache mock; it should NOT appear in health_check_cache_params
    mock_redis_success.some_unrelated_field = "should-not-appear-in-health-check"

    # Add a field on the underlying `cache` object that SHOULD appear
    mock_redis_success.cache.redis_kwargs = {"host": "localhost", "port": 6379}

    response = client.get("/cache/ping", headers={"Authorization": "Bearer sk-1234"})
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"

    data = response.json()
    print("/cache/ping response data=", json.dumps(data, indent=4))
    health_check_cache_params = data.get("health_check_cache_params", {})
    # The unrelated field we attached at the top-level of litellm.cache should *not* be present
    assert (
        "some_unrelated_field" not in health_check_cache_params
    ), "Found an unexpected field from the mock_redis_success object in health_check_cache_params"

    # The field we attached to 'mock_redis_success.cache' should be present and correctly reported
    assert (
        "redis_kwargs" in health_check_cache_params
    ), "Expected field on `litellm.cache.cache` was not found in health_check_cache_params"
    assert health_check_cache_params["redis_kwargs"] == {
        "host": "localhost",
        "port": 6379,
    }


def test_cache_ping_with_redis_version_float(mock_redis_success):
    """Test cache ping works when redis_version is a float"""
    # Set redis_version as a float
    mock_redis_success.cache.redis_version = 7.2

    response = client.get("/cache/ping", headers={"Authorization": "Bearer sk-1234"})
    assert response.status_code == 200

    data = response.json()
    print("data=", json.dumps(data, indent=4))
    assert data["status"] == "healthy"
    assert data["cache_type"] == "redis"

    cache_params = data["health_check_cache_params"]
    assert isinstance(cache_params, dict)
    assert isinstance(cache_params.get("redis_version"), float)


@pytest.fixture
def mock_redis_client_list_restricted(mocker):
    """Mock Redis cache where CLIENT LIST is restricted (like GCP Redis)"""

    def mock_client_list():
        raise Exception("ERR unknown command 'CLIENT'")

    def mock_info():
        return {
            "redis_version": "6.2.7",
            "used_memory": "1000000",
            "connected_clients": "5",
            "keyspace_hits": "1000",
            "keyspace_misses": "100",
        }

    mock_cache = mocker.MagicMock()
    mock_cache.type = "redis"
    mock_cache.cache = RedisCache(host="localhost", port=6379, password="hello")
    mock_cache.cache.client_list = mock_client_list
    mock_cache.cache.info = mock_info

    mocker.patch.object(litellm, "cache", mock_cache)
    return mock_cache


@pytest.fixture
def mock_redis_client_list_success(mocker):
    """Mock Redis cache where CLIENT LIST works normally"""

    def mock_client_list():
        return [
            {"id": "1", "addr": "127.0.0.1:54321", "name": "client1"},
            {"id": "2", "addr": "127.0.0.1:54322", "name": "client2"},
        ]

    def mock_info():
        return {
            "redis_version": "6.2.7",
            "used_memory": "1000000",
            "connected_clients": "2",
        }

    mock_cache = mocker.MagicMock()
    mock_cache.type = "redis"
    mock_cache.cache = RedisCache(host="localhost", port=6379, password="hello")
    mock_cache.cache.client_list = mock_client_list
    mock_cache.cache.info = mock_info

    mocker.patch.object(litellm, "cache", mock_cache)
    return mock_cache


def test_cache_redis_info_no_cache():
    """Test /cache/redis/info when no cache is initialized"""
    original_cache = litellm.cache
    litellm.cache = None

    response = client.get(
        "/cache/redis/info", headers={"Authorization": "Bearer sk-1234"}
    )
    assert response.status_code == 503

    data = response.json()
    assert "Cache not initialized" in data["detail"]

    # Restore original cache
    litellm.cache = original_cache
