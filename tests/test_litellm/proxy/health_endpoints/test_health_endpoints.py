import os
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
from prisma.errors import ClientNotConnectedError, HTTPClientClosedError, PrismaError
from litellm.proxy.health_endpoints._health_endpoints import (
    _db_health_readiness_check,
    db_health_cache,
    health_services_endpoint,
)

# Import shared proxy test helpers from conftest
from tests.test_litellm.proxy.conftest import create_proxy_test_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError(),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_readiness_check_with_prisma_error(prisma_error):
    """
    Test that when prisma_client.health_check() raises a PrismaError and
    allow_requests_on_db_unavailable is True, the function should not raise an error
    and return the cached health status.
    """
    # Mock the prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.health_check.side_effect = prisma_error

    # Reset the health cache to a known state
    global db_health_cache
    db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(minutes=5),
    }

    # Patch the imports and general_settings
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        # Call the function
        result = await _db_health_readiness_check()

        # Verify that the function called health_check
        mock_prisma_client.health_check.assert_called_once()

        # Verify that the function returned the cache
        assert result is not None
        assert result["status"] == "unknown"  # Should retain the status from the cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        PrismaError(),
        ClientNotConnectedError(),
        HTTPClientClosedError(),
    ],
)
async def test_db_health_readiness_check_with_error_and_flag_off(prisma_error):
    """
    Test that when prisma_client.health_check() raises a DB error but
    allow_requests_on_db_unavailable is False, the exception should be raised.
    """
    # Mock the prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.health_check.side_effect = prisma_error

    # Reset the health cache
    global db_health_cache
    db_health_cache = {
        "status": "unknown",
        "last_updated": datetime.now() - timedelta(minutes=5),
    }

    # Patch the imports and general_settings where the flag is False
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client), patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": False},
    ):
        # The function should raise the exception
        with pytest.raises(Exception) as excinfo:
            await _db_health_readiness_check()

        # Verify that the raised exception is the same
        assert excinfo.value == prisma_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_message",
    [
        ("healthy", ""),
        ("unhealthy", "queue not reachable"),
    ],
)
async def test_health_services_endpoint_sqs(status, error_message):
    """
    Verify the /health/services SQS branch returns expected status and message
    based on SQSLogger.async_health_check().
    """
    with patch("litellm.integrations.sqs.SQSLogger") as MockSQSLogger:
        mock_instance = MagicMock()
        mock_instance.async_health_check = AsyncMock(
            return_value={"status": status, "error_message": error_message}
        )
        MockSQSLogger.return_value = mock_instance

        result = await health_services_endpoint(service="sqs")

        assert result["status"] == status
        assert result["message"] == error_message
        mock_instance.async_health_check.assert_awaited_once()


@pytest.fixture(scope="function")
def proxy_client(monkeypatch):
    """
    Fixture that starts a proxy server instance for testing.
    Uses the actual FastAPI app from proxy_server which includes all routers.
    
    Note: TestClient doesn't start a real HTTP server - it runs the FastAPI app
    in-process. However, it DOES trigger FastAPI's lifespan events (startup/shutdown)
    when used as a context manager, which initializes the proxy server components.
    
    Database access:
    - If DATABASE_URL is set in environment, the proxy will automatically connect
    - Database connection happens during lifespan startup events
    - To enable database access, set DATABASE_URL environment variable before running tests
    
    Redis cache:
    - If REDIS_HOST is set in environment, Redis cache will be automatically configured
    - Cache configuration is included in /health/readiness endpoint response
    """
    client = create_proxy_test_client(monkeypatch)
    with client:
        yield client


def test_health_liveliness_endpoint(proxy_client):
    """
    Test that /health/liveliness endpoint returns 200 OK with "I'm alive!" message.
    This is a critical orchestration endpoint that must be simple and fast.
    """
    # Measure the time taken for the health check call
    start_time = time.perf_counter()
    
    # Make GET request to /health/liveliness
    response = proxy_client.get("/health/liveliness")
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    
    # Assert response status
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    
    # Assert response content (FastAPI JSON-encodes the string)
    assert response.json() == "I'm alive!", f"Expected 'I'm alive!' message, got: {response.json()}"
    
    # Verify response is fast (should be < 100ms for a simple endpoint)
    # This is critical for orchestration systems that poll frequently
    assert duration_ms < 100, f"Health check took {duration_ms:.2f}ms, expected < 100ms for a simple endpoint"
    
    # Log the duration for visibility (useful for CI/CD monitoring)
    print(f"\n/health/liveliness response time: {duration_ms:.2f}ms")


def test_health_liveness_endpoint(proxy_client):
    """
    Test that /health/liveness endpoint (Kubernetes standard name) also works.
    """
    # Measure the time taken for the health check call
    start_time = time.perf_counter()
    
    # Make GET request to /health/liveness
    response = proxy_client.get("/health/liveness")
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    
    # Assert response status
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    
    # Assert response content (FastAPI JSON-encodes the string)
    assert response.json() == "I'm alive!", f"Expected 'I'm alive!' message, got: {response.json()}"
    
    # Verify response is fast (should be < 100ms for a simple endpoint)
    assert duration_ms < 100, f"Health check took {duration_ms:.2f}ms, expected < 100ms for a simple endpoint"
    
    # Log the duration for visibility (useful for CI/CD monitoring)
    print(f"\n/health/liveness response time: {duration_ms:.2f}ms")


def test_health_readiness(proxy_client):
    """
    Test /health/readiness endpoint.
    Database and Redis are optional - the endpoint should work whether they're available or not.
    
    If DATABASE_URL is set, the endpoint will check database connectivity.
    If REDIS_HOST is set, the endpoint will report cache status.
    If neither is set, the endpoint should still return a valid health status.
    """
    # Measure the time taken for the health check call
    start_time = time.perf_counter()
    
    # Make GET request to /health/readiness
    response = proxy_client.get("/health/readiness")
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    
    # Assert response status
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    
    # Verify response is fast (readiness may include DB check if available, so < 500ms is reasonable)
    # This is critical for orchestration systems (Kubernetes) that poll frequently
    assert duration_ms < 500, f"Health check took {duration_ms:.2f}ms, expected < 500ms for readiness endpoint"
    
    # Assert response contains expected fields
    response_data = response.json()
    assert "status" in response_data, "Response should contain 'status' field"
    assert "litellm_version" in response_data, "Response should contain 'litellm_version' field"
    
    # Display all health endpoint response fields (matches what /health/readiness returns)
    print("\n" + "-"*60)
    print("HEALTH ENDPOINT RESPONSE")
    print("-"*60)
    print(f"Status: {response_data.get('status', 'unknown')}")
    print(f"Database: {response_data.get('db', 'not reported')}")
    print(f"LiteLLM Version: {response_data.get('litellm_version', 'unknown')}")
    print(f"Success Callbacks: {response_data.get('success_callbacks', [])}")
    print(f"Cache: {response_data.get('cache', 'none')}")
    print(f"Use AioHTTP Transport: {response_data.get('use_aiohttp_transport', 'unknown')}")
    print(f"Response time: {duration_ms:.2f}ms")
    
    # If database status is reported, verify it's a valid status
    # Database may be "connected", "disconnected", "unknown", or "Not connected" (when prisma_client is None)
    if "db" in response_data:
        db_status = response_data["db"]
        # Database status can be any of these valid states
        assert db_status in ["connected", "disconnected", "unknown", "Not connected"], \
            f"Unexpected db status: {db_status}"
    
    print("="*60 + "\n")

