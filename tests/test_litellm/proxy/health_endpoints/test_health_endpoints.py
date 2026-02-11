import os
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
from prisma.errors import ClientNotConnectedError, HTTPClientClosedError, PrismaError

from litellm.proxy.health_endpoints._health_endpoints import (
    _db_health_readiness_check,
    db_health_cache,
    get_callback_identifier,
    health_license_endpoint,
    health_services_endpoint,
)
from litellm.proxy.health_endpoints._health_endpoints import (
    test_model_connection as health_test_model_connection,
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


@pytest.mark.asyncio
async def test_health_license_endpoint_with_active_license():
    license_data = {
        "expiration_date": "2099-01-01",
        "allowed_features": ["feature-a"],
        "max_users": 100,
        "max_teams": 5,
    }
    mock_license_check = SimpleNamespace(
        license_str="test-license",
        public_key=None,
        airgapped_license_data=license_data,
        verify_license_without_api_request=MagicMock(return_value=True),
    )

    with patch(
        "litellm.proxy.proxy_server._license_check",
        mock_license_check,
    ), patch(
        "litellm.proxy.proxy_server.premium_user",
        True,
    ), patch(
        "litellm.proxy.proxy_server.premium_user_data",
        license_data,
    ):
        response = await health_license_endpoint(user_api_key_dict=MagicMock())

    assert response["has_license"] is True
    assert response["license_type"] == "enterprise"
    assert response["expiration_date"] == "2099-01-01"
    assert response["allowed_features"] == ["feature-a"]
    assert response["limits"] == {"max_users": 100, "max_teams": 5}


@pytest.mark.asyncio
async def test_health_license_endpoint_without_valid_license():
    mock_license_check = SimpleNamespace(
        license_str="invalid-key",
        public_key=None,
        airgapped_license_data=None,
        verify_license_without_api_request=MagicMock(return_value=False),
    )

    with patch(
        "litellm.proxy.proxy_server._license_check",
        mock_license_check,
    ), patch(
        "litellm.proxy.proxy_server.premium_user",
        False,
    ), patch(
        "litellm.proxy.proxy_server.premium_user_data",
        None,
    ):
        response = await health_license_endpoint(user_api_key_dict=MagicMock())

    assert response["has_license"] is True
    assert response["license_type"] == "community"
    assert response["expiration_date"] is None
    assert response["allowed_features"] == []
    assert response["limits"] == {"max_users": None, "max_teams": None}


@pytest.mark.asyncio
async def test_test_model_connection_loads_config_from_router():
    """
    Test that /health/test_connection automatically loads model configuration
    (including resolved environment variables) from the router when model name is provided.
    """
    # Mock request
    mock_request = MagicMock()
    
    # Mock user_api_key_dict
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.token = "test-token"
    
    # Mock prisma_client
    mock_prisma_client = MagicMock()
    
    # Mock router with model configuration
    mock_router = MagicMock()
    mock_deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "azure/gpt-4o",
            "api_key": "resolved-api-key-from-env",
            "api_base": "https://resolved-endpoint.openai.azure.com/",
            "api_version": "2024-10-21",
        },
        "model_info": {},
    }
    mock_router.get_model_list.return_value = [mock_deployment]
    
    # Mock ModelManagementAuthChecks - patch at the source module since it's imported inside the function
    mock_can_user_make_model_call = AsyncMock()
    
    # Mock litellm.ahealth_check
    mock_health_check_result = {
        "status": "healthy",
        "response_time_ms": 100,
    }
    mock_ahealth_check = AsyncMock(return_value=mock_health_check_result)
    
    # Mock run_with_timeout
    mock_run_with_timeout = AsyncMock(return_value=mock_health_check_result)
    
    # Mock _update_litellm_params_for_health_check
    def mock_update_params(model_info, litellm_params):
        # Just return params with messages added
        params = litellm_params.copy()
        params["messages"] = [{"role": "user", "content": "test"}]
        return params
    
    # Mock _resolve_os_environ_variables
    def mock_resolve_os_environ(params):
        return params
    
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client,
    ), patch(
        "litellm.proxy.proxy_server.llm_router",
        mock_router,
    ), patch(
        "litellm.proxy.proxy_server.premium_user",
        False,
    ), patch(
        "litellm.proxy.management_endpoints.model_management_endpoints.ModelManagementAuthChecks.can_user_make_model_call",
        mock_can_user_make_model_call,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.litellm.ahealth_check",
        mock_ahealth_check,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints.run_with_timeout",
        mock_run_with_timeout,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints._update_litellm_params_for_health_check",
        mock_update_params,
    ), patch(
        "litellm.proxy.health_endpoints._health_endpoints._resolve_os_environ_variables",
        mock_resolve_os_environ,
    ):
        # Call the endpoint with only model name (no credentials)
        result = await health_test_model_connection(
            request=mock_request,
            mode="chat",
            litellm_params={"model": "gpt-4o"},
            model_info={},
            user_api_key_dict=mock_user_api_key_dict,
        )
        
        # Verify router.get_model_list was called with the model name
        mock_router.get_model_list.assert_called_once_with(model_name="gpt-4o")
        
        # Verify that run_with_timeout was called (which wraps ahealth_check)
        assert mock_run_with_timeout.called
        
        # Get the call args to verify merged params
        call_args = mock_run_with_timeout.call_args
        assert call_args is not None
        
        # The first arg should be the coroutine from ahealth_check
        # We need to check what was passed to ahealth_check
        ahealth_check_call_args = mock_ahealth_check.call_args
        assert ahealth_check_call_args is not None
        model_params = ahealth_check_call_args.kwargs.get("model_params", {})
        
        # Verify that config params were loaded and merged
        # Note: request params override config params, so model from request is used
        assert model_params.get("api_key") == "resolved-api-key-from-env"
        assert model_params.get("api_base") == "https://resolved-endpoint.openai.azure.com/"
        assert model_params.get("api_version") == "2024-10-21"
        assert model_params.get("model") == "gpt-4o"  # Request param overrides config param
        
        # Verify result
        assert result["status"] == "success"
        assert "result" in result


@pytest.mark.asyncio
async def test_health_services_endpoint_datadog_llm_observability():
    """
    Verify that 'datadog_llm_observability' is accepted as a valid service
    by the /health/services endpoint and does not raise a 400 error.

    Regression test for: https://github.com/BerriAI/litellm/issues/XXXX
    The service was missing from the allowed services validation list.
    """
    from litellm.proxy.health_endpoints._health_endpoints import (
        health_services_endpoint,
    )

    # Mock datadog_llm_observability to be in success_callback so the generic branch handles it
    with patch("litellm.success_callback", ["datadog_llm_observability"]):
        result = await health_services_endpoint(
            service="datadog_llm_observability"
        )

    # Should not raise HTTPException(400) and should return success
    assert result["status"] == "success"
    assert "datadog_llm_observability" in result["message"]


@pytest.mark.asyncio
async def test_health_services_endpoint_rejects_unknown_service():
    """
    Verify that an unknown service name is rejected with a 400 error.
    """
    from litellm.proxy._types import ProxyException

    with pytest.raises(ProxyException):
        await health_services_endpoint(
            service="totally_unknown_service_xyz"
        )


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


def test_get_callback_identifier_string_and_object_with_callback_name():
    """
    Test get_callback_identifier with string callbacks and objects with callback_name attribute.
    
    Covers:
    - String callback (returned as-is)
    - Object with callback_name attribute
    - Object with empty/None callback_name (should fall through to other checks)
    """
    from litellm.proxy.health_endpoints._health_endpoints import get_callback_identifier
    
    # Test 1: String callback should be returned as-is
    assert get_callback_identifier("datadog") == "datadog"
    assert get_callback_identifier("langfuse") == "langfuse"
    
    # Test 2: Object with callback_name attribute
    class MockCallbackWithName:
        def __init__(self, name):
            self.callback_name = name
    
    callback_obj = MockCallbackWithName("custom_callback")
    assert get_callback_identifier(callback_obj) == "custom_callback"
    
    # Test 3: Object with empty callback_name should fall through
    callback_obj_empty = MockCallbackWithName("")
    # This should fall through to CustomLoggerRegistry or callback_name() fallback
    # We'll verify it doesn't return empty string
    result = get_callback_identifier(callback_obj_empty)
    assert result != ""  # Should not return empty string
    assert isinstance(result, str)  # Should still return a string


def test_get_callback_identifier_custom_logger_registry_and_fallback():
    """
    Test get_callback_identifier with CustomLoggerRegistry lookup and fallback scenarios.
    
    Covers:
    - Object registered in CustomLoggerRegistry
    - Object with callback_name that matches registry entry
    - Fallback to callback_name() helper function
    """
    from litellm.proxy.health_endpoints._health_endpoints import get_callback_identifier
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry
    
    # Test 1: Object registered in CustomLoggerRegistry (without callback_name attribute)
    # Mock a class that's registered in the registry
    class MockRegisteredLogger:
        pass
    
    # Mock the registry to return callback strings for our mock class
    with patch.object(
        CustomLoggerRegistry,
        'get_all_callback_strs_from_class_type',
        return_value=['mock_logger']
    ):
        mock_instance = MockRegisteredLogger()
        result = get_callback_identifier(mock_instance)
        assert result == "mock_logger"
    
    # Test 2: Object with callback_name that matches registry entry
    class MockCallbackWithMatchingName:
        def __init__(self):
            self.callback_name = "matched_name"
    
    callback_with_matching = MockCallbackWithMatchingName()
    # Mock registry to return list containing the matching name
    with patch.object(
        CustomLoggerRegistry,
        'get_all_callback_strs_from_class_type',
        return_value=['matched_name', 'other_name']
    ):
        result = get_callback_identifier(callback_with_matching)
        assert result == "matched_name"
    
    # Test 3: Object with falsy callback_name (empty string), should use registry
    class MockCallbackWithEmptyName:
        def __init__(self):
            self.callback_name = ""  # Empty string is falsy
    
    callback_empty = MockCallbackWithEmptyName()
    # Mock registry to return list - should use first registry entry since callback_name is falsy
    with patch.object(
        CustomLoggerRegistry,
        'get_all_callback_strs_from_class_type',
        return_value=['registry_name']
    ):
        result = get_callback_identifier(callback_empty)
        assert result == "registry_name"
    
    # Test 3b: Object with truthy callback_name not in registry - returns callback_name immediately
    # (This tests that truthy callback_name takes precedence over registry)
    class MockCallbackWithNonMatchingName:
        def __init__(self):
            self.callback_name = "non_matching"
    
    callback_non_matching = MockCallbackWithNonMatchingName()
    # Even if registry has different values, truthy callback_name is returned first
    with patch.object(
        CustomLoggerRegistry,
        'get_all_callback_strs_from_class_type',
        return_value=['registry_name']
    ):
        result = get_callback_identifier(callback_non_matching)
        # Should return callback_name because it's truthy (checked before registry)
        assert result == "non_matching"
    
    # Test 4: Object not in registry, falls back to callback_name() helper
    class UnregisteredCallback:
        def __init__(self):
            pass
    
    unregistered = UnregisteredCallback()
    # Mock registry to return empty list (not registered)
    with patch.object(
        CustomLoggerRegistry,
        'get_all_callback_strs_from_class_type',
        return_value=[]
    ):
        result = get_callback_identifier(unregistered)
        # Should fall back to callback_name() which returns __class__.__name__
        assert result == "UnregisteredCallback"
    
    # Test 5: Function callback (not a class instance)
    def my_callback_function():
        pass
    
    # Function won't have __class__, so it will skip registry check and go to callback_name()
    result = get_callback_identifier(my_callback_function)
    # Should fall back to callback_name() which returns __name__
    assert result == "my_callback_function"
