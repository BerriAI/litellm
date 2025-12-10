import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest
from prisma.errors import ClientNotConnectedError, HTTPClientClosedError, PrismaError

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.health_endpoints._health_endpoints import (
    _db_health_readiness_check,
    db_health_cache,
    health_services_endpoint,
    test_model_connection as health_test_model_connection,
)


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

