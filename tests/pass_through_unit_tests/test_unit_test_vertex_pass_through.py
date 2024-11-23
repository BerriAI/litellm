import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import (
    get_litellm_virtual_key,
    vertex_proxy_route,
)


@pytest.mark.asyncio
async def test_get_litellm_virtual_key():
    """
    Test that the get_litellm_virtual_key function correctly handles the API key authentication
    """
    # Test with x-litellm-api-key
    mock_request = Mock()
    mock_request.headers = {"x-litellm-api-key": "test-key-123"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"

    # Test with Authorization header
    mock_request.headers = {"Authorization": "Bearer auth-key-456"}
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer auth-key-456"

    # Test with both headers (x-litellm-api-key should take precedence)
    mock_request.headers = {
        "x-litellm-api-key": "test-key-123",
        "Authorization": "Bearer auth-key-456",
    }
    result = get_litellm_virtual_key(mock_request)
    assert result == "Bearer test-key-123"


@pytest.mark.asyncio
async def test_vertex_proxy_route_api_key_auth():
    """
    Critical

    This is how Vertex AI JS SDK will Auth to Litellm Proxy
    """
    # Mock dependencies
    mock_request = Mock()
    mock_request.headers = {"x-litellm-api-key": "test-key-123"}
    mock_request.method = "POST"
    mock_response = Mock()

    with patch(
        "litellm.proxy.vertex_ai_endpoints.vertex_endpoints.user_api_key_auth"
    ) as mock_auth:
        mock_auth.return_value = {"api_key": "test-key-123"}

        with patch(
            "litellm.proxy.vertex_ai_endpoints.vertex_endpoints.create_pass_through_route"
        ) as mock_pass_through:
            mock_pass_through.return_value = AsyncMock(
                return_value={"status": "success"}
            )

            # Call the function
            result = await vertex_proxy_route(
                endpoint="v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
                request=mock_request,
                fastapi_response=mock_response,
            )

            # Verify user_api_key_auth was called with the correct Bearer token
            mock_auth.assert_called_once()
            call_args = mock_auth.call_args[1]
            assert call_args["api_key"] == "Bearer test-key-123"
