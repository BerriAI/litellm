"""
Test Azure Sentinel logging integration
"""

import datetime
from unittest.mock import AsyncMock, patch

import pytest

from litellm.integrations.azure_sentinel.azure_sentinel import AzureSentinelLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.mark.asyncio
async def test_azure_sentinel_oauth_and_send_batch():
    """Test that Azure Sentinel logger gets OAuth token and sends batch to API"""
    test_dcr_id = "dcr-test123456789"
    test_endpoint = "https://test-dce.eastus-1.ingest.monitor.azure.com"
    test_tenant_id = "test-tenant-id"
    test_client_id = "test-client-id"
    test_client_secret = "test-client-secret"

    with patch("asyncio.create_task"):
        logger = AzureSentinelLogger(
            dcr_immutable_id=test_dcr_id,
            endpoint=test_endpoint,
            tenant_id=test_tenant_id,
            client_id=test_client_id,
            client_secret=test_client_secret,
        )

    # Create test payload
    standard_payload = StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        model="gpt-3.5-turbo",
        status="success",
        messages=[{"role": "user", "content": "Hello"}],
        response={"choices": [{"message": {"content": "Hi"}}]},
    )

    # Add to queue
    logger.log_queue.append(standard_payload)

    # Mock OAuth token response
    from unittest.mock import MagicMock
    
    mock_token_response = MagicMock()
    mock_token_response.status_code = 200
    mock_token_response.json = MagicMock(return_value={
        "access_token": "test-bearer-token",
        "expires_in": 3600,
    })
    mock_token_response.text = "Success"

    # Mock API response
    mock_api_response = MagicMock()
    mock_api_response.status_code = 204
    mock_api_response.text = "Success"

    # Mock HTTP client - first call for token, second for API
    async def mock_post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            return mock_token_response
        return mock_api_response

    logger.async_httpx_client.post = AsyncMock(side_effect=mock_post)

    # Send batch
    await logger.async_send_batch()

    # Verify OAuth token request was made
    assert logger.async_httpx_client.post.called

    # Verify API request was made
    call_count = logger.async_httpx_client.post.call_count
    assert call_count >= 2  # At least token + API call

    # Get the API call (last call)
    api_call_args = logger.async_httpx_client.post.call_args_list[-1]
    assert test_dcr_id in api_call_args.kwargs["url"]
    assert test_endpoint in api_call_args.kwargs["url"]

    # Verify headers
    headers = api_call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Bearer ")

    # Verify queue is cleared
    assert len(logger.log_queue) == 0


def test_azure_sentinel_init_validation():
    """Test that Azure Sentinel logger validates required parameters"""
    with patch("asyncio.create_task"):
        # Test missing dcr_immutable_id
        with pytest.raises(ValueError, match="AZURE_SENTINEL_DCR_IMMUTABLE_ID is required"):
            AzureSentinelLogger(
                endpoint="https://test.com",
                tenant_id="test-tenant",
                client_id="test-client",
                client_secret="test-secret",
            )

        # Test missing endpoint
        with pytest.raises(ValueError, match="AZURE_SENTINEL_ENDPOINT is required"):
            AzureSentinelLogger(
                dcr_immutable_id="dcr-test",
                tenant_id="test-tenant",
                client_id="test-client",
                client_secret="test-secret",
            )

        # Test missing tenant_id
        with pytest.raises(ValueError, match="AZURE_SENTINEL_TENANT_ID"):
            AzureSentinelLogger(
                dcr_immutable_id="dcr-test",
                endpoint="https://test.com",
                client_id="test-client",
                client_secret="test-secret",
            )

        # Test successful init
        logger = AzureSentinelLogger(
            dcr_immutable_id="dcr-test123",
            endpoint="https://test-dce.eastus-1.ingest.monitor.azure.com",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )
        assert logger.dcr_immutable_id == "dcr-test123"
        assert logger.stream_name == "Custom-LiteLLM"
        assert "dcr-test123" in logger.api_endpoint
