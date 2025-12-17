"""
Test Azure Sentinel logging integration
"""

import base64
import datetime
from unittest.mock import AsyncMock, patch

import pytest

from litellm.integrations.azure_sentinel.azure_sentinel import AzureSentinelLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.mark.asyncio
async def test_azure_sentinel_signature_and_send_batch():
    """Test that Azure Sentinel logger builds correct signature and sends batch to API"""
    test_workspace_id = "test-workspace-id"
    test_shared_key = base64.b64encode(b"test-key").decode()

    with patch("asyncio.create_task"):
        logger = AzureSentinelLogger(
            workspace_id=test_workspace_id, shared_key=test_shared_key
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

    # Mock HTTP client
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = "Success"
    logger.async_httpx_client.post = AsyncMock(return_value=mock_response)

    # Send batch
    await logger.async_send_batch()

    # Verify request was made
    assert logger.async_httpx_client.post.called

    # Verify request parameters
    call_args = logger.async_httpx_client.post.call_args
    assert call_args.kwargs["url"] == logger.api_endpoint

    # Verify headers
    headers = call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert headers["Log-Type"] == "LiteLLM"
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("SharedKey ")
    assert test_workspace_id in headers["Authorization"]
    assert "x-ms-date" in headers

    # Verify queue is cleared
    assert len(logger.log_queue) == 0


def test_azure_sentinel_init_validation():
    """Test that Azure Sentinel logger validates required parameters"""
    with patch("asyncio.create_task"):
        # Test missing workspace_id
        with pytest.raises(ValueError, match="AZURE_SENTINEL_WORKSPACE_ID is required"):
            AzureSentinelLogger(shared_key=base64.b64encode(b"test-key").decode())

        # Test missing shared_key
        with pytest.raises(ValueError, match="AZURE_SENTINEL_SHARED_KEY is required"):
            AzureSentinelLogger(workspace_id="test-workspace-id")

        # Test successful init
        logger = AzureSentinelLogger(
            workspace_id="test-workspace-id",
            shared_key=base64.b64encode(b"test-key").decode(),
        )
        assert logger.workspace_id == "test-workspace-id"
        assert logger.log_type == "LiteLLM"
