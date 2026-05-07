"""
Test Azure Sentinel logging integration
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.azure_sentinel.azure_sentinel import AzureSentinelLogger
from litellm.types.utils import StandardAuditLogPayload, StandardLoggingPayload


def _close_periodic_flush_task(coro):
    coro.close()
    return None


@pytest.mark.asyncio
async def test_azure_sentinel_oauth_and_send_batch():
    """Test that Azure Sentinel logger gets OAuth token and sends batch to API"""
    test_dcr_id = "dcr-test123456789"
    test_endpoint = "https://test-dce.eastus-1.ingest.monitor.azure.com"
    test_tenant_id = "test-tenant-id"
    test_client_id = "test-client-id"
    test_client_secret = "test-client-secret"

    with patch("asyncio.create_task", side_effect=_close_periodic_flush_task):
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

    mock_token_response = MagicMock()
    mock_token_response.status_code = 200
    mock_token_response.json = MagicMock(
        return_value={
            "access_token": "test-bearer-token",
            "expires_in": 3600,
        }
    )
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


@pytest.mark.asyncio
async def test_azure_sentinel_queues_audit_log_event():
    """Test that Azure Sentinel supports direct audit log callbacks"""
    with patch("asyncio.create_task", side_effect=_close_periodic_flush_task):
        logger = AzureSentinelLogger(
            dcr_immutable_id="dcr-test123456789",
            endpoint="https://test-dce.eastus-1.ingest.monitor.azure.com",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

    logger.batch_size = 2
    logger.async_send_audit_batch = AsyncMock()

    audit_log = StandardAuditLogPayload(
        id="audit-123",
        updated_at="2026-05-06T04:39:00+00:00",
        changed_by="user-1",
        changed_by_api_key="sk-test",
        action="created",
        table_name="LiteLLM_TeamTable",
        object_id="team-1",
        before_value=None,
        updated_values='{"team_alias": "sentinel-demo"}',
    )

    await logger.async_log_audit_log_event(audit_log)

    assert logger.audit_log_queue == [audit_log]
    logger.async_send_audit_batch.assert_not_called()

    await logger.async_log_audit_log_event(audit_log)

    assert logger.audit_log_queue == [audit_log, audit_log]
    logger.async_send_audit_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_azure_sentinel_sends_audit_log_payload_to_ingestion_api():
    """Test that queued audit logs are sent to Azure Monitor Logs Ingestion"""
    with patch("asyncio.create_task", side_effect=_close_periodic_flush_task):
        logger = AzureSentinelLogger(
            dcr_immutable_id="dcr-test123456789",
            endpoint="https://test-dce.eastus-1.ingest.monitor.azure.com",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

    audit_log = StandardAuditLogPayload(
        id="audit-123",
        updated_at="2026-05-06T04:39:00+00:00",
        changed_by="user-1",
        changed_by_api_key="sk-test",
        action="created",
        table_name="LiteLLM_TeamTable",
        object_id="team-1",
        before_value=None,
        updated_values='{"team_alias": "sentinel-demo"}',
    )
    await logger.async_log_audit_log_event(audit_log)

    mock_token_response = MagicMock()
    mock_token_response.status_code = 200
    mock_token_response.json = MagicMock(
        return_value={
            "access_token": "test-bearer-token",
            "expires_in": 3600,
        }
    )
    mock_token_response.text = "Success"

    mock_api_response = MagicMock()
    mock_api_response.status_code = 204
    mock_api_response.text = "Success"

    async def mock_post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            return mock_token_response
        return mock_api_response

    logger.async_httpx_client.post = AsyncMock(side_effect=mock_post)

    await logger.flush_queue()

    api_call_args = logger.async_httpx_client.post.call_args_list[-1]
    body = json.loads(api_call_args.kwargs["data"].decode("utf-8"))
    assert body == [audit_log]
    assert "dcr-test123456789" in api_call_args.kwargs["url"]
    assert "Custom-LiteLLM" in api_call_args.kwargs["url"]
    assert len(logger.audit_log_queue) == 0


@pytest.mark.asyncio
async def test_azure_sentinel_flushes_standard_and_audit_logs_separately():
    """Test mixed callback roles do not send schema-mismatched batches."""
    with patch("asyncio.create_task", side_effect=_close_periodic_flush_task):
        logger = AzureSentinelLogger(
            dcr_immutable_id="dcr-test123456789",
            stream_name="Custom-LiteLLM-Standard",
            audit_stream_name="Custom-LiteLLM-Audit",
            endpoint="https://test-dce.eastus-1.ingest.monitor.azure.com",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

    standard_payload = StandardLoggingPayload(
        id="standard-123",
        call_type="completion",
        model="gpt-3.5-turbo",
        status="success",
        messages=[{"role": "user", "content": "Hello"}],
        response={"choices": [{"message": {"content": "Hi"}}]},
    )
    audit_log = StandardAuditLogPayload(
        id="audit-123",
        updated_at="2026-05-06T04:39:00+00:00",
        changed_by="user-1",
        changed_by_api_key="sk-test",
        action="created",
        table_name="LiteLLM_TeamTable",
        object_id="team-1",
        before_value=None,
        updated_values='{"team_alias": "sentinel-demo"}',
    )

    logger.log_queue.append(standard_payload)
    await logger.async_log_audit_log_event(audit_log)

    mock_token_response = MagicMock()
    mock_token_response.status_code = 200
    mock_token_response.json = MagicMock(
        return_value={
            "access_token": "test-bearer-token",
            "expires_in": 3600,
        }
    )
    mock_token_response.text = "Success"

    mock_api_response = MagicMock()
    mock_api_response.status_code = 204
    mock_api_response.text = "Success"

    async def mock_post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            return mock_token_response
        return mock_api_response

    logger.async_httpx_client.post = AsyncMock(side_effect=mock_post)

    await logger.flush_queue()

    ingestion_calls = [
        call
        for call in logger.async_httpx_client.post.call_args_list
        if "dataCollectionRules" in call.kwargs["url"]
    ]
    assert len(ingestion_calls) == 2

    standard_call, audit_call = ingestion_calls
    assert "Custom-LiteLLM-Standard" in standard_call.kwargs["url"]
    assert json.loads(standard_call.kwargs["data"].decode("utf-8")) == [
        standard_payload
    ]
    assert "Custom-LiteLLM-Audit" in audit_call.kwargs["url"]
    assert json.loads(audit_call.kwargs["data"].decode("utf-8")) == [audit_log]
