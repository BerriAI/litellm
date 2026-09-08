"""
Test Azure Sentinel logging integration
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import Request, Response
from pydantic import BaseModel, computed_field

from litellm.integrations.azure_sentinel.azure_sentinel import AzureSentinelLogger
from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError
from litellm.types.integrations.azure_sentinel import AZURE_SENTINEL_MAX_PAYLOAD_SIZE_BYTES
from litellm.types.utils import StandardAuditLogPayload, StandardLoggingPayload


def _close_periodic_flush_task(coro):
    coro.close()


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


@pytest.mark.asyncio
async def test_azure_sentinel_audit_stream_name_from_env_var(monkeypatch):
    """Audit stream resolves from AZURE_SENTINEL_AUDIT_STREAM_NAME when the string
    callback constructs the logger with no audit_stream_name argument."""
    monkeypatch.setenv("AZURE_SENTINEL_STREAM_NAME", "Custom-LiteLLM-Standard")
    monkeypatch.setenv("AZURE_SENTINEL_AUDIT_STREAM_NAME", "Custom-LiteLLM-Audit")

    with patch("asyncio.create_task", side_effect=_close_periodic_flush_task):
        logger = AzureSentinelLogger(
            dcr_immutable_id="dcr-test123456789",
            endpoint="https://test-dce.eastus-1.ingest.monitor.azure.com",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

    assert logger.audit_stream_name == "Custom-LiteLLM-Audit"
    assert "streams/Custom-LiteLLM-Audit" in logger.audit_api_endpoint
    assert "streams/Custom-LiteLLM-Standard" in logger.api_endpoint

    with patch("asyncio.create_task", side_effect=_close_periodic_flush_task):
        explicit_logger = AzureSentinelLogger(
            dcr_immutable_id="dcr-test123456789",
            endpoint="https://test-dce.eastus-1.ingest.monitor.azure.com",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
            audit_stream_name="Custom-LiteLLM-Explicit",
        )

    assert explicit_logger.audit_stream_name == "Custom-LiteLLM-Explicit"


def _build_logger(**overrides):
    kwargs = {
        "dcr_immutable_id": "dcr-test123456789",
        "endpoint": "https://test-dce.eastus-1.ingest.monitor.azure.com",
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        **overrides,
    }
    with patch("asyncio.create_task", side_effect=_close_periodic_flush_task):
        return AzureSentinelLogger(**kwargs)


@pytest.fixture
def _no_authority_host_env(monkeypatch):
    monkeypatch.delenv("AZURE_SENTINEL_AUTHORITY_HOST", raising=False)
    monkeypatch.delenv("AZURE_AUTHORITY_HOST", raising=False)


@pytest.mark.parametrize(
    "authority_host, expected_authority, expected_scope",
    [
        (None, "https://login.microsoftonline.com", "https://monitor.azure.com/.default"),
        ("https://login.microsoftonline.us", "https://login.microsoftonline.us", "https://monitor.azure.us/.default"),
        ("https://login.microsoftonline.us/", "https://login.microsoftonline.us", "https://monitor.azure.us/.default"),
        ("login.microsoftonline.us", "https://login.microsoftonline.us", "https://monitor.azure.us/.default"),
        ("https://adfs.contoso.example", "https://adfs.contoso.example", "https://monitor.azure.com/.default"),
    ],
)
def test_azure_sentinel_resolves_authority_host_and_audience_together(
    _no_authority_host_env, authority_host, expected_authority, expected_scope
):
    """Both the Entra authority and the Azure Monitor audience must follow the configured cloud.

    Moving only the authority leaves a sovereign deployment asking sovereign Entra for the
    commercial audience, which the sovereign ingestion endpoint rejects.
    """
    logger = _build_logger(**({} if authority_host is None else {"authority_host": authority_host}))

    assert logger.authority_host == expected_authority
    assert logger.oauth_scope == expected_scope


def test_azure_sentinel_authority_host_from_env_var(_no_authority_host_env, monkeypatch):
    """AZURE_AUTHORITY_HOST is the documented setting and the string callback constructs the logger
    with no arguments, so the env var alone has to move both values."""
    monkeypatch.setenv("AZURE_AUTHORITY_HOST", "https://login.microsoftonline.us")

    logger = _build_logger()

    assert logger.authority_host == "https://login.microsoftonline.us"
    assert logger.oauth_scope == "https://monitor.azure.us/.default"


@pytest.mark.asyncio
async def test_azure_sentinel_token_request_uses_sovereign_authority_and_audience(_no_authority_host_env):
    """The resolved values must reach the wire, not just the instance attributes."""
    logger = _build_logger(authority_host="https://login.microsoftonline.us")
    logger.log_queue.append(
        StandardLoggingPayload(
            id="test_id",
            call_type="completion",
            model="gpt-3.5-turbo",
            status="success",
            messages=[{"role": "user", "content": "Hello"}],
            response={"choices": [{"message": {"content": "Hi"}}]},
        )
    )

    mock_token_response = MagicMock()
    mock_token_response.status_code = 200
    mock_token_response.json = MagicMock(return_value={"access_token": "test-bearer-token", "expires_in": 3600})
    mock_token_response.text = "Success"
    mock_api_response = MagicMock()
    mock_api_response.status_code = 204
    mock_api_response.text = "Success"

    async def mock_post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            return mock_token_response
        return mock_api_response

    logger.async_httpx_client.post = AsyncMock(side_effect=mock_post)

    await logger.async_send_batch()

    token_calls = [
        call for call in logger.async_httpx_client.post.call_args_list if "oauth2/v2.0/token" in call.kwargs["url"]
    ]
    assert len(token_calls) == 1
    assert token_calls[0].kwargs["url"] == "https://login.microsoftonline.us/test-tenant-id/oauth2/v2.0/token"
    assert token_calls[0].kwargs["data"]["scope"] == "https://monitor.azure.us/.default"


def test_azure_sentinel_authority_host_prefers_the_sentinel_scoped_env_var(_no_authority_host_env, monkeypatch):
    """AZURE_AUTHORITY_HOST is shared with Azure OpenAI and the azure_storage callback, so a deployment
    whose Sentinel workspace lives in a different cloud than the rest of its Azure resources needs a
    Sentinel-scoped override. This mirrors how tenant, client id and secret already resolve."""
    monkeypatch.setenv("AZURE_AUTHORITY_HOST", "https://login.microsoftonline.com")
    monkeypatch.setenv("AZURE_SENTINEL_AUTHORITY_HOST", "https://login.microsoftonline.us")

    logger = _build_logger()

    assert logger.authority_host == "https://login.microsoftonline.us"
    assert logger.oauth_scope == "https://monitor.azure.us/.default"


def test_azure_sentinel_authority_host_argument_outranks_the_scoped_env_var(_no_authority_host_env, monkeypatch):
    """An explicit constructor argument is the most specific source and has to win, otherwise a
    deployment that exports the scoped variable silently overrides an SDK caller."""
    monkeypatch.setenv("AZURE_SENTINEL_AUTHORITY_HOST", "https://login.microsoftonline.us")

    logger = _build_logger(authority_host="https://login.microsoftonline.com")

    assert logger.authority_host == "https://login.microsoftonline.com"
    assert logger.oauth_scope == "https://monitor.azure.com/.default"


def _standard_payloads(count, filler_bytes=0):
    return [
        StandardLoggingPayload(
            id=f"standard-{i}",
            call_type="completion",
            model="gpt-3.5-turbo",
            status="success",
            messages=[{"role": "user", "content": "x" * filler_bytes}],
            response={"choices": [{"message": {"content": "Hi"}}]},
        )
        for i in range(count)
    ]


def _audit_payloads(count, filler_bytes=0):
    return [
        StandardAuditLogPayload(
            id=f"audit-{i}",
            updated_at="2026-05-06T04:39:00+00:00",
            changed_by="user-1",
            changed_by_api_key="sk-test",
            action="created",
            table_name="LiteLLM_TeamTable",
            object_id="team-1",
            before_value=None,
            updated_values=json.dumps({"team_alias": "x" * filler_bytes}),
        )
        for i in range(count)
    ]


QUEUE_CASES = [
    pytest.param("log_queue", "async_send_batch", _standard_payloads, id="standard"),
    pytest.param("audit_log_queue", "async_send_audit_batch", _audit_payloads, id="audit"),
]


def _token_response():
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(return_value={"access_token": "test-bearer-token", "expires_in": 3600})
    response.text = "Success"
    return response


def _install_ingestion(logger, on_ingest):
    """Route the OAuth call to a canned token and every ingestion call to `on_ingest(body_bytes)`."""

    async def _post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            return _token_response()
        return await on_ingest(kwargs["data"])

    logger.async_httpx_client.post = AsyncMock(side_effect=_post)


def _accepted():
    return Response(204, request=Request("POST", "https://example.com"), text="")


def _too_large(*, raised):
    request = Request("POST", "https://example.com")
    response = Response(413, request=request, text="Payload Too Large")
    if raised:
        raise MaskedHTTPStatusError(httpx.HTTPStatusError("413", request=request, response=response))
    return response


def _rejected(status_code, *, raised):
    """litellm's http handler calls raise_for_status, so a real rejection arrives raised, not returned."""
    request = Request("POST", "https://example.com")
    response = Response(status_code, request=request, text=f"rejected with {status_code}")
    if raised:
        raise MaskedHTTPStatusError(httpx.HTTPStatusError(str(status_code), request=request, response=response))
    return response


def _awaiting_retry(logger, queue_attr):
    return getattr(logger, "logs_awaiting_retry" if queue_attr == "log_queue" else "audit_logs_awaiting_retry")


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_splits_a_batch_that_would_exceed_the_ingestion_cap(
    queue_attr, send_method, build_payloads
):
    """Azure Monitor rejects a body over 1MB uncompressed, so an oversize batch has to be split
    before it is sent instead of being posted whole and lost."""
    logger = _build_logger()
    records = build_payloads(4, filler_bytes=400_000)
    setattr(logger, queue_attr, list(records))

    sent_bodies = []

    async def _on_ingest(data):
        sent_bodies.append(data)
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert len(sent_bodies) > 1
    assert all(len(body) <= AZURE_SENTINEL_MAX_PAYLOAD_SIZE_BYTES for body in sent_bodies)
    delivered = [record["id"] for body in sent_bodies for record in json.loads(body.decode("utf-8"))]
    assert delivered == [record["id"] for record in records]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raised", [True, False], ids=["raised", "returned"])
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_halves_the_batch_on_413(queue_attr, send_method, build_payloads, raised):
    """A 413 the size estimate did not predict must halve the batch and retry, not drop it.

    litellm's http handler raises MaskedHTTPStatusError on a 4xx, so the raised path is the one
    a real Azure Monitor 413 takes, and both are covered here.
    """
    logger = _build_logger()
    records = build_payloads(4)
    setattr(logger, queue_attr, list(records))

    delivered = []

    async def _on_ingest(data):
        body = json.loads(data.decode("utf-8"))
        if len(body) > 1:
            return _too_large(raised=raised)
        delivered.extend(record["id"] for record in body)
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert delivered == [record["id"] for record in records]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_drops_only_the_lone_record_that_still_413s(queue_attr, send_method, build_payloads):
    """One undeliverable record must not take its siblings down with it or wedge the queue."""
    logger = _build_logger()
    records = build_payloads(4)
    poison = records[2]["id"]
    setattr(logger, queue_attr, list(records))

    delivered = []

    async def _on_ingest(data):
        body = json.loads(data.decode("utf-8"))
        if any(record["id"] == poison for record in body):
            return _too_large(raised=True)
        delivered.extend(record["id"] for record in body)
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await asyncio.wait_for(getattr(logger, send_method)(), timeout=10)

    assert delivered == [record["id"] for record in records if record["id"] != poison]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_requeues_only_what_a_transient_failure_left_undelivered(
    queue_attr, send_method, build_payloads
):
    """Records Azure Monitor already accepted must not be sent twice, and the rest must survive
    for the next flush instead of being cleared."""
    logger = _build_logger()
    records = build_payloads(4)
    setattr(logger, queue_attr, list(records))

    delivered = []

    async def _on_ingest(data):
        body = json.loads(data.decode("utf-8"))
        if len(body) > 2:
            return _too_large(raised=True)
        if any(record["id"] == records[2]["id"] for record in body):
            raise httpx.ConnectError("connection reset")
        delivered.extend(record["id"] for record in body)
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert delivered == [records[0]["id"], records[1]["id"]]
    assert getattr(logger, queue_attr) == records[2:]


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_requeues_the_batch_on_a_non_success_status(queue_attr, send_method, build_payloads):
    """A 500 from ingestion is retryable, so the batch has to stay queued."""
    logger = _build_logger()
    records = build_payloads(3)
    setattr(logger, queue_attr, list(records))

    async def _on_ingest(data):
        return Response(500, request=Request("POST", "https://example.com"), text="Internal Server Error")

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert getattr(logger, queue_attr) == records


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_requeues_the_batch_when_the_oauth_token_call_fails(
    queue_attr, send_method, build_payloads
):
    """Losing the token is transient, so the batch must not be dropped on the way to the wire."""
    logger = _build_logger()
    records = build_payloads(2)
    setattr(logger, queue_attr, list(records))

    ingestion_calls = []

    async def _post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            failed = MagicMock()
            failed.status_code = 401
            failed.text = "Unauthorized"
            return failed
        ingestion_calls.append(kwargs["url"])
        return _accepted()

    logger.async_httpx_client.post = AsyncMock(side_effect=_post)

    await getattr(logger, send_method)()

    assert ingestion_calls == []
    assert getattr(logger, queue_attr) == records


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_caps_the_retry_queue_at_max_queue_size(queue_attr, send_method, build_payloads):
    """Retrying forever against an unreachable workspace must not grow the queue without bound,
    so the oldest records go once the queue is over its limit."""
    logger = _build_logger(max_queue_size=3)
    records = build_payloads(4)
    setattr(logger, queue_attr, list(records))

    async def _on_ingest(data):
        raise httpx.ConnectError("connection reset")

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert getattr(logger, queue_attr) == records[1:]


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_keeps_records_queued_during_a_send(queue_attr, send_method, build_payloads):
    """The queue is detached before sending, so a record logged mid-flush is kept and lands behind
    anything the failed send hands back."""
    logger = _build_logger()
    records = build_payloads(2)
    late_record = build_payloads(1)[0]
    late_record["id"] = "logged-during-send"
    setattr(logger, queue_attr, list(records))

    async def _on_ingest(data):
        getattr(logger, queue_attr).append(late_record)
        raise httpx.ConnectError("connection reset")

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert getattr(logger, queue_attr) == [*records, late_record]


def _poison(record):
    """A mixed-type set makes safe_dumps raise TypeError while sorting it, so the record can never be serialized."""
    field = "messages" if "messages" in record else "updated_values"
    record[field] = {1, "a"}
    return record


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_drops_only_the_record_that_cannot_be_serialized(queue_attr, send_method, build_payloads):
    """A record that raises during serialization used to escape the send, which killed the periodic
    flush task for good and lost the already-detached batch with it. It has to be isolated and
    dropped alone, with the flush completing normally."""
    logger = _build_logger()
    records = build_payloads(4)
    poison = _poison(records[2])["id"]
    setattr(logger, queue_attr, list(records))

    delivered = []

    async def _on_ingest(data):
        delivered.extend(record["id"] for record in json.loads(data.decode("utf-8")))
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await asyncio.wait_for(logger.flush_queue(), timeout=10)

    assert delivered == [record["id"] for record in records if record["id"] != poison]
    assert getattr(logger, queue_attr) == []


async def _log(logger, queue_attr, record):
    if queue_attr == "log_queue":
        await logger.async_log_success_event({"standard_logging_object": record}, None, None, None)
        return
    await logger.async_log_audit_log_event(record)


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_retries_on_the_flush_timer_not_on_every_record_while_the_destination_is_down(
    queue_attr, send_method, build_payloads
):
    """Requeued records keep the queue at or over batch_size, so without a guard every new record
    re-sent the whole growing queue. While a retry is pending only the periodic flush may send, and
    a successful flush hands the trigger back to the batch size."""
    logger = _build_logger(batch_size=3)
    records = build_payloads(11)

    attempts = []
    destination_down = True

    async def _on_ingest(data):
        attempts.append([record["id"] for record in json.loads(data.decode("utf-8"))])
        if destination_down:
            raise httpx.ConnectError("connection reset")
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    for record in records[:8]:
        await _log(logger, queue_attr, record)

    assert attempts == [[record["id"] for record in records[:3]]]
    assert getattr(logger, queue_attr) == records[:8]

    destination_down = False
    await logger.flush_queue()
    for record in records[8:]:
        await _log(logger, queue_attr, record)

    assert [record_id for attempt in attempts[1:-1] for record_id in attempt] == [record["id"] for record in records[:8]]
    assert all(len(attempt) <= 3 for attempt in attempts[1:-1])
    assert attempts[-1] == [record["id"] for record in records[8:]]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_threshold_send_waits_for_an_in_flight_timer_flush(
    queue_attr, send_method, build_payloads
):
    """A batch-size send that overlapped the periodic flush could finish after it and requeue its
    newer records in front of the older ones, so the max_queue_size trim would then drop the
    newest records instead of the oldest. Both paths have to take the flush lock, and a waiter
    that gets the lock after a failed flush stands down instead of resending the whole queue."""
    logger = _build_logger(batch_size=2)
    records = build_payloads(4)
    setattr(logger, queue_attr, list(records[:2]))

    attempts = []
    timer_send_started = asyncio.Event()
    release_timer_send = asyncio.Event()

    async def _on_ingest(data):
        attempts.append([record["id"] for record in json.loads(data.decode("utf-8"))])
        if len(attempts) == 1:
            timer_send_started.set()
            await release_timer_send.wait()
        raise httpx.ConnectError("connection reset")

    _install_ingestion(logger, _on_ingest)

    timer_flush = asyncio.create_task(logger.flush_queue())
    await asyncio.wait_for(timer_send_started.wait(), timeout=10)
    await _log(logger, queue_attr, records[2])
    threshold_send = asyncio.create_task(_log(logger, queue_attr, records[3]))
    await asyncio.sleep(0)
    release_timer_send.set()
    await asyncio.wait_for(timer_flush, timeout=10)
    await asyncio.wait_for(threshold_send, timeout=10)

    assert attempts == [[record["id"] for record in records[:2]]]
    assert getattr(logger, queue_attr) == records


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_concurrent_threshold_sends_collapse_into_one_attempt_while_the_destination_is_down(
    queue_attr, send_method, build_payloads
):
    """Records logged while a threshold send is blocked on the wire all see the retry flag still
    unset and queue up on the flush lock. Each waiter has to recheck under the lock, or every one
    of them resends the growing queue as soon as the first attempt fails."""
    logger = _build_logger(batch_size=2)
    records = build_payloads(6)

    attempts = []
    first_send_started = asyncio.Event()
    release_first_send = asyncio.Event()
    destination_down = True

    async def _on_ingest(data):
        attempts.append([record["id"] for record in json.loads(data.decode("utf-8"))])
        if len(attempts) == 1:
            first_send_started.set()
            await release_first_send.wait()
        if destination_down:
            raise httpx.ConnectError("connection reset")
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await _log(logger, queue_attr, records[0])
    first_send = asyncio.create_task(_log(logger, queue_attr, records[1]))
    await asyncio.wait_for(first_send_started.wait(), timeout=10)
    waiters = [asyncio.create_task(_log(logger, queue_attr, record)) for record in records[2:]]
    await asyncio.sleep(0)
    release_first_send.set()
    await asyncio.wait_for(asyncio.gather(first_send, *waiters), timeout=10)

    assert attempts == [[record["id"] for record in records[:2]]]
    assert getattr(logger, queue_attr) == records

    destination_down = False
    await logger.flush_queue()

    assert [record_id for attempt in attempts[1:] for record_id in attempt] == [record["id"] for record in records]
    assert all(len(attempt) <= 2 for attempt in attempts[1:])
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_requeues_a_cancelled_send(
    queue_attr, send_method, build_payloads
):
    """Cancellation after detaching a batch must preserve the detached records for a later flush."""
    logger = _build_logger()
    records = build_payloads(2)
    setattr(logger, queue_attr, list(records))

    async def _on_ingest(data):
        raise asyncio.CancelledError

    _install_ingestion(logger, _on_ingest)

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await getattr(logger, send_method)()

    assert type(excinfo.value) is asyncio.CancelledError
    assert getattr(logger, queue_attr) == records
    assert _awaiting_retry(logger, queue_attr)


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_requeues_a_send_cancelled_before_it_reached_the_wire(
    queue_attr, send_method, build_payloads
):
    """Cancellation can land on the token call, before any record was sent, and the detached batch
    has to survive that too."""
    logger = _build_logger()
    records = build_payloads(2)
    setattr(logger, queue_attr, list(records))
    logger.async_httpx_client.post = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await getattr(logger, send_method)()

    assert type(excinfo.value) is asyncio.CancelledError
    assert getattr(logger, queue_attr) == records
    assert _awaiting_retry(logger, queue_attr)


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_does_not_resend_the_half_delivered_before_a_cancelled_split(
    queue_attr, send_method, build_payloads
):
    """A batch over the size cap goes out in pieces, so a cancellation partway through must requeue
    only the pieces the destination never accepted, or the accepted ones land in Sentinel twice."""
    logger = _build_logger()
    records = build_payloads(8, filler_bytes=400_000)
    setattr(logger, queue_attr, list(records))

    attempts = []
    cancel_after_the_first_piece = True

    async def _on_ingest(data):
        attempts.append([record["id"] for record in json.loads(data.decode("utf-8"))])
        if cancel_after_the_first_piece and len(attempts) > 1:
            raise asyncio.CancelledError
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await getattr(logger, send_method)()

    assert type(excinfo.value) is asyncio.CancelledError
    assert attempts == [[record["id"] for record in records[:2]], [record["id"] for record in records[2:4]]]
    assert getattr(logger, queue_attr) == records[2:]

    cancel_after_the_first_piece = False
    await logger.flush_queue()

    assert [record_id for attempt in attempts[2:] for record_id in attempt] == [record["id"] for record in records[2:]]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_threshold_waiter_does_not_send_a_sub_batch_after_success(
    queue_attr, send_method, build_payloads
):
    """A successful threshold send can leave one record behind, so a waiter must not send it
    before the next record completes a batch."""
    logger = _build_logger(batch_size=2)
    records = build_payloads(3)

    attempts = []
    first_send_started = asyncio.Event()
    release_first_send = asyncio.Event()

    async def _on_ingest(data):
        attempts.append([record["id"] for record in json.loads(data.decode("utf-8"))])
        if len(attempts) == 1:
            first_send_started.set()
            await release_first_send.wait()
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await _log(logger, queue_attr, records[0])
    first_send = asyncio.create_task(_log(logger, queue_attr, records[1]))
    await asyncio.wait_for(first_send_started.wait(), timeout=10)
    waiter = asyncio.create_task(_log(logger, queue_attr, records[2]))
    await asyncio.sleep(0)
    release_first_send.set()
    await asyncio.wait_for(asyncio.gather(first_send, waiter), timeout=10)

    assert attempts == [[record["id"] for record in records[:2]]]
    assert getattr(logger, queue_attr) == [records[2]]


@pytest.mark.asyncio
async def test_azure_sentinel_threshold_send_only_sends_the_queue_that_crossed_the_threshold():
    """The standard and audit queues retry independently: crossing the audit threshold must not
    resend standard records that are waiting for the periodic flush."""
    logger = _build_logger(batch_size=2)
    standard_records = _standard_payloads(2)
    audit_records = _audit_payloads(2)
    logger.log_queue = list(standard_records)
    logger.logs_awaiting_retry = True

    attempts = []

    async def _on_ingest(data):
        attempts.append([record["id"] for record in json.loads(data.decode("utf-8"))])
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    for record in audit_records:
        await logger.async_log_audit_log_event(record)

    assert attempts == [[record["id"] for record in audit_records]]
    assert logger.audit_log_queue == []
    assert logger.log_queue == standard_records


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_keeps_the_batch_when_ingestion_raises_a_retryable_status(
    queue_attr, send_method, build_payloads, status_code
):
    """A 5xx, a timeout or a throttle can clear on the next flush, so the whole batch stays queued
    and the awaiting-retry flag hands the send back to the timer."""
    logger = _build_logger()
    records = build_payloads(3)
    setattr(logger, queue_attr, list(records))

    async def _on_ingest(data):
        return _rejected(status_code, raised=True)

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert getattr(logger, queue_attr) == records
    assert _awaiting_retry(logger, queue_attr)


@pytest.mark.asyncio
@pytest.mark.parametrize("raised", [True, False], ids=["raised", "returned"])
@pytest.mark.parametrize("status_code", [400, 403, 404])
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_drops_the_batch_when_ingestion_rejects_it_for_good(
    queue_attr, send_method, build_payloads, status_code, raised
):
    """A permanent 4xx is dropped, the flag is cleared and the next records go out on their own."""
    logger = _build_logger(batch_size=2)
    rejected_records = build_payloads(2)
    later_records = build_payloads(4)[2:]
    setattr(logger, queue_attr, list(rejected_records))

    delivered = []
    destination_rejects = True

    async def _on_ingest(data):
        if destination_rejects:
            return _rejected(status_code, raised=raised)
        delivered.extend(record["id"] for record in json.loads(data.decode("utf-8")))
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert getattr(logger, queue_attr) == []
    assert not _awaiting_retry(logger, queue_attr)

    destination_rejects = False
    for record in later_records:
        await _log(logger, queue_attr, record)

    assert delivered == [record["id"] for record in later_records]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_keeps_the_whole_batch_when_the_first_piece_of_a_split_fails(
    queue_attr, send_method, build_payloads
):
    """When the first half of a split hits a retryable error the untried second half must be kept
    too, in the original order, instead of being sent ahead of records that are still pending."""
    logger = _build_logger()
    records = build_payloads(4)
    setattr(logger, queue_attr, list(records))

    attempts = []

    async def _on_ingest(data):
        body = json.loads(data.decode("utf-8"))
        attempts.append([record["id"] for record in body])
        if len(body) > 2:
            return _too_large(raised=True)
        return _rejected(503, raised=True)

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert attempts == [[record["id"] for record in records], [record["id"] for record in records[:2]]]
    assert getattr(logger, queue_attr) == records
    assert _awaiting_retry(logger, queue_attr)


class _RaisesWhileDumping(BaseModel):
    @computed_field
    @property
    def rendered(self) -> str:
        raise RuntimeError("this field cannot be rendered")


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_drops_only_the_record_whose_serialization_raises_an_unexpected_error(
    queue_attr, send_method, build_payloads
):
    """Serialization can fail with any exception class, not just TypeError or ValueError, because
    safe_dumps hands pydantic models to model_dump. A record that raises anything has to be isolated
    and dropped alone, or the flush dies with the whole batch."""
    logger = _build_logger()
    records = build_payloads(4)
    poison = records[1]
    poison["messages" if "messages" in poison else "updated_values"] = _RaisesWhileDumping()
    setattr(logger, queue_attr, list(records))

    delivered = []

    async def _on_ingest(data):
        delivered.extend(record["id"] for record in json.loads(data.decode("utf-8")))
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await asyncio.wait_for(logger.flush_queue(), timeout=10)

    assert delivered == [record["id"] for record in records if record is not poison]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_send_cancelled_by_a_timeout_surfaces_as_a_timeout(
    queue_attr, send_method, build_payloads
):
    """The logging worker bounds each flush with asyncio.wait_for, which on Python 3.12 only turns
    an exact CancelledError into TimeoutError. A subclass carrying the undelivered records would
    escape the worker as an unhandled error, so the send must re-raise the plain class."""
    logger = _build_logger()
    records = build_payloads(2)
    setattr(logger, queue_attr, list(records))

    async def _on_ingest(data):
        await asyncio.sleep(60)
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(getattr(logger, send_method)(), timeout=0.05)

    assert getattr(logger, queue_attr) == records
    assert _awaiting_retry(logger, queue_attr)


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_never_sends_more_than_batch_size_records_in_one_request(
    queue_attr, send_method, build_payloads
):
    """A recovery flush can find far more than batch_size records queued. Splitting on the count
    first keeps each request at the configured size and bounds how much of the queue is serialized
    just to measure it."""
    logger = _build_logger(batch_size=2)
    records = build_payloads(5)
    setattr(logger, queue_attr, list(records))

    attempts = []

    async def _on_ingest(data):
        attempts.append([record["id"] for record in json.loads(data.decode("utf-8"))])
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert attempts == [
        [records[0]["id"], records[1]["id"]],
        [records[2]["id"]],
        [records[3]["id"], records[4]["id"]],
    ]
    assert getattr(logger, queue_attr) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code, expected_queue",
    [pytest.param(503, "kept", id="503-kept"), pytest.param(401, "dropped", id="401-dropped")],
)
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_oauth_rejection_follows_the_same_retry_rule_as_ingestion(
    queue_attr, send_method, build_payloads, status_code, expected_queue
):
    """The token endpoint raises through the same http handler as ingestion. A 5xx there is
    transient and keeps the batch, a 401 means the client secret is wrong and would fail every
    retry, so the batch is dropped instead of wedging the queue."""
    logger = _build_logger()
    records = build_payloads(2)
    setattr(logger, queue_attr, list(records))

    ingestion_calls = []

    async def _post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            return _rejected(status_code, raised=True)
        ingestion_calls.append(kwargs["url"])
        return _accepted()

    logger.async_httpx_client.post = AsyncMock(side_effect=_post)

    await getattr(logger, send_method)()

    assert ingestion_calls == []
    assert getattr(logger, queue_attr) == (records if expected_queue == "kept" else [])
    assert _awaiting_retry(logger, queue_attr) is (expected_queue == "kept")


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_attr, send_method, build_payloads", QUEUE_CASES)
async def test_azure_sentinel_does_not_stay_in_retry_mode_when_the_queue_cap_trims_everything(
    queue_attr, send_method, build_payloads
):
    """With max_queue_size at 0 the cap drops every requeued record, so there is nothing for the
    timer to retry. The flag must follow the retained queue, or every later threshold send is
    skipped until the timer happens to fire."""
    logger = _build_logger(batch_size=2, max_queue_size=0)
    lost_records = build_payloads(2)
    later_records = build_payloads(4)[2:]
    setattr(logger, queue_attr, list(lost_records))

    delivered = []
    destination_down = True

    async def _on_ingest(data):
        if destination_down:
            raise httpx.ConnectError("connection reset")
        delivered.extend(record["id"] for record in json.loads(data.decode("utf-8")))
        return _accepted()

    _install_ingestion(logger, _on_ingest)

    await getattr(logger, send_method)()

    assert getattr(logger, queue_attr) == []
    assert not _awaiting_retry(logger, queue_attr)

    destination_down = False
    for record in later_records:
        await _log(logger, queue_attr, record)

    assert delivered == [record["id"] for record in later_records]
