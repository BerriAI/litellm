import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.integrations.microsoft_purview.microsoft_purview import (
    MicrosoftPurviewLogger,
)


@pytest.fixture
def valid_env_vars(monkeypatch):
    monkeypatch.setenv("MICROSOFT_PURVIEW_TENANT_ID", "test-tenant")
    monkeypatch.setenv("MICROSOFT_PURVIEW_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("MICROSOFT_PURVIEW_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("MICROSOFT_PURVIEW_APP_NAME", "test-app")
    monkeypatch.setenv("MICROSOFT_PURVIEW_APP_VERSION", "1.0.0")
    monkeypatch.setenv("MICROSOFT_PURVIEW_APP_ID", "test-app-id")


@pytest.mark.asyncio
async def test_init_with_all_env_vars(valid_env_vars):
    logger = MicrosoftPurviewLogger()
    assert logger.tenant_id == "test-tenant"
    assert logger.client_id == "test-client-id"
    assert logger.client_secret == "test-secret"
    assert logger.app_name == "test-app"
    assert logger.app_version == "1.0.0"
    assert logger.app_id == "test-app-id"
    assert logger.oauth_scope == "https://graph.microsoft.com/.default"


def test_init_missing_tenant_id_raises(monkeypatch):
    monkeypatch.delenv("MICROSOFT_PURVIEW_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.setenv("MICROSOFT_PURVIEW_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("MICROSOFT_PURVIEW_CLIENT_SECRET", "test-secret")
    with pytest.raises(
        ValueError,
        match="MICROSOFT_PURVIEW_TENANT_ID is required to use Microsoft Purview integration",
    ):
        MicrosoftPurviewLogger()


def test_init_missing_client_id_raises(monkeypatch):
    monkeypatch.setenv("MICROSOFT_PURVIEW_TENANT_ID", "test-tenant")
    monkeypatch.delenv("MICROSOFT_PURVIEW_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.setenv("MICROSOFT_PURVIEW_CLIENT_SECRET", "test-secret")
    with pytest.raises(
        ValueError,
        match="MICROSOFT_PURVIEW_CLIENT_ID is required to use Microsoft Purview integration",
    ):
        MicrosoftPurviewLogger()


def test_init_missing_client_secret_raises(monkeypatch):
    monkeypatch.setenv("MICROSOFT_PURVIEW_TENANT_ID", "test-tenant")
    monkeypatch.setenv("MICROSOFT_PURVIEW_CLIENT_ID", "test-client-id")
    monkeypatch.delenv("MICROSOFT_PURVIEW_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    with pytest.raises(
        ValueError,
        match="MICROSOFT_PURVIEW_CLIENT_SECRET is required to use Microsoft Purview integration",
    ):
        MicrosoftPurviewLogger()


@pytest.mark.asyncio
async def test_extract_user_id_from_metadata(valid_env_vars):
    logger = MicrosoftPurviewLogger(default_user_id="default-user")

    # Priority 1: metadata.user_api_key_user_id
    payload1 = {"metadata": {"user_api_key_user_id": "test-user-1"}}
    assert logger._extract_user_id(payload1) == "test-user-1"

    # Priority 2: end_user
    payload2 = {"end_user": "test-user-2"}
    assert logger._extract_user_id(payload2) == "test-user-2"

    # Priority 3: fallback
    payload3 = {}
    assert logger._extract_user_id(payload3) == "default-user"


@pytest.mark.asyncio
async def test_serialize_messages(valid_env_vars):
    logger = MicrosoftPurviewLogger()

    # List format
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi"},
    ]
    result = logger._serialize_messages(messages)
    assert "[user]: hello world" in result
    assert "[assistant]: hi" in result

    # String format
    assert logger._serialize_messages("hello world") == "hello world"


@pytest.mark.asyncio
async def test_extract_response_text(valid_env_vars):
    logger = MicrosoftPurviewLogger()

    # Standard format
    payload = {
        "response": {"choices": [{"message": {"content": "this is a response"}}]}
    }
    assert logger._extract_response_text(payload) == "this is a response"

    # String format
    payload_str = {"response": "just a string response"}
    assert logger._extract_response_text(payload_str) == "just a string response"


@pytest.mark.asyncio
async def test_build_process_content_request(valid_env_vars):
    logger = MicrosoftPurviewLogger()

    payload = {
        "trace_id": "test-trace-123",
        "startTime": 1700000000,
        "endTime": 1700000010,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "response": {"choices": [{"message": {"content": "4"}}]},
    }

    req = logger._build_process_content_request(payload)

    assert "contentToProcess" in req
    content_to_process = req["contentToProcess"]

    assert "contentEntries" in content_to_process
    assert len(content_to_process["contentEntries"]) == 2

    prompt_entry = content_to_process["contentEntries"][0]
    assert prompt_entry["identifier"] == "test-trace-123-prompt"
    assert prompt_entry["name"] == "LLM Prompt"
    assert prompt_entry["content"]["data"] == "[user]: What is 2+2?"
    assert prompt_entry["sequenceNumber"] == 0
    assert prompt_entry["correlationId"] == "test-trace-123"

    response_entry = content_to_process["contentEntries"][1]
    assert response_entry["identifier"] == "test-trace-123-response"
    assert response_entry["name"] == "LLM Response"
    assert response_entry["content"]["data"] == "4"
    assert response_entry["sequenceNumber"] == 1
    assert response_entry["correlationId"] == "test-trace-123"

    # Verify metadata
    assert content_to_process["integratedAppMetadata"]["name"] == "test-app"
    assert (
        content_to_process["protectedAppMetadata"]["applicationLocation"]["value"]
        == "test-app-id"
    )


@pytest.mark.asyncio
async def test_async_log_success_event_queues(valid_env_vars):
    logger = MicrosoftPurviewLogger(batch_size=5)

    kwargs = {"model": "gpt-4", "standard_logging_object": {"trace_id": "1"}}

    await logger.async_log_success_event(kwargs, None, None, None)
    assert len(logger.log_queue) == 1
    assert logger.log_queue[0]["trace_id"] == "1"


@pytest.mark.asyncio
async def test_async_log_failure_event_queues(valid_env_vars):
    logger = MicrosoftPurviewLogger(batch_size=5)

    kwargs = {"model": "gpt-4", "standard_logging_object": {"trace_id": "2"}}

    await logger.async_log_failure_event(kwargs, None, None, None)
    assert len(logger.log_queue) == 1
    assert logger.log_queue[0]["trace_id"] == "2"


@pytest.mark.asyncio
async def test_async_send_batch_success(valid_env_vars, monkeypatch):
    logger = MicrosoftPurviewLogger()

    # Add dummy payload to queue
    payload = {
        "trace_id": "test-trace-123",
        "metadata": {"user_api_key_user_id": "user-A"},
        "messages": ["test message"],
    }
    logger.log_queue.append(payload)

    # Mock token
    async def mock_get_token():
        return "fake-token"

    logger._get_oauth_token = mock_get_token

    # Mock the http client
    mock_post = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    logger.async_httpx_client.post = mock_post

    await logger.async_send_batch()

    # Check that HTTP post was called correctly
    assert mock_post.called
    assert len(logger.log_queue) == 0
    call_kwargs = mock_post.call_args[1]
    assert "url" in call_kwargs
    assert "users/user-A/dataSecurityAndGovernance/processContent" in call_kwargs["url"]
    assert "Bearer fake-token" in call_kwargs["headers"]["Authorization"]


@pytest.mark.asyncio
async def test_oauth_token_caching(valid_env_vars):
    logger = MicrosoftPurviewLogger()

    import time

    logger.oauth_token = "cached-token"
    logger.oauth_token_expires_at = time.time() + 3600

    mock_post = AsyncMock()
    logger.async_httpx_client.post = mock_post

    token = await logger._get_oauth_token()

    assert token == "cached-token"
    assert not mock_post.called


@pytest.mark.asyncio
async def test_oauth_token_refresh(valid_env_vars):
    logger = MicrosoftPurviewLogger()

    # Expired token
    import time

    logger.oauth_token = "expired-token"
    logger.oauth_token_expires_at = time.time() - 3600

    # Mock token response
    mock_post = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "new-token", "expires_in": 3600}
    mock_post.return_value = mock_response
    logger.async_httpx_client.post = mock_post

    token = await logger._get_oauth_token()

    assert token == "new-token"
    assert logger.oauth_token == "new-token"
    assert mock_post.called
    assert logger.oauth_token_expires_at > time.time()
