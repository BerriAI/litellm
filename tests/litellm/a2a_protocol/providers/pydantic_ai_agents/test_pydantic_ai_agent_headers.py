"""
Tests for Pydantic AI agents header forwarding via agent_extra_headers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.a2a_protocol.providers.pydantic_ai_agents.transformation import (
    PydanticAITransformation,
)


def _build_mock_client(response_payload):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=response_payload)

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.mark.asyncio
async def test_send_non_streaming_request_forwards_agent_extra_headers():
    """agent_extra_headers should be merged into the outbound HTTP request headers."""
    completed_payload = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "result": {
            "id": "task-1",
            "kind": "task",
            "status": {"state": "completed"},
            "history": [
                {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "msg-1",
                }
            ],
            "artifacts": [],
        },
    }
    mock_client = _build_mock_client(completed_payload)

    with patch(
        "litellm.a2a_protocol.providers.pydantic_ai_agents.transformation.get_async_httpx_client",
        return_value=mock_client,
    ):
        await PydanticAITransformation.send_non_streaming_request(
            api_base="http://example.test",
            request_id="req-1",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "messageId": "msg-user-1",
                }
            },
            agent_extra_headers={
                "x-tenant-id": "acme",
                "authorization": "Bearer caller-supplied",
            },
        )

    assert mock_client.post.await_count == 1
    sent_headers = mock_client.post.await_args.kwargs["headers"]
    assert sent_headers["x-tenant-id"] == "acme"
    assert sent_headers["authorization"] == "Bearer caller-supplied"
    assert sent_headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_send_non_streaming_request_without_headers_preserves_content_type():
    """When no agent_extra_headers are passed, behavior is unchanged."""
    completed_payload = {
        "jsonrpc": "2.0",
        "id": "req-2",
        "result": {
            "id": "task-2",
            "kind": "task",
            "status": {"state": "completed"},
            "history": [],
            "artifacts": [
                {
                    "artifactId": "a-1",
                    "parts": [{"kind": "text", "text": "ok"}],
                }
            ],
        },
    }
    mock_client = _build_mock_client(completed_payload)

    with patch(
        "litellm.a2a_protocol.providers.pydantic_ai_agents.transformation.get_async_httpx_client",
        return_value=mock_client,
    ):
        await PydanticAITransformation.send_non_streaming_request(
            api_base="http://example.test",
            request_id="req-2",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "messageId": "msg-user-2",
                }
            },
        )

    sent_headers = mock_client.post.await_args.kwargs["headers"]
    assert sent_headers == {"Content-Type": "application/json"}


@pytest.mark.asyncio
async def test_content_type_is_preserved_when_caller_tries_to_override():
    """A caller-supplied Content-Type must not displace application/json."""
    completed_payload = {
        "jsonrpc": "2.0",
        "id": "req-3",
        "result": {
            "id": "task-3",
            "kind": "task",
            "status": {"state": "completed"},
            "history": [],
            "artifacts": [
                {
                    "artifactId": "a-2",
                    "parts": [{"kind": "text", "text": "ok"}],
                }
            ],
        },
    }
    mock_client = _build_mock_client(completed_payload)

    with patch(
        "litellm.a2a_protocol.providers.pydantic_ai_agents.transformation.get_async_httpx_client",
        return_value=mock_client,
    ):
        await PydanticAITransformation.send_non_streaming_request(
            api_base="http://example.test",
            request_id="req-3",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "messageId": "msg-user-3",
                }
            },
            agent_extra_headers={"Content-Type": "text/plain"},
        )

    sent_headers = mock_client.post.await_args.kwargs["headers"]
    assert sent_headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_provider_config_threads_agent_extra_headers():
    """End-to-end: PydanticAIProviderConfig forwards agent_extra_headers down the stack."""
    from litellm.a2a_protocol.providers.pydantic_ai_agents.config import (
        PydanticAIProviderConfig,
    )

    completed_payload = {
        "jsonrpc": "2.0",
        "id": "req-4",
        "result": {
            "id": "task-4",
            "kind": "task",
            "status": {"state": "completed"},
            "history": [],
            "artifacts": [
                {
                    "artifactId": "a-3",
                    "parts": [{"kind": "text", "text": "ok"}],
                }
            ],
        },
    }
    mock_client = _build_mock_client(completed_payload)

    with patch(
        "litellm.a2a_protocol.providers.pydantic_ai_agents.transformation.get_async_httpx_client",
        return_value=mock_client,
    ):
        await PydanticAIProviderConfig().handle_non_streaming(
            request_id="req-4",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "messageId": "msg-user-4",
                }
            },
            api_base="http://example.test",
            agent_extra_headers={"x-trace-id": "abc-123"},
        )

    sent_headers = mock_client.post.await_args.kwargs["headers"]
    assert sent_headers["x-trace-id"] == "abc-123"
    assert sent_headers["Content-Type"] == "application/json"
