"""
Test A2A completion bridge streaming transformation to proper A2A format.

Tests that the completion bridge emits proper A2A streaming events:
1. Task event (kind: "task") - Initial task with status "submitted"
2. Status update (kind: "status-update") - Status "working"
3. Artifact update (kind: "artifact-update") - Content delivery
4. Status update (kind: "status-update") - Final "completed" status
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestA2AStreamingTransformation:
    """Test the A2A streaming transformation creates proper events."""

    def test_create_task_event(self):
        """Test that create_task_event produces proper A2A task event structure."""
        from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
            A2ACompletionBridgeTransformation,
            A2AStreamingContext,
        )

        input_message = {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello"}],
            "messageId": "msg-123",
        }
        ctx = A2AStreamingContext(request_id="req-456", input_message=input_message)

        event = A2ACompletionBridgeTransformation.create_task_event(ctx)

        # Validate structure
        assert event["jsonrpc"] == "2.0"
        assert event["id"] == "req-456"
        assert event["result"]["kind"] == "task"
        assert event["result"]["status"]["state"] == "submitted"
        assert "contextId" in event["result"]
        assert "id" in event["result"]  # task id
        assert "history" in event["result"]
        assert len(event["result"]["history"]) == 1
        assert event["result"]["history"][0]["role"] == "user"

    def test_create_status_update_working(self):
        """Test that create_status_update_event produces proper working status."""
        from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
            A2ACompletionBridgeTransformation,
            A2AStreamingContext,
        )

        ctx = A2AStreamingContext(
            request_id="req-456",
            input_message={"role": "user", "parts": []},
        )

        event = A2ACompletionBridgeTransformation.create_status_update_event(
            ctx=ctx,
            state="working",
            final=False,
            message_text="Processing...",
        )

        assert event["result"]["kind"] == "status-update"
        assert event["result"]["status"]["state"] == "working"
        assert event["result"]["final"] is False
        assert "taskId" in event["result"]
        assert "contextId" in event["result"]
        assert "timestamp" in event["result"]["status"]

    def test_create_artifact_update(self):
        """Test that create_artifact_update_event produces proper artifact event."""
        from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
            A2ACompletionBridgeTransformation,
            A2AStreamingContext,
        )

        ctx = A2AStreamingContext(
            request_id="req-456",
            input_message={"role": "user", "parts": []},
        )

        event = A2ACompletionBridgeTransformation.create_artifact_update_event(
            ctx=ctx,
            text="Hello, I am an AI assistant.",
        )

        assert event["result"]["kind"] == "artifact-update"
        assert "artifact" in event["result"]
        assert "artifactId" in event["result"]["artifact"]
        assert event["result"]["artifact"]["name"] == "response"
        assert event["result"]["artifact"]["parts"][0]["kind"] == "text"
        assert event["result"]["artifact"]["parts"][0]["text"] == "Hello, I am an AI assistant."


@pytest.mark.asyncio
async def test_handle_streaming_emits_proper_events():
    """Test that handle_streaming emits events in correct order with proper structure."""
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    # Mock litellm.acompletion to return a streaming response
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta = MagicMock()
    mock_chunk1.choices[0].delta.content = "Hello"

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta = MagicMock()
    mock_chunk2.choices[0].delta.content = " world"

    async def mock_streaming_response():
        yield mock_chunk1
        yield mock_chunk2

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_streaming_response()

        params = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hi"}],
                "messageId": "msg-123",
            }
        }

        events = []
        async for event in A2ACompletionBridgeHandler.handle_streaming(
            request_id="req-456",
            params=params,
            litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
            api_base="http://localhost:2024",
        ):
            events.append(event)

        # Should have 4 events: task, working, artifact, completed
        assert len(events) == 4

        # Event 1: task submitted
        assert events[0]["result"]["kind"] == "task"
        assert events[0]["result"]["status"]["state"] == "submitted"

        # Event 2: status working
        assert events[1]["result"]["kind"] == "status-update"
        assert events[1]["result"]["status"]["state"] == "working"
        assert events[1]["result"]["final"] is False

        # Event 3: artifact update with accumulated content
        assert events[2]["result"]["kind"] == "artifact-update"
        assert events[2]["result"]["artifact"]["parts"][0]["text"] == "Hello world"

        # Event 4: status completed
        assert events[3]["result"]["kind"] == "status-update"
        assert events[3]["result"]["status"]["state"] == "completed"
        assert events[3]["result"]["final"] is True


@pytest.mark.asyncio
async def test_handle_streaming_forwards_api_key():
    """Test that handle_streaming forwards api_key from litellm_params to acompletion."""
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = "Response"

    async def mock_streaming_response():
        yield mock_chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_streaming_response()

        params = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hi"}],
                "messageId": "msg-123",
            }
        }

        events = []
        async for event in A2ACompletionBridgeHandler.handle_streaming(
            request_id="req-456",
            params=params,
            litellm_params={
                "custom_llm_provider": "azure_ai",
                "model": "agents/asst_123",
                "api_key": "test-api-key-12345",
            },
            api_base="https://example.azure.com/",
        ):
            events.append(event)

        # Verify acompletion was called with api_key
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["api_key"] == "test-api-key-12345"
        assert call_kwargs["api_base"] == "https://example.azure.com/"
        assert call_kwargs["model"] == "azure_ai/agents/asst_123"


@pytest.mark.asyncio
async def test_handle_non_streaming_forwards_api_key():
    """Test that handle_non_streaming forwards api_key from litellm_params to acompletion."""
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Hello!"
    mock_response.id = "resp-123"

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        params = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hi"}],
                "messageId": "msg-123",
            }
        }

        await A2ACompletionBridgeHandler.handle_non_streaming(
            request_id="req-456",
            params=params,
            litellm_params={
                "custom_llm_provider": "azure_ai",
                "model": "agents/asst_456",
                "api_key": "my-secret-api-key",
            },
            api_base="https://my-azure.com/",
        )

        # Verify acompletion was called with api_key
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["api_key"] == "my-secret-api-key"
        assert call_kwargs["api_base"] == "https://my-azure.com/"
        assert call_kwargs["model"] == "azure_ai/agents/asst_456"

