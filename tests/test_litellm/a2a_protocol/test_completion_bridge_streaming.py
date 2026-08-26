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

from litellm.types.utils import Choices, Message, ModelResponse


class TestA2AStreamingTransformation:
    """Test the A2A streaming transformation creates proper events."""

    def test_a2a_metadata_forwarded_to_completion_params(self):
        from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
            A2ACompletionBridgeTransformation,
        )

        message = {
            "role": "user",
            "parts": [{"text": "Reply to ticket #4823"}],
            "metadata": {"skillId": "draft_reply"},
        }
        openai_messages = A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(message)
        # Metadata is forwarded on the run payload only, not duplicated on messages.
        assert "metadata" not in openai_messages[0]

        completion_params: dict = {
            "model": "langgraph/agent",
            "messages": openai_messages,
        }
        A2ACompletionBridgeTransformation.apply_forward_metadata_to_completion_params(
            completion_params=completion_params,
            a2a_message=message,
            params={"metadata": {"trace": "abc"}},
        )
        assert completion_params["extra_body"]["metadata"] == {
            "trace": "abc",
            "skillId": "draft_reply",
        }

    def test_configured_metadata_wins_over_forwarded_a2a_metadata(self):
        from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
            A2ACompletionBridgeTransformation,
        )

        # Agent-owner-configured run metadata in ``extra_body``.
        completion_params: dict = {
            "model": "langgraph/agent",
            "messages": [],
            "extra_body": {
                "metadata": {"owner_tag": "prod", "trace": "server-set"},
                "other": "keep",
            },
        }
        # Client tries to overwrite ``trace`` and inject a new key.
        message = {
            "role": "user",
            "parts": [{"text": "hi"}],
            "metadata": {"trace": "client-spoof", "skillId": "draft_reply"},
        }
        A2ACompletionBridgeTransformation.apply_forward_metadata_to_completion_params(
            completion_params=completion_params,
            a2a_message=message,
            params={"metadata": {"trace": "client-spoof-2"}},
        )
        assert completion_params["extra_body"]["other"] == "keep"
        assert completion_params["extra_body"]["metadata"] == {
            "owner_tag": "prod",
            "trace": "server-set",
            "skillId": "draft_reply",
        }

    def test_langgraph_transform_preserves_message_metadata(self):
        from litellm.llms.langgraph.chat.transformation import LangGraphConfig

        config = LangGraphConfig()
        request = config.transform_request(
            model="langgraph/agent",
            messages=[
                {
                    "role": "user",
                    "content": "Reply to ticket #4823",
                    "metadata": {"skillId": "draft_reply"},
                }
            ],
            optional_params={},
            litellm_params={"stream": False},
            headers={},
        )
        assert request["input"]["messages"][-1]["metadata"] == {
            "skillId": "draft_reply",
        }

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
    mock_chunk2.choices[0].finish_reason = "length"
    mock_chunk2.usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}

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

        # Should have 5 events: task, working, two artifacts, completed
        assert len(events) == 5

        # Event 1: task submitted
        assert events[0]["result"]["kind"] == "task"
        assert events[0]["result"]["status"]["state"] == "submitted"

        # Event 2: status working
        assert events[1]["result"]["kind"] == "status-update"
        assert events[1]["result"]["status"]["state"] == "working"
        assert events[1]["result"]["final"] is False

        # Event 3: first artifact update
        assert events[2]["result"]["kind"] == "artifact-update"
        assert events[2]["result"]["artifact"]["parts"][0]["text"] == "Hello"

        # Event 4: second artifact update
        assert events[3]["result"]["kind"] == "artifact-update"
        assert events[3]["result"]["artifact"]["parts"][0]["text"] == " world"
        assert (
            events[2]["result"]["artifact"]["artifactId"]
            == events[3]["result"]["artifact"]["artifactId"]
        )

        # Event 5: status completed
        assert events[4]["result"]["kind"] == "status-update"
        assert events[4]["result"]["status"]["state"] == "completed"
        assert events[4]["result"]["final"] is True
        assert events[4]["result"]["finish_reason"] == "length"
        assert events[4]["usage"]["total_tokens"] == 5


def test_build_completion_params_keeps_bridge_routing_fields():
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    params = A2ACompletionBridgeHandler._build_completion_params(
        params={"message": {"role": "user", "parts": []}},
        litellm_params={
            "custom_llm_provider": "openai",
            "model": "agent",
            "api_base": "https://untrusted.example",
            "stream": False,
        },
        api_base="https://configured.example",
        agent_extra_headers=None,
        stream=True,
    )

    assert params["api_base"] == "https://configured.example"
    assert params["stream"] is True


@pytest.mark.asyncio
async def test_handle_streaming_accumulates_logprobs_and_provider_metadata():
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    chunks = []
    for token in ("a", "b"):
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = None
        choice.delta.content = token
        choice.logprobs = {"content": [{"token": token}]}
        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.system_fingerprint = "fp-1"
        chunk.service_tier = "scale"
        chunks.append(chunk)
    chunks[-1].choices[0].finish_reason = "stop"

    async def mock_streaming_response():
        for chunk in chunks:
            yield chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_streaming_response()
        events = [
            event
            async for event in A2ACompletionBridgeHandler.handle_streaming(
                request_id="req-metadata",
                params={"message": {"role": "user", "parts": []}},
                litellm_params={"custom_llm_provider": "openai", "model": "agent"},
            )
        ]

    result = events[-1]
    assert result["system_fingerprint"] == "fp-1"
    assert result["service_tier"] == "scale"
    assert result["result"]["choices"][0]["logprobs"]["content"] == [
        {"token": "a"},
        {"token": "b"},
    ]


@pytest.mark.asyncio
async def test_handle_streaming_preserves_multiple_choices():
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    mock_chunk = MagicMock()
    first_choice = MagicMock()
    first_choice.index = 0
    first_choice.finish_reason = None
    first_choice.delta.content = "first"
    second_choice = MagicMock()
    second_choice.index = 1
    second_choice.finish_reason = "length"
    second_choice.delta.content = "second"
    mock_chunk.choices = [first_choice, second_choice]

    async def mock_streaming_response():
        yield mock_chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_streaming_response()
        events = [
            event
            async for event in A2ACompletionBridgeHandler.handle_streaming(
                request_id="req-choices",
                params={"message": {"role": "user", "parts": []}},
                litellm_params={"custom_llm_provider": "langgraph", "model": "agent", "n": 2},
            )
        ]

    choices = events[-1]["result"]["choices"]
    assert [choice["index"] for choice in choices] == [0, 1]
    assert choices[0]["message"]["parts"][0]["text"] == "first"
    assert choices[1]["message"]["parts"][0]["text"] == "second"
    assert choices[1]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_handle_streaming_preserves_non_text_delta_fields():
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    delta = MagicMock()
    delta.content = ""
    delta.tool_calls = None
    delta.model_dump.return_value = {
        "audio": {"data": "abc"},
        "reasoning_content": "thinking",
        "provider_specific_fields": {"trace_id": "trace-1"},
    }
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.delta = delta
    choice.logprobs = {"content": []}
    chunk = MagicMock()
    chunk.choices = [choice]

    async def mock_streaming_response():
        yield chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_streaming_response()
        events = [
            event
            async for event in A2ACompletionBridgeHandler.handle_streaming(
                request_id="req-fields",
                params={"message": {"role": "user", "parts": []}},
                litellm_params={"custom_llm_provider": "langgraph", "model": "agent"},
            )
        ]

    result = events[-1]["result"]
    choice_result = result["choices"][0]
    assert choice_result["delta"] == {
        "audio": {"data": "abc"},
        "reasoning_content": "thinking",
        "provider_specific_fields": {"trace_id": "trace-1"},
    }
    assert choice_result["logprobs"] == {"content": []}


@pytest.mark.asyncio
async def test_provider_config_receives_full_message_history():
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    provider_config = MagicMock()
    provider_config.handle_non_streaming = AsyncMock(return_value={"result": {}})
    messages = [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
    ]
    params = {
        "message": {"role": "user", "parts": []},
        "messages": messages,
    }

    with patch(
        "litellm.a2a_protocol.litellm_completion_bridge.handler.A2AProviderConfigManager.get_provider_config",
        return_value=provider_config,
    ):
        await A2ACompletionBridgeHandler.handle_non_streaming(
            request_id="req-1",
            params=params,
            litellm_params={"custom_llm_provider": "langflow", "model": "flow"},
        )

    assert provider_config.handle_non_streaming.await_args.kwargs["params"]["messages"] == messages


def test_response_transform_preserves_audio_and_logprobs():
    from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
        A2ACompletionBridgeTransformation,
    )

    response = ModelResponse(
        id="resp-1",
        model="test-model",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="hello",
                    role="assistant",
                    audio={"data": "abc", "expires_at": 1, "transcript": "hello"},
                ),
                logprobs={"content": []},
            )
        ],
    )

    transformed = A2ACompletionBridgeTransformation.openai_response_to_a2a_response(response)

    assert transformed["result"]["audio"]["data"] == "abc"
    assert transformed["result"]["logprobs"] == {"content": []}


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
