"""
Test A2A cost calculator with cost_per_query parameter.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger


class CostLogger(CustomLogger):
    """Custom logger to capture response_cost."""

    def __init__(self):
        self.response_cost: Optional[float] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        slp = kwargs.get("standard_logging_object")
        if slp:
            self.response_cost = slp.get("response_cost") if isinstance(slp, dict) else getattr(slp, "response_cost", None)


@pytest.mark.asyncio
async def test_asend_message_uses_cost_per_query():
    """
    Test that asend_message uses cost_per_query param for response_cost.
    """
    from litellm.a2a_protocol import asend_message

    # Setup logger
    litellm.logging_callback_manager._reset_all_callbacks()
    cost_logger = CostLogger()
    litellm.callbacks = [cost_logger]

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    # Mock response with required fields
    mock_response = MagicMock()
    mock_response.model_dump = MagicMock(return_value={
        "id": "test-123",
        "jsonrpc": "2.0",
        "result": {"status": "completed"},
    })
    mock_client.send_message = AsyncMock(return_value=mock_response)

    # Mock request
    mock_request = MagicMock()
    mock_request.id = "test-123"

    # Call asend_message with cost_per_query
    await asend_message(
        a2a_client=mock_client,
        request=mock_request,
        cost_per_query=0.05,
    )

    await asyncio.sleep(0.1)

    assert cost_logger.response_cost == 0.05


class TokenLogger(CustomLogger):
    """Custom logger to capture token usage."""

    def __init__(self):
        self.standard_logging_payload = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object")


@pytest.mark.asyncio
async def test_asend_message_token_tracking():
    """
    Test that asend_message tracks input/output/total tokens.
    """
    from litellm.a2a_protocol import asend_message

    # Setup logger
    litellm.logging_callback_manager._reset_all_callbacks()
    token_logger = TokenLogger()
    litellm.callbacks = [token_logger]

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    # Realistic A2A response with message parts
    mock_response = MagicMock()
    mock_response.model_dump = MagicMock(return_value={
        "id": "test-123",
        "jsonrpc": "2.0",
        "result": {
            "status": {"state": "completed"},
            "message": {
                "role": "assistant",
                "parts": [{"kind": "text", "text": "Hello! I am your assistant. How can I help you today?"}],
                "messageId": "msg-456",
            }
        },
    })
    mock_client.send_message = AsyncMock(return_value=mock_response)

    # Mock request with message parts
    mock_request = MagicMock()
    mock_request.id = "test-123"
    mock_request.params = MagicMock()
    mock_request.params.message = {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello, what can you do?"}],
        "messageId": "msg-123",
    }

    await asend_message(
        a2a_client=mock_client,
        request=mock_request,
    )

    await asyncio.sleep(0.1)

    slp = token_logger.standard_logging_payload
    assert slp is not None, "standard_logging_payload should be captured"
    print("\n=== Token Tracking Results ===")
    print(f"prompt_tokens: {slp.get('prompt_tokens')}")
    print(f"completion_tokens: {slp.get('completion_tokens')}")
    print(f"total_tokens: {slp.get('total_tokens')}")


@pytest.mark.asyncio
async def test_asend_message_streaming_token_tracking():
    """
    Test that streaming A2A iterator collects chunks and calculates correct token counts.
    """
    from litellm.a2a_protocol.streaming_iterator import A2AStreamingIterator
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    # Create mock streaming chunks
    async def mock_streaming_response():
        """Mock async generator yielding A2A streaming chunks."""
        # Chunk 1: status update
        chunk1 = MagicMock()
        chunk1.model_dump = MagicMock(return_value={
            "id": "test-123",
            "jsonrpc": "2.0",
            "result": {"status": {"state": "working"}},
        })
        yield chunk1

        # Chunk 2: partial message
        chunk2 = MagicMock()
        chunk2.model_dump = MagicMock(return_value={
            "id": "test-123",
            "jsonrpc": "2.0",
            "result": {
                "status": {"state": "working"},
                "message": {
                    "role": "assistant",
                    "parts": [{"kind": "text", "text": "Hello! "}],
                    "messageId": "msg-456",
                }
            },
        })
        yield chunk2

        # Chunk 3: completed with full message
        chunk3 = MagicMock()
        chunk3.model_dump = MagicMock(return_value={
            "id": "test-123",
            "jsonrpc": "2.0",
            "result": {
                "status": {"state": "completed"},
                "message": {
                    "role": "assistant",
                    "parts": [{"kind": "text", "text": "Hello! I am your assistant."}],
                    "messageId": "msg-456",
                }
            },
        })
        yield chunk3

    # Mock request with message parts
    mock_request = MagicMock()
    mock_request.id = "test-123"
    mock_request.params = MagicMock()
    mock_request.params.message = {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello, what can you do?"}],
        "messageId": "msg-123",
    }

    # Create logging obj
    logging_obj = LiteLLMLoggingObj(
        model="a2a_agent/test-agent",
        messages=[],
        stream=True,
        call_type="asend_message_streaming",
        litellm_call_id="test-call-id",
        start_time=None,
        function_id="test-function",
    )
    logging_obj.model_call_details["custom_llm_provider"] = "a2a_agent"

    # Create streaming iterator with logging
    stream = mock_streaming_response()
    streaming_iterator = A2AStreamingIterator(
        stream=stream,
        request=mock_request,
        logging_obj=logging_obj,
        agent_name="test-agent",
    )

    # Consume streaming response
    chunks = []
    async for chunk in streaming_iterator:
        chunks.append(chunk)

    # Wait for async tasks to complete
    await asyncio.sleep(0.1)

    print(f"\n=== Streaming Token Tracking Results ===")
    print(f"Received {len(chunks)} chunks")
    print(f"Collected text parts: {streaming_iterator.collected_text_parts}")

    # Verify chunks collected
    assert len(chunks) == 3

    # Verify text was collected from chunks
    assert len(streaming_iterator.collected_text_parts) > 0

    # Verify usage was set on logging obj
    usage = logging_obj.model_call_details.get("usage")
    assert usage is not None, "usage should be set on logging_obj"

    print(f"prompt_tokens: {usage.prompt_tokens}")
    print(f"completion_tokens: {usage.completion_tokens}")
    print(f"total_tokens: {usage.total_tokens}")

    # Verify token counts are calculated
    # Input: "Hello, what can you do?" = ~7 tokens
    # Output: "Hello! I am your assistant." = ~6 tokens
    assert usage.prompt_tokens > 0, "prompt_tokens should be > 0"
    assert usage.completion_tokens > 0, "completion_tokens should be > 0"
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
