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


class TokenAndCostLogger(CustomLogger):
    """Custom logger to capture both token counts and cost."""

    def __init__(self):
        self.response_cost: Optional[float] = None
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        slp = kwargs.get("standard_logging_object")
        if slp:
            self.response_cost = slp.get("response_cost") if isinstance(slp, dict) else getattr(slp, "response_cost", None)
            self.prompt_tokens = slp.get("prompt_tokens") if isinstance(slp, dict) else getattr(slp, "prompt_tokens", None)
            self.completion_tokens = slp.get("completion_tokens") if isinstance(slp, dict) else getattr(slp, "completion_tokens", None)


@pytest.mark.asyncio
async def test_asend_message_uses_input_output_cost_per_token():
    """
    Test that asend_message calculates cost using input_cost_per_token and output_cost_per_token.
    Validates exact cost calculation: cost = (prompt_tokens * input_cost) + (completion_tokens * output_cost)
    """
    from litellm.a2a_protocol import asend_message

    # Setup logger
    litellm.logging_callback_manager._reset_all_callbacks()
    token_cost_logger = TokenAndCostLogger()
    litellm.callbacks = [token_cost_logger]

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

    # Define specific cost per token values
    input_cost_per_token = 0.00001  # $0.01 per 1000 tokens
    output_cost_per_token = 0.00002  # $0.02 per 1000 tokens

    await asend_message(
        a2a_client=mock_client,
        request=mock_request,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
    )

    await asyncio.sleep(0.1)

    # Get actual token counts from logger
    prompt_tokens = token_cost_logger.prompt_tokens
    completion_tokens = token_cost_logger.completion_tokens
    response_cost = token_cost_logger.response_cost

    print(f"\n=== Token-Based Cost Results ===")
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    print(f"input_cost_per_token: {input_cost_per_token}")
    print(f"output_cost_per_token: {output_cost_per_token}")
    print(f"response_cost: {response_cost}")

    # Verify tokens were captured
    assert prompt_tokens is not None, "prompt_tokens should be captured"
    assert completion_tokens is not None, "completion_tokens should be captured"
    assert response_cost is not None, "response_cost should be captured"

    # Calculate expected cost
    expected_cost = (prompt_tokens * input_cost_per_token) + (completion_tokens * output_cost_per_token)
    print(f"expected_cost: {expected_cost}")

    # Verify exact cost calculation
    assert response_cost == expected_cost, f"response_cost {response_cost} should equal expected {expected_cost}"


class AgentIdLogger(CustomLogger):
    """Custom logger to capture agent_id from kwargs."""

    def __init__(self):
        self.agent_id: Optional[str] = None
        self.kwargs: Optional[dict] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.kwargs = kwargs
        self.agent_id = kwargs.get("agent_id")


@pytest.mark.asyncio
async def test_asend_message_passes_agent_id_to_callback():
    """
    Test that asend_message passes agent_id to callbacks via kwargs.
    """
    from litellm.a2a_protocol import asend_message

    # Setup logger
    litellm.logging_callback_manager._reset_all_callbacks()
    agent_id_logger = AgentIdLogger()
    litellm.callbacks = [agent_id_logger]

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    # Mock response
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

    test_agent_id = "agent-uuid-12345"

    # Call asend_message with agent_id
    await asend_message(
        a2a_client=mock_client,
        request=mock_request,
        agent_id=test_agent_id,
    )

    await asyncio.sleep(0.1)

    # Verify agent_id was passed to callback
    assert agent_id_logger.agent_id == test_agent_id, f"Expected agent_id '{test_agent_id}', got '{agent_id_logger.agent_id}'"


class MetadataLogger(CustomLogger):
    """Custom logger to capture metadata from kwargs for proxy spend tracking."""

    def __init__(self):
        self.metadata: Optional[dict] = None
        self.litellm_params: Optional[dict] = None
        self.user_api_key: Optional[str] = None
        self.user_id: Optional[str] = None
        self.team_id: Optional[str] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.litellm_params = kwargs.get("litellm_params", {})
        self.metadata = self.litellm_params.get("metadata", {})
        self.user_api_key = self.metadata.get("user_api_key")
        self.user_id = self.metadata.get("user_api_key_user_id")
        self.team_id = self.metadata.get("user_api_key_team_id")


@pytest.mark.asyncio
async def test_asend_message_streaming_propagates_metadata():
    """
    Test that asend_message_streaming propagates metadata to logging object.
    This ensures user_api_key, user_id, team_id are available for SpendLogs.
    """
    from litellm.a2a_protocol import asend_message_streaming

    # Setup logger
    litellm.logging_callback_manager._reset_all_callbacks()
    metadata_logger = MetadataLogger()
    litellm.logging_callback_manager.add_litellm_async_success_callback(metadata_logger)

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    # Mock streaming response
    async def mock_stream():
        yield MagicMock(model_dump=lambda mode, exclude_none: {"chunk": 1})
        yield MagicMock(model_dump=lambda mode, exclude_none: {"chunk": 2})

    mock_client.send_message_streaming = MagicMock(return_value=mock_stream())

    # Mock request
    mock_request = MagicMock()
    mock_request.id = "test-stream-metadata"
    mock_request.params = MagicMock()
    mock_request.params.message = {"role": "user", "parts": [{"kind": "text", "text": "Hello"}]}

    # Metadata from proxy (contains user_api_key, user_id, team_id for SpendLogs)
    test_metadata = {
        "user_api_key": "sk-test-key-hash-12345",
        "user_api_key_user_id": "user-uuid-123",
        "user_api_key_team_id": "team-uuid-456",
    }

    # Consume streaming response with metadata
    chunks = []
    async for chunk in asend_message_streaming(
        a2a_client=mock_client,
        request=mock_request,
        metadata=test_metadata,
    ):
        chunks.append(chunk)

    await asyncio.sleep(0.2)

    # Verify metadata was propagated to callback
    assert metadata_logger.user_api_key == "sk-test-key-hash-12345"
    assert metadata_logger.user_id == "user-uuid-123"
    assert metadata_logger.team_id == "team-uuid-456"


@pytest.mark.asyncio
async def test_asend_message_streaming_triggers_callbacks():
    """
    Test that asend_message_streaming triggers callbacks after stream completes.
    """
    from litellm.a2a_protocol import asend_message_streaming

    # Setup logger - must use logging_callback_manager to properly register
    litellm.logging_callback_manager._reset_all_callbacks()
    callback_logger = AgentIdLogger()
    litellm.logging_callback_manager.add_litellm_async_success_callback(callback_logger)
    litellm.logging_callback_manager.add_litellm_success_callback(callback_logger)

    # Mock A2A client
    mock_client = MagicMock()
    mock_client._litellm_agent_card = MagicMock()
    mock_client._litellm_agent_card.name = "test-agent"

    # Mock streaming response
    async def mock_stream():
        yield MagicMock(model_dump=lambda mode, exclude_none: {"chunk": 1})
        yield MagicMock(model_dump=lambda mode, exclude_none: {"chunk": 2})

    mock_client.send_message_streaming = MagicMock(return_value=mock_stream())

    # Mock request
    mock_request = MagicMock()
    mock_request.id = "test-stream-123"
    mock_request.params = MagicMock()
    mock_request.params.message = {"role": "user", "parts": [{"kind": "text", "text": "Hello"}]}

    test_agent_id = "test-agent-id-streaming"

    # Consume streaming response
    chunks = []
    async for chunk in asend_message_streaming(
        a2a_client=mock_client,
        request=mock_request,
        agent_id=test_agent_id,
    ):
        chunks.append(chunk)

    await asyncio.sleep(0.2)

    # Verify chunks were received
    assert len(chunks) == 2

    # Verify callbacks WERE triggered after stream completed
    assert callback_logger.kwargs is not None, "Streaming should trigger callbacks after completion"
    assert callback_logger.agent_id == test_agent_id, f"Expected agent_id '{test_agent_id}', got '{callback_logger.agent_id}'"
