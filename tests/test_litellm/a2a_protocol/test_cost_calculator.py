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
