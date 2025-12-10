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
