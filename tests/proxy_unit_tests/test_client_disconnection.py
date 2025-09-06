"""
Test client disconnection detection functionality.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock

from litellm.proxy.common_request_processing import _check_request_disconnection


@pytest.mark.asyncio
async def test_check_request_disconnection_with_disconnect():
    """Test that _check_request_disconnection cancels task when client disconnects."""
    mock_request = AsyncMock()
    mock_request.receive.side_effect = [
        {"type": "http.request"},   # First call
        {"type": "http.disconnect"} # Second call - disconnect
    ]

    mock_llm_task = AsyncMock()

    await _check_request_disconnection(mock_request, mock_llm_task)

    mock_llm_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_check_request_disconnection_no_disconnect():
    """Test that _check_request_disconnection handles normal requests."""
    mock_request = AsyncMock()
    mock_request.receive.return_value = {"type": "http.request"}

    mock_llm_task = AsyncMock()

    # This will timeout after 600 seconds, but we don't need to wait
    # Just test that it doesn't crash immediately
    task = asyncio.create_task(_check_request_disconnection(mock_request, mock_llm_task))
    await asyncio.sleep(0.1)  # Let it run briefly
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Task should not be cancelled during normal operation
    mock_llm_task.cancel.assert_not_called()