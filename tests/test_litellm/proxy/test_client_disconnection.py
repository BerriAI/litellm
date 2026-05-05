"""
Test client disconnection detection functionality.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.common_request_processing import _check_request_disconnection


@pytest.mark.asyncio
async def test_check_request_disconnection_with_disconnect():
    """Test that _check_request_disconnection cancels task and sets event when client disconnects."""
    mock_request = AsyncMock()
    mock_request.receive.side_effect = [
        {"type": "http.request"},  # First call
        {"type": "http.disconnect"},  # Second call - disconnect
    ]

    mock_llm_task = MagicMock()  # sync mock so .cancel() doesn't return a coroutine
    disconnect_event = asyncio.Event()

    with patch(
        "litellm.proxy.common_request_processing.asyncio.sleep", new_callable=AsyncMock
    ):
        await _check_request_disconnection(
            mock_request, mock_llm_task, disconnect_event
        )

    mock_llm_task.cancel.assert_called_once()
    assert disconnect_event.is_set()


@pytest.mark.asyncio
async def test_check_request_disconnection_no_disconnect():
    """Test that _check_request_disconnection does not cancel task during normal operation."""
    mock_request = AsyncMock()
    mock_request.receive.return_value = {"type": "http.request"}

    mock_llm_task = MagicMock()  # sync mock so .cancel() doesn't return a coroutine
    disconnect_event = asyncio.Event()

    task = asyncio.create_task(
        _check_request_disconnection(mock_request, mock_llm_task, disconnect_event)
    )
    await asyncio.sleep(0.1)  # Let it run briefly
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    mock_llm_task.cancel.assert_not_called()
    assert not disconnect_event.is_set()
