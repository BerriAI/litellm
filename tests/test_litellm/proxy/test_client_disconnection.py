"""
Test client disconnection detection functionality.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.common_request_processing import _check_request_disconnection


@pytest.mark.asyncio
async def test_check_request_disconnection_with_disconnect():
    """Disconnect path: polling sees disconnected only after is_disconnected becomes True."""
    mock_request = MagicMock(spec=["is_disconnected"])
    mock_request.is_disconnected = AsyncMock(side_effect=[False, True])

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
    assert mock_request.is_disconnected.await_count == 2


@pytest.mark.asyncio
async def test_check_request_disconnection_no_disconnect():
    """Cancel watcher mid-flight: LLM task must not be cancelled like a disconnect."""
    mock_request = MagicMock(spec=["is_disconnected"])
    mock_request.is_disconnected = AsyncMock(return_value=False)

    mock_llm_task = MagicMock()  # sync mock so .cancel() doesn't return a coroutine
    disconnect_event = asyncio.Event()

    task = asyncio.create_task(
        _check_request_disconnection(mock_request, mock_llm_task, disconnect_event)
    )
    await asyncio.sleep(0.1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    mock_llm_task.cancel.assert_not_called()
    assert not disconnect_event.is_set()
