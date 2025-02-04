import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
import pytest
from fastapi import Request
from litellm.proxy.utils import _get_redoc_url, _get_docs_url

sys.path.insert(0, os.path.abspath("../.."))
import litellm


@pytest.mark.asyncio
async def test_disable_error_logs():
    """
    Test that the error logs are not written to the database when disable_error_logs is True
    """
    # Mock the necessary components
    mock_prisma_client = AsyncMock()
    mock_general_settings = {"disable_error_logs": True}

    with patch(
        "litellm.proxy.proxy_server.general_settings", mock_general_settings
    ), patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):

        # Create a test exception
        test_exception = Exception("Test error")
        test_kwargs = {
            "model": "gpt-4",
            "exception": test_exception,
            "optional_params": {},
            "litellm_params": {"metadata": {}},
        }

        # Call the failure handler
        from litellm.proxy.proxy_server import _PROXY_failure_handler

        await _PROXY_failure_handler(
            kwargs=test_kwargs,
            completion_response=None,
            start_time="2024-01-01",
            end_time="2024-01-01",
        )

        # Verify prisma client was not called to create error logs
        if hasattr(mock_prisma_client, "db"):
            assert not mock_prisma_client.db.litellm_errorlogs.create.called


@pytest.mark.asyncio
async def test_disable_spend_logs():
    """
    Test that the spend logs are not written to the database when disable_spend_logs is True
    """
    # Mock the necessary components
    mock_prisma_client = Mock()
    mock_prisma_client.spend_log_transactions = []

    with patch("litellm.proxy.proxy_server.disable_spend_logs", True), patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ):
        from litellm.proxy.proxy_server import update_database

        # Call update_database with disable_spend_logs=True
        await update_database(
            token="fake-token",
            response_cost=0.1,
            user_id="user123",
            completion_response=None,
            start_time="2024-01-01",
            end_time="2024-01-01",
        )
        # Verify no spend logs were added
        assert len(mock_prisma_client.spend_log_transactions) == 0


@pytest.mark.asyncio
async def test_enable_error_logs():
    """
    Test that the error logs are written to the database when disable_error_logs is False
    """
    # Mock the necessary components
    mock_prisma_client = AsyncMock()
    mock_general_settings = {"disable_error_logs": False}

    with patch(
        "litellm.proxy.proxy_server.general_settings", mock_general_settings
    ), patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):

        # Create a test exception
        test_exception = Exception("Test error")
        test_kwargs = {
            "model": "gpt-4",
            "exception": test_exception,
            "optional_params": {},
            "litellm_params": {"metadata": {}},
        }

        # Call the failure handler
        from litellm.proxy.proxy_server import _PROXY_failure_handler

        await _PROXY_failure_handler(
            kwargs=test_kwargs,
            completion_response=None,
            start_time="2024-01-01",
            end_time="2024-01-01",
        )

        # Verify prisma client was called to create error logs
        if hasattr(mock_prisma_client, "db"):
            assert mock_prisma_client.db.litellm_errorlogs.create.called
