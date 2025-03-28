import importlib
import json
import os
import socket
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import click
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

import litellm


@pytest.mark.asyncio
async def test_initialize_scheduled_jobs_credentials(monkeypatch):
    """
    Test that get_credentials is only called when store_model_in_db is True
    """
    monkeypatch.delenv("DISABLE_PRISMA_SCHEMA_UPDATE", raising=False)
    monkeypatch.delenv("STORE_MODEL_IN_DB", raising=False)
    from litellm.proxy.proxy_server import ProxyStartupEvent
    from litellm.proxy.utils import ProxyLogging

    # Mock dependencies
    mock_prisma_client = MagicMock()
    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.slack_alerting_instance = MagicMock()
    mock_proxy_config = AsyncMock()

    with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config), patch(
        "litellm.proxy.proxy_server.store_model_in_db", False
    ):  # set store_model_in_db to False

        # Test when store_model_in_db is False
        await ProxyStartupEvent.initialize_scheduled_background_jobs(
            general_settings={},
            prisma_client=mock_prisma_client,
            proxy_budget_rescheduler_min_time=1,
            proxy_budget_rescheduler_max_time=2,
            proxy_batch_write_at=5,
            proxy_logging_obj=mock_proxy_logging,
        )

        # Verify get_credentials was not called
        mock_proxy_config.get_credentials.assert_not_called()

    # Now test with store_model_in_db = True
    with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config), patch(
        "litellm.proxy.proxy_server.store_model_in_db", True
    ), patch("litellm.proxy.proxy_server.get_secret_bool", return_value=True):

        await ProxyStartupEvent.initialize_scheduled_background_jobs(
            general_settings={},
            prisma_client=mock_prisma_client,
            proxy_budget_rescheduler_min_time=1,
            proxy_budget_rescheduler_max_time=2,
            proxy_batch_write_at=5,
            proxy_logging_obj=mock_proxy_logging,
        )

        # Verify get_credentials was called both directly and scheduled
        assert mock_proxy_config.get_credentials.call_count == 1  # Direct call

        # Verify a scheduled job was added for get_credentials
        mock_scheduler_calls = [
            call[0] for call in mock_proxy_config.get_credentials.mock_calls
        ]
        assert len(mock_scheduler_calls) > 0
