import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.mark.asyncio
async def test_async_post_call_failure_hook():
    # Setup
    logger = _ProxyDBLogger()

    # Mock user_api_key_dict
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="test_alias",
        user_email="test@example.com",
        user_id="test_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        team_alias="test_team_alias",
        end_user_id="test_end_user_id",
    )

    # Mock request data
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"original_key": "original_value"},
        "proxy_server_request": {"request_id": "test_request_id"},
    }

    # Mock exception
    original_exception = Exception("Test exception")

    # Mock update_database function
    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        # Call the method
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        # Assertions
        mock_update_database.assert_called_once()

        # Check the arguments passed to update_database
        call_args = mock_update_database.call_args[1]
        print("call_args", json.dumps(call_args, indent=4, default=str))
        assert call_args["token"] == "test_api_key"
        assert call_args["response_cost"] == 0.0
        assert call_args["user_id"] == "test_user_id"
        assert call_args["end_user_id"] == "test_end_user_id"
        assert call_args["team_id"] == "test_team_id"
        assert call_args["org_id"] == "test_org_id"
        assert call_args["completion_response"] == original_exception

        # Check that metadata was properly updated
        assert "litellm_params" in call_args["kwargs"]
        assert call_args["kwargs"]["litellm_params"]["proxy_server_request"] == {
            "request_id": "test_request_id"
        }
        metadata = call_args["kwargs"]["litellm_params"]["metadata"]
        assert metadata["user_api_key"] == "test_api_key"
        assert metadata["status"] == "failure"
        assert "error_information" in metadata
        assert metadata["original_key"] == "original_value"


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_non_llm_route():
    # Setup
    logger = _ProxyDBLogger()

    # Mock user_api_key_dict with a non-LLM route
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="test_alias",
        user_email="test@example.com",
        user_id="test_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        team_alias="test_team_alias",
        end_user_id="test_end_user_id",
        request_route="/custom/route",  # Non-LLM route
    )

    # Mock request data
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"original_key": "original_value"},
        "proxy_server_request": {"request_id": "test_request_id"},
    }

    # Mock exception
    original_exception = Exception("Test exception")

    # Mock update_database function
    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        # Call the method
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        # Assert that update_database was NOT called for non-LLM routes
        mock_update_database.assert_not_called()
