import datetime
import json
import os
import unittest
from typing import Final, List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

import litellm


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [True, False])
async def test_pubsub_spend_exports_charge_evaluation_creator(
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger
    from litellm.litellm_core_utils.litellm_logging import create_dummy_standard_logging_payload
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm, "gcs_pub_sub_use_v1", legacy)
    logger: Final = GcsPubSubLogger(project_id="project", topic_id="spend")
    source: Final = {"user_api_key_user_id": "sampled", "user_api_key_billing_user_id": "admin"}
    kwargs: Final = {
        "model": "eval-model",
        "response_cost": 0.25,
        "litellm_params": {"metadata": source},
        "standard_logging_object": {
            **create_dummy_standard_logging_payload(),
            "metadata": source,
            "response_cost": 0.25,
        },
    }
    now: Final = datetime.datetime(2026, 1, 1)

    await logger.async_log_success_event(kwargs, litellm.ModelResponse(id="eval", choices=[]), now, now)

    assert len(logger.log_queue) == 1
    row: Final = logger.log_queue[0]
    assert (row["user"] if legacy else row["metadata"]["user_api_key_user_id"]) == "admin"
    assert source["user_api_key_user_id"] == "sampled"


@pytest.mark.asyncio
async def test_construct_request_headers_project_id_from_env(monkeypatch):
    """Test that construct_request_headers uses GCS_PUBSUB_PROJECT_ID environment variable."""
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger

    # Set up test environment variable
    test_project_id = "test-project-123"
    monkeypatch.setenv("GCS_PUBSUB_PROJECT_ID", test_project_id)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.premium_user",
        True,
    )

    try:
        # Create handler with no project_id
        handler = GcsPubSubLogger(
            topic_id="test-topic", credentials_path="test-path.json"
        )

        # Mock the Vertex AI auth calls
        mock_auth_header = "mock-auth-header"
        mock_token = "mock-token"

        with patch(
            "litellm.vertex_chat_completion._ensure_access_token_async"
        ) as mock_ensure_token:
            mock_ensure_token.return_value = (mock_auth_header, test_project_id)

            with patch(
                "litellm.vertex_chat_completion._get_token_and_url"
            ) as mock_get_token:
                mock_get_token.return_value = (mock_token, "mock-url")

                # Call construct_request_headers
                headers = await handler.construct_request_headers()

                # Verify headers
                assert headers == {
                    "Authorization": f"Bearer {mock_token}",
                    "Content-Type": "application/json",
                }

                # Verify _ensure_access_token_async was called with correct project_id
                mock_ensure_token.assert_called_once_with(
                    credentials="test-path.json",
                    project_id=test_project_id,
                    custom_llm_provider="vertex_ai",
                )
    finally:
        # Clean up environment variable
        del os.environ["GCS_PUBSUB_PROJECT_ID"]
