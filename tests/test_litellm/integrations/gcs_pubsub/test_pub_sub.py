import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path
import litellm


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
