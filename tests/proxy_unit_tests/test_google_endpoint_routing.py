import json
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import app, initialize

# Create a TestClient
client = TestClient(app)

@pytest.fixture
def mock_user_api_key_dict():
    """Mock user API key dictionary."""
    return UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        user_email="test@example.com",
        team_id="test_team_id",
        max_budget=100.0,
        spend=0.0,
        user_role="internal_user",
        allowed_cache_controls=[],
        metadata={},
        tpm_limit=None,
        rpm_limit=None,
    )

def mock_user_api_key_auth_dependency(mock_user_api_key_dict):
    async def override():
        return mock_user_api_key_dict
    return override

@pytest.mark.asyncio
async def test_google_generate_content_with_slashes_in_model_name(mock_user_api_key_dict):
    """
    Test that the google_generate_content endpoint correctly handles model names with slashes.
    """
    with patch("litellm.proxy.proxy_server.llm_router") as mock_llm_router:
        mock_llm_router.agenerate_content = AsyncMock()
        
        # Override the dependency
        app.dependency_overrides[UserAPIKeyAuth] = mock_user_api_key_auth_dependency(mock_user_api_key_dict)

        # Initialize the router
        await initialize(model_list=[
            {
                "model_name": "bedrock/claude-sonnet-3.7",
                "litellm_params": {
                    "model": "bedrock/claude-3-sonnet-20240229-v1:0",
                },
            }
        ])

        response = client.post("/v1beta/models/bedrock/claude-sonnet-3.7:generateContent", json={})
        
        # Reset the dependency override
        app.dependency_overrides = {}
        
        mock_llm_router.agenerate_content.assert_called_once()
        call_args = mock_llm_router.agenerate_content.call_args[1]
        assert call_args["model"] == "bedrock/claude-sonnet-3.7"