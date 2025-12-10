
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.google_endpoints.endpoints import google_generate_content
from fastapi import Request, Response
from fastapi.datastructures import Headers
from litellm.proxy.proxy_server import initialize
from litellm.utils import ModelResponse

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


@pytest.fixture
def mock_request(request):
    """Create a mock FastAPI request with the sample payload."""
    mock_req = MagicMock(spec=Request)
    mock_req.headers = Headers({"content-type": "application/json"})
    mock_req.method = "POST"
    mock_req.url.path = request.param.get("path")
    
    async def mock_body():
        return json.dumps(request.param.get("payload", {})).encode('utf-8')
    
    mock_req.body = mock_body
    return mock_req


@pytest.fixture  
def mock_response():
    """Create a mock FastAPI response."""
    return MagicMock(spec=Response)


@pytest.mark.asyncio
@pytest.mark.parametrize("mock_request", [{"path": "/v1beta/models/bedrock/claude-sonnet-3.7:generateContent", "payload": {"contents": [{"parts":[{"text": "The quick brown fox jumps over the lazy dog."}]}]}}], indirect=True)
async def test_google_generate_content_with_slashes_in_model_name(
    mock_request, mock_response, mock_user_api_key_dict
):
    """
    Test that the google_generate_content endpoint correctly handles model names with slashes.
    """
    config = {
        "model_list": [
            {
                "model_name": "bedrock/claude-sonnet-3.7",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                },
            }
        ]
    }
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_config.yaml"
    with open(config_fp, "w") as f:
        yaml.dump(config, f)

    try:
        await initialize(config=config_fp)

        with patch("litellm.proxy.proxy_server.llm_router.agenerate_content", new_callable=AsyncMock) as mock_agenerate_content:
            mock_agenerate_content.return_value = ModelResponse()
            
            await google_generate_content(
                request=mock_request,
                model_name="bedrock/claude-sonnet-3.7",
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_agenerate_content.assert_called_once()
            _, call_kwargs = mock_agenerate_content.call_args
            assert call_kwargs["model"] == "bedrock/claude-sonnet-3.7"
    finally:
        if os.path.exists(config_fp):
            os.remove(config_fp)
