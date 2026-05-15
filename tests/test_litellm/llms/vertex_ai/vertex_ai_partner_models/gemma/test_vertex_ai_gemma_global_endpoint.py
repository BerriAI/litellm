"""
Tests for Vertex AI Gemma MaaS models that route through the partner-models
OpenAI-compatible path (https://aiplatform.googleapis.com/.../endpoints/openapi).

These tests verify that:
1. The correct global URL is constructed (https://aiplatform.googleapis.com)
2. get_vertex_region resolves to "global" when model_cost says so
3. acompletion() goes through the OpenAI-compatible handler and hits
   /endpoints/openapi/chat/completions
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.vertex_ai import VertexPartnerProvider


@pytest.fixture(autouse=True)
def clean_vertex_env():
    """Clear Google/Vertex AI environment variables before each test to prevent test isolation issues."""
    saved_env = {}
    env_vars_to_clear = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEXAI_PROJECT",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "VERTEX_AI_PROJECT",
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    yield

    for var, value in saved_env.items():
        os.environ[var] = value


class TestVertexBaseGetVertexRegionGemma:
    """Test the get_vertex_region method for Gemma MaaS via model_cost lookup."""

    def test_global_model_no_user_region_returns_global(self):
        vertex_base = VertexBase()

        with patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/google/gemma-4-26b-a4b-it-maas": {
                    "supported_regions": ["global"]
                }
            },
            clear=False,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region=None,
                model="google/gemma-4-26b-a4b-it-maas",
            )
            assert result == "global"

    def test_global_model_with_unsupported_user_region_overrides(self):
        vertex_base = VertexBase()

        with patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/google/gemma-4-26b-a4b-it-maas": {
                    "supported_regions": ["global"]
                }
            },
            clear=False,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region="us-central1",
                model="google/gemma-4-26b-a4b-it-maas",
            )
            assert result == "global"


class TestCreateVertexURLGemma:
    """Test that create_vertex_url produces the expected OpenAI-compatible URL."""

    def test_global_location_url_format(self):
        url = VertexBase.create_vertex_url(
            vertex_location="global",
            vertex_project="test-project",
            partner=VertexPartnerProvider.llama,
            stream=False,
            model="google/gemma-4-26b-a4b-it-maas",
        )

        assert url.startswith("https://aiplatform.googleapis.com")
        assert "global-aiplatform.googleapis.com" not in url
        assert "/locations/global/" in url
        assert url.endswith("/endpoints/openapi/chat/completions")

    def test_regional_location_url_format(self):
        url = VertexBase.create_vertex_url(
            vertex_location="us-central1",
            vertex_project="test-project",
            partner=VertexPartnerProvider.llama,
            stream=False,
            model="google/gemma-4-26b-a4b-it-maas",
        )

        assert url.startswith("https://us-central1-aiplatform.googleapis.com")
        assert "/locations/us-central1/" in url
        assert url.endswith("/endpoints/openapi/chat/completions")


@pytest.mark.asyncio
async def test_vertex_ai_gemma_global_endpoint_url():
    """
    End-to-end: acompletion on vertex_ai/google/gemma-4-26b-a4b-it-maas should
    POST to the global endpoints/openapi/chat/completions URL.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = {
        "id": "chatcmpl-gemma-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "google/gemma-4-26b-a4b-it-maas",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler,
        patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
            return_value=("fake-token", "test-project"),
        ),
        patch.dict(
            "sys.modules",
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
        patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/google/gemma-4-26b-a4b-it-maas": {
                    "supported_regions": ["global"]
                }
            },
            clear=False,
        ),
    ):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

        response = await litellm.acompletion(
            model="vertex_ai/google/gemma-4-26b-a4b-it-maas",
            messages=[{"role": "user", "content": "Hello"}],
            vertex_ai_project="test-project",
        )

        mock_http_handler.return_value.post.assert_called_once()

        call_args = mock_http_handler.return_value.post.call_args
        called_url = call_args.kwargs["url"]

        assert called_url.startswith("https://aiplatform.googleapis.com")
        assert "global-aiplatform.googleapis.com" not in called_url
        assert "/locations/global/" in called_url
        assert "/endpoints/openapi/chat/completions" in called_url

        assert response.model == "google/gemma-4-26b-a4b-it-maas"
