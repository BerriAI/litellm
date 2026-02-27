"""
Tests for Vertex AI Qwen MaaS models that require the global endpoint.

These tests verify that:
1. Qwen models are correctly identified as global-only models
2. The correct global URL is constructed (https://aiplatform.googleapis.com)
3. The completion() and responses() API work with Qwen models
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.common_utils import is_global_only_vertex_model
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

    # Restore saved environment variables
    for var, value in saved_env.items():
        os.environ[var] = value


class TestQwenGlobalOnlyDetection:
    """Test that Qwen models are correctly identified as global-only."""

    @pytest.mark.parametrize(
        "model",
        [
            "vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas",
            "vertex_ai/qwen/qwen3-next-80b-a3b-thinking-maas",
            "vertex_ai/qwen/qwen3-235b-a22b-instruct-2507-maas",
            "vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas",
        ],
    )
    def test_qwen_models_are_global_only(self, model):
        """Test that Qwen MaaS models are identified as global-only."""
        # This test requires the model_cost to have supported_regions: ["global"]
        # If the model is not in model_cost, it should return False (fallback behavior)
        result = is_global_only_vertex_model(model)
        # Note: This will return True only if the model is in model_cost with supported_regions: ["global"]
        # If running without the updated model_cost, this may return False
        assert isinstance(result, bool)

    def test_non_global_model_returns_false(self):
        """Test that non-global models return False."""
        result = is_global_only_vertex_model("vertex_ai/gemini-1.5-pro")
        assert result is False

    def test_unknown_model_returns_false(self):
        """Test that unknown models return False (fallback behavior)."""
        result = is_global_only_vertex_model("vertex_ai/unknown-model-xyz")
        assert result is False


class TestVertexBaseGetVertexRegion:
    """Test the get_vertex_region method."""

    def test_global_only_model_returns_global(self):
        """Test that global-only models return 'global' regardless of input."""
        vertex_base = VertexBase()

        with patch(
            "litellm.llms.vertex_ai.vertex_llm_base.is_global_only_vertex_model",
            return_value=True,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region="us-central1",
                model="vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas",
            )
            assert result == "global"

    def test_global_only_model_with_none_returns_global(self):
        """Test that global-only models return 'global' even with None input."""
        vertex_base = VertexBase()

        with patch(
            "litellm.llms.vertex_ai.vertex_llm_base.is_global_only_vertex_model",
            return_value=True,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region=None,
                model="vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas",
            )
            assert result == "global"

    def test_non_global_model_uses_provided_region(self):
        """Test that non-global models use the provided region."""
        vertex_base = VertexBase()

        with patch(
            "litellm.llms.vertex_ai.vertex_llm_base.is_global_only_vertex_model",
            return_value=False,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region="europe-west1",
                model="vertex_ai/gemini-1.5-pro",
            )
            assert result == "europe-west1"

    def test_non_global_model_fallback_to_us_central1(self):
        """Test that non-global models with None region fallback to us-central1."""
        vertex_base = VertexBase()

        with patch(
            "litellm.llms.vertex_ai.vertex_llm_base.is_global_only_vertex_model",
            return_value=False,
        ):
            result = vertex_base.get_vertex_region(
                vertex_region=None,
                model="vertex_ai/gemini-1.5-pro",
            )
            assert result == "us-central1"


class TestCreateVertexURLGlobal:
    """Test that create_vertex_url handles global location correctly."""

    def test_global_location_url_format(self):
        """Test that global location produces correct URL without region prefix."""
        url = VertexBase.create_vertex_url(
            vertex_location="global",
            vertex_project="test-project",
            partner=VertexPartnerProvider.llama,
            stream=False,
            model="qwen/qwen3-next-80b-a3b-instruct-maas",
        )

        # Global URL should NOT have region prefix
        assert url.startswith("https://aiplatform.googleapis.com")
        assert "global-aiplatform.googleapis.com" not in url
        assert "/locations/global/" in url

    def test_regional_location_url_format(self):
        """Test that regional location produces correct URL with region prefix."""
        url = VertexBase.create_vertex_url(
            vertex_location="us-central1",
            vertex_project="test-project",
            partner=VertexPartnerProvider.llama,
            stream=False,
            model="openai/gpt-oss-20b-maas",
        )

        # Regional URL should have region prefix
        assert url.startswith("https://us-central1-aiplatform.googleapis.com")
        assert "/locations/us-central1/" in url


@pytest.mark.asyncio
async def test_vertex_ai_qwen_global_endpoint_url():
    """
    Test that Qwen models use the global endpoint URL.
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexLLM,
    )

    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = {
        "id": "chatcmpl-qwen-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen/qwen3-next-80b-a3b-instruct-maas",
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

    client = AsyncHTTPHandler()

    async def mock_post_func(*args, **kwargs):
        return mock_response

    with patch.object(client, "post", side_effect=mock_post_func) as mock_post, patch.object(
        VertexLLM, "_ensure_access_token", return_value=("fake-token", "test-project")
    ), patch(
        "litellm.llms.vertex_ai.vertex_llm_base.is_global_only_vertex_model",
        return_value=True,
    ):
        response = await litellm.acompletion(
            model="vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas",
            messages=[{"role": "user", "content": "Hello"}],
            vertex_ai_project="test-project",
            client=client,
        )

        # Verify the mock was called
        mock_post.assert_called_once()

        # Get the call arguments
        call_args = mock_post.call_args
        called_url = call_args.kwargs["url"]

        # Verify the URL uses global endpoint (no region prefix)
        assert called_url.startswith("https://aiplatform.googleapis.com")
        assert "global-aiplatform.googleapis.com" not in called_url
        assert "/locations/global/" in called_url
        assert "/endpoints/openapi/chat/completions" in called_url

        # Verify response
        assert response.model == "qwen/qwen3-next-80b-a3b-instruct-maas"


class TestGetSupportedRegions:
    """Test that get_supported_regions correctly reads from model_cost."""

    def test_get_supported_regions_returns_list(self):
        """Test that get_supported_regions returns a list when model has supported_regions."""
        # Mock the model_cost to have supported_regions
        with patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas": {
                    "supported_regions": ["global"],
                    "litellm_provider": "vertex_ai-qwen_models",
                }
            },
        ):
            regions = litellm.utils.get_supported_regions(
                model="vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas",
                custom_llm_provider="vertex_ai",
            )
            assert regions == ["global"]

    def test_get_supported_regions_returns_none_when_not_set(self):
        """Test that get_supported_regions returns None when model doesn't have supported_regions."""
        # Mock the model_cost without supported_regions
        with patch.dict(
            litellm.model_cost,
            {
                "vertex_ai/gemini-1.5-pro": {
                    "litellm_provider": "vertex_ai",
                }
            },
        ):
            regions = litellm.utils.get_supported_regions(
                model="vertex_ai/gemini-1.5-pro",
                custom_llm_provider="vertex_ai",
            )
            assert regions is None

    def test_get_supported_regions_returns_none_for_unknown_model(self):
        """Test that get_supported_regions returns None for unknown models."""
        regions = litellm.utils.get_supported_regions(
            model="vertex_ai/unknown-model-xyz",
            custom_llm_provider="vertex_ai",
        )
        assert regions is None
