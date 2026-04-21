"""
Tests for xAI image generation (OpenAI-compatible /v1/images/generations).
"""

import os
from unittest.mock import patch

import litellm
from litellm.constants import XAI_API_BASE
from litellm.types.utils import ImageResponse


class TestXAIImageGeneration:
    def setup_method(self):
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

    def test_grok_imagine_image_in_model_map(self):
        from litellm.utils import get_model_info

        raw = litellm.model_cost.get("xai/grok-imagine-image")
        assert raw is not None
        assert raw.get("mode") == "image_generation"
        assert "/v1/images/generations" in raw.get("supported_endpoints", [])

        info = get_model_info("xai/grok-imagine-image")
        assert info is not None
        assert info.get("mode") == "image_generation"

    @patch("litellm.images.main.openai_chat_completions.image_generation")
    def test_image_generation_uses_xai_default_api_base(self, mock_image_gen):
        mock_image_gen.return_value = ImageResponse()
        litellm.image_generation(
            prompt="a red balloon",
            model="xai/grok-imagine-image",
            api_key="sk-xai-test",
            api_base=None,
        )
        kwargs = mock_image_gen.call_args.kwargs
        assert kwargs["api_base"] == XAI_API_BASE
        assert kwargs["api_key"] == "sk-xai-test"
        assert kwargs["model"] == "grok-imagine-image"
