import os
from unittest.mock import patch

import pytest


from litellm.llms.vertex_ai.image_generation.vertex_gemini_transformation import (
    VertexAIGeminiImageGenerationConfig,
)
from litellm.llms.vertex_ai.image_generation.vertex_imagen_transformation import (
    VertexAIImagenImageGenerationConfig,
)


class TestVertexAIImageGenerationIntegration:
    """Integration tests for Vertex AI image generation"""

    @pytest.mark.skipif(
        not os.getenv("VERTEXAI_PROJECT"),
        reason="Vertex AI credentials not set",
    )
    def test_gemini_image_generation_config_validation(self):
        """Test that Gemini config can validate environment"""
        config = VertexAIGeminiImageGenerationConfig()
        with (
            patch.object(config, "_resolve_vertex_project", return_value="test-project"),
            patch.object(config, "_resolve_vertex_location", return_value="us-central1"),
            patch.object(config, "_ensure_access_token", return_value=("token", None)),
        ):
            headers = config.validate_environment(
                headers={},
                model="gemini-2.5-flash-image",
                messages=[],
                optional_params={},
                litellm_params={},
            )
            assert "Authorization" in headers

    @pytest.mark.skipif(
        not os.getenv("VERTEXAI_PROJECT"),
        reason="Vertex AI credentials not set",
    )
    def test_imagen_image_generation_config_validation(self):
        """Test that Imagen config can validate environment"""
        config = VertexAIImagenImageGenerationConfig()
        with (
            patch.object(config, "_resolve_vertex_project", return_value="test-project"),
            patch.object(config, "_resolve_vertex_location", return_value="us-central1"),
            patch.object(config, "_ensure_access_token", return_value=("token", None)),
        ):
            headers = config.validate_environment(
                headers={},
                model="imagegeneration@006",
                messages=[],
                optional_params={},
                litellm_params={},
            )
            assert "Authorization" in headers
