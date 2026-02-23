import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.vertex_ai.image_generation import (
    get_vertex_ai_image_generation_config,
)
from litellm.llms.vertex_ai.image_generation.vertex_gemini_transformation import (
    VertexAIGeminiImageGenerationConfig,
)
from litellm.llms.vertex_ai.image_generation.vertex_imagen_transformation import (
    VertexAIImagenImageGenerationConfig,
)


class TestVertexAIGeminiImageGenerationConfig:
    def setup_method(self):
        """Set up test fixtures"""
        self.config = VertexAIGeminiImageGenerationConfig()

    def test_get_supported_openai_params(self):
        """Test get_supported_openai_params returns correct params"""
        supported = self.config.get_supported_openai_params("gemini-2.5-flash-image")
        assert "n" in supported
        assert "size" in supported

    def test_map_openai_params_n(self):
        """Test mapping n parameter to candidate_count"""
        non_default_params = {"n": 3}
        optional_params = {}
        result = self.config.map_openai_params(
            non_default_params, optional_params, "gemini-2.5-flash-image", False
        )
        assert result.get("candidate_count") == 3

    def test_map_openai_params_size(self):
        """Test mapping size parameter to aspectRatio"""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}
        result = self.config.map_openai_params(
            non_default_params, optional_params, "gemini-2.5-flash-image", False
        )
        assert result.get("aspectRatio") == "1:1"

    def test_map_openai_params_size_16_9(self):
        """Test mapping 16:9 size"""
        non_default_params = {"size": "1792x1024"}
        optional_params = {}
        result = self.config.map_openai_params(
            non_default_params, optional_params, "gemini-2.5-flash-image", False
        )
        assert result.get("aspectRatio") == "16:9"

    def test_map_size_to_aspect_ratio(self):
        """Test size to aspect ratio mapping"""
        assert self.config._map_size_to_aspect_ratio("1024x1024") == "1:1"
        assert self.config._map_size_to_aspect_ratio("1792x1024") == "16:9"
        assert self.config._map_size_to_aspect_ratio("1024x1792") == "9:16"
        assert self.config._map_size_to_aspect_ratio("1280x896") == "4:3"
        assert self.config._map_size_to_aspect_ratio("896x1280") == "3:4"
        assert self.config._map_size_to_aspect_ratio("unknown") == "1:1"  # default

    def test_transform_image_generation_request_basic(self):
        """Test basic request transformation"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "contents" in request
        assert "generationConfig" in request
        assert request["generationConfig"]["responseModalities"] == ["IMAGE"]
        assert request["contents"][0]["parts"][0]["text"] == "A nano banana"

    def test_transform_image_generation_request_with_aspect_ratio(self):
        """Test request transformation with aspectRatio"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={"aspectRatio": "16:9"},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"

    def test_transform_image_generation_request_with_image_size(self):
        """Test request transformation with imageSize (Gemini 3 Pro)"""
        request = self.config.transform_image_generation_request(
            model="gemini-3-pro-image-preview",
            prompt="A nano banana",
            optional_params={"imageSize": "4K"},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["imageConfig"]["imageSize"] == "4K"

    def test_transform_image_generation_request_with_candidate_count(self):
        """Test request transformation with candidate_count"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={"candidate_count": 2},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["candidateCount"] == 2

    def test_transform_image_generation_request_with_n(self):
        """Test request transformation with n parameter"""
        request = self.config.transform_image_generation_request(
            model="gemini-2.5-flash-image",
            prompt="A nano banana",
            optional_params={"n": 2},
            litellm_params={},
            headers={},
        )
        assert request["generationConfig"]["candidateCount"] == 2

    def test_transform_image_generation_response(self):
        """Test response transformation"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "base64_encoded_image_data",
                                }
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 93,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 54,
                    },
                    {
                        "modality": "IMAGE",
                        "tokenCount": 39,
                    }
                ],
                "candidatesTokenCount": 17,
                "totalTokenCount": 110,
            }
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="gemini-2.5-flash-image",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64_encoded_image_data"
        assert result.data[0].url is None
        assert result.usage.input_tokens == 93
        assert result.usage.input_tokens_details.text_tokens == 54
        assert result.usage.input_tokens_details.image_tokens == 39
        assert result.usage.output_tokens == 17
        assert result.usage.total_tokens == 110


    def test_transform_image_generation_response_multiple_images(self):
        """Test response transformation with multiple images"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "image1",
                                }
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "image2",
                                }
                            },
                        ]
                    }
                }
            ]
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="gemini-2.5-flash-image",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].b64_json == "image1"
        assert result.data[1].b64_json == "image2"

    def test_transform_image_generation_response_signature(self):
        """Test response transformation includes thoughtSignature for Gemini 3 Pro"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "base64_encoded_image_data",
                                },
                                "thoughtSignature": "test_signature_abc123",
                            }
                        ]
                    }
                }
            ]
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="gemini-3-pro-image-preview",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64_encoded_image_data"
        assert result.data[0].provider_specific_fields["thought_signature"] == "test_signature_abc123"


class TestVertexAIImagenImageGenerationConfig:
    def setup_method(self):
        """Set up test fixtures"""
        self.config = VertexAIImagenImageGenerationConfig()

    def test_get_supported_openai_params(self):
        """Test get_supported_openai_params returns correct params"""
        supported = self.config.get_supported_openai_params("imagegeneration@006")
        assert "n" in supported
        assert "size" in supported

    def test_map_openai_params_n(self):
        """Test mapping n parameter to sampleCount"""
        non_default_params = {"n": 3}
        optional_params = {}
        result = self.config.map_openai_params(
            non_default_params, optional_params, "imagegeneration@006", False
        )
        assert result.get("sampleCount") == 3

    def test_map_openai_params_size(self):
        """Test mapping size parameter to aspectRatio"""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}
        result = self.config.map_openai_params(
            non_default_params, optional_params, "imagegeneration@006", False
        )
        assert result.get("aspectRatio") == "1:1"

    def test_map_size_to_aspect_ratio(self):
        """Test size to aspect ratio mapping"""
        assert self.config._map_size_to_aspect_ratio("1024x1024") == "1:1"
        assert self.config._map_size_to_aspect_ratio("1792x1024") == "16:9"
        assert self.config._map_size_to_aspect_ratio("unknown") == "1:1"  # default

    def test_transform_image_generation_request_basic(self):
        """Test basic request transformation"""
        request = self.config.transform_image_generation_request(
            model="imagegeneration@006",
            prompt="A cat",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "instances" in request
        assert "parameters" in request
        assert request["instances"][0]["prompt"] == "A cat"
        assert request["parameters"]["sampleCount"] == 1

    def test_transform_image_generation_request_with_params(self):
        """Test request transformation with parameters"""
        request = self.config.transform_image_generation_request(
            model="imagegeneration@006",
            prompt="A cat",
            optional_params={"sampleCount": 2, "aspectRatio": "16:9"},
            litellm_params={},
            headers={},
        )
        assert request["parameters"]["sampleCount"] == 2
        assert request["parameters"]["aspectRatio"] == "16:9"

    def test_transform_image_generation_response(self):
        """Test response transformation"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"bytesBase64Encoded": "base64_encoded_image_data"}
            ]
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="imagegeneration@006",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64_encoded_image_data"
        assert result.data[0].url is None

    def test_transform_image_generation_response_multiple_images(self):
        """Test response transformation with multiple images"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"bytesBase64Encoded": "image1"},
                {"bytesBase64Encoded": "image2"},
            ]
        }
        mock_response.headers = {}

        from litellm.types.utils import ImageResponse

        model_response = ImageResponse()
        result = self.config.transform_image_generation_response(
            model="imagegeneration@006",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].b64_json == "image1"
        assert result.data[1].b64_json == "image2"


class TestGetVertexAIImageGenerationConfig:
    """Test the router function that selects the correct config"""

    def test_get_gemini_model_config(self):
        """Test that Gemini models return Gemini config"""
        config = get_vertex_ai_image_generation_config("gemini-2.5-flash-image")
        assert isinstance(config, VertexAIGeminiImageGenerationConfig)

        config = get_vertex_ai_image_generation_config("gemini-3-pro-image-preview")
        assert isinstance(config, VertexAIGeminiImageGenerationConfig)

        config = get_vertex_ai_image_generation_config(
            "vertex_ai/gemini-2.5-flash-image"
        )
        assert isinstance(config, VertexAIGeminiImageGenerationConfig)

    def test_get_imagen_model_config(self):
        """Test that Imagen models return Imagen config"""
        config = get_vertex_ai_image_generation_config("imagegeneration@006")
        assert isinstance(config, VertexAIImagenImageGenerationConfig)

        config = get_vertex_ai_image_generation_config("imagen-4.0-generate-001")
        assert isinstance(config, VertexAIImagenImageGenerationConfig)

        config = get_vertex_ai_image_generation_config(
            "vertex_ai/imagegeneration@006"
        )
        assert isinstance(config, VertexAIImagenImageGenerationConfig)

    def test_get_non_gemini_model_config(self):
        """Test that non-Gemini models default to Imagen config"""
        config = get_vertex_ai_image_generation_config("some-other-model")
        assert isinstance(config, VertexAIImagenImageGenerationConfig)


class TestVertexAIImageGenerationIntegration:
    """Integration tests for Vertex AI image generation"""

    @pytest.mark.skipif(
        not os.getenv("VERTEXAI_PROJECT"),
        reason="Vertex AI credentials not set",
    )
    def test_gemini_image_generation_config_validation(self):
        """Test that Gemini config can validate environment"""
        config = VertexAIGeminiImageGenerationConfig()
        with patch.object(
            config, "_resolve_vertex_project", return_value="test-project"
        ), patch.object(
            config, "_resolve_vertex_location", return_value="us-central1"
        ), patch.object(
            config, "_ensure_access_token", return_value=("token", None)
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
        with patch.object(
            config, "_resolve_vertex_project", return_value="test-project"
        ), patch.object(
            config, "_resolve_vertex_location", return_value="us-central1"
        ), patch.object(
            config, "_ensure_access_token", return_value=("token", None)
        ):
            headers = config.validate_environment(
                headers={},
                model="imagegeneration@006",
                messages=[],
                optional_params={},
                litellm_params={},
            )
            assert "Authorization" in headers

    def test_gemini_get_complete_url(self):
        """Test Gemini config URL generation"""
        config = VertexAIGeminiImageGenerationConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-2.5-flash-image",
            optional_params={},
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
            },
        )
        assert "test-project" in url
        assert "us-central1" in url
        assert "gemini-2.5-flash-image" in url
        assert "generateContent" in url

    def test_imagen_get_complete_url(self):
        """Test Imagen config URL generation"""
        config = VertexAIImagenImageGenerationConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="imagegeneration@006",
            optional_params={},
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
            },
        )
        assert "test-project" in url
        assert "us-central1" in url
        assert "imagegeneration@006" in url
        assert "predict" in url

